# GeGraphExecutor 最终设计方案

## 1. 架构概述

### 1.1 设计目标

为 xLLM 框架新增 `GeGraphExecutorImpl`，使用华为 Ascend GE V2 接口执行 epair 模型，实现：
- **继承设计**：EpModel 继承 CausalLM，符合模型抽象
- **简洁架构**：Model 与 Graph 直接绑定，无需单例管理
- **可扩展性**：支持多种模型类型（LLM/VLM/Rec）
- **性能优化**：避免 H2D 拷贝，复用 Worker Stream

### 1.2 核心架构

```
执行链路: Worker -> Engine -> GraphExecutor -> EpModel.forward() -> EpairModelLoader.RunModelWithStreamAsync

┌─────────────────────────────────────────────────────────────┐
│                 ExecutorImpl 接口（固定）                     │
│  run(tokens, positions, kv_caches, params)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               GeGraphExecutorImpl                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  run(): 实现 ExecutorImpl 接口                       │   │
│  │    └─> model_->forward(tokens, positions, ...)      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      EpModel : CausalLM                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  成员变量: EpairModelLoader loader_                  │   │
│  │                                                      │   │
│  │  load_model(loader):                                 │   │
│  │    ├─> GEInitializeV2()（进程级别，首次调用时初始化） │   │
│  │    ├─> 加载 epair 模型                               │   │
│  │    └─> CompileAndLoad()                              │   │
│  │                                                      │   │
│  │  forward(tokens, positions, kv_caches, params):      │   │
│  │    ├─> 按 input_names_ 从数据源构造图输入            │   │
│  │    ├─> RunModelWithStreamAsync()                     │   │
│  │    └─> 返回 ModelOutput                              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**使用流程**：
```
1. 创建 EpModel（继承 CausalLM）
2. 调用 load_model() 加载 epair 模型（内部完成 GE 初始化）
3. 将 EpModel 传入 GeGraphExecutorImpl
4. Executor 调用 model_->forward() 执行推理
```

**设计要点**：
- EpModel 自己管理 EpairModelLoader，Model 与 Graph 绑定
- GE 初始化在 load_model() 中完成（首次调用时初始化，使用静态变量保证进程级别）
- 不需要单例管理图，每个 EpModel 管理自己的图资源

## 2. 核心设计决策

### 2.1 EpModel 管理 epair 模型

**设计目标**：
- `EpModel` 直接继承 `CausalLM`（通用设计，不限 REC 模型）
- 使用 GE 图执行方式，而非传统的 PyTorch 模型执行
- `EpairModelLoader` 通过 `std::unique_ptr` 管理（无默认构造函数）
- 在 `load_model()` 中完成模型加载和编译
- 通过 `forward()` 函数执行模型推理
- KV Cache 由图内部原地更新，Host 不管理

**EpairModelLoader 对外 API**（`torch_delegate/epair_model_loader.h`）：
```cpp
namespace td {
class EpairModelLoader {
 public:
  explicit EpairModelLoader(const char *archive_path);  // 解析 epair + 构建 GE Graph，不编译
  Status CompileAndLoad(const std::map<ge::AscendString, ge::AscendString> &options, void *stream);
  Status RunModelWithStreamAsync(void *stream, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);
  Status GetGEGraph(ge::Graph *graph) const;  // 获取 GE Graph 副本
};
}
```

**架构设计**：
```cpp
class EpModel : public CausalLM {
public:
    EpModel(const torch::TensorOptions& options);
    ~EpModel() override;
    
    ModelOutput forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params) override;
    
    torch::Tensor logits(const torch::Tensor& hidden_states,
                         const torch::Tensor& selected_idxes) override;
    
    void load_model(std::unique_ptr<ModelLoader> loader) override;
    torch::Device device() const override;
    const torch::TensorOptions& options() const override;
    
    void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
    void update_expert_weight(int32_t) override {}
    
    bool IsInitialized() const { return initialized_; }
    
private:
    std::unique_ptr<td::EpairModelLoader> loader_;  // unique_ptr 管理
    torch::TensorOptions options_;
    uint64_t device_id_ = 0;
    bool initialized_ = false;
    
    // 图输入/输出有序名字列表（从 GetGEGraph() 解析，按 index 排序）
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    // 预分配的输出 tensors（固定 shape 时复用）
    std::vector<gert::Tensor> preallocated_outputs_;
    std::vector<void*> output_dev_mems_;          // 需要 aclrtFree 的内存
    
    // 辅助方法
    void ParseGraphIO();                           // 从 GetGEGraph() 解析输入输出
    std::vector<gert::Tensor> BuildGraphInputs(
        const torch::Tensor& tokens, const torch::Tensor& positions,
        std::vector<KVCache>& kv_caches, const ModelInputParams& params);
    ModelOutput ConvertOutputs(std::vector<gert::Tensor>& device_outputs);
};
```

**优点**：
- ✅ 继承设计：EpModel 继承 CausalLM，可无缝接入 Executor
- ✅ 模型无关：不限 REC/LLM/VLM，任何 epair 模型均可加载
- ✅ 封装性好：模型加载、编译、执行统一管理
- ✅ 职责清晰：EpModel 负责模型执行，Executor 负责调度
- ✅ 零侵入：ModelInputParams 和 Pipeline 层无需任何改动

### 2.2 图输入映射设计

**问题**：
- 不同模型的 GE 图输入节点命名和顺序可能不同
- `RunModelWithStreamAsync` 要求按精确顺序传入 `vector<gert::Tensor>`
- 需要将 xLLM 运行时数据映射到图的输入节点

**核心洞察**：数据已经在正确的位置了，不需要额外的映射表。

| 输入类型 | 数据来源 | 已有路径 |
|---|---|---|
| tokens | `forward()` 函数参数 | 直接可用 |
| positions | `forward()` 函数参数 | 直接可用 |
| kv_caches | `forward()` 函数参数 | 直接可用 |
| 自定义输入（如 sparse_embedding） | `params.multimodal.mm_data` | MMData 本身就是 `string → torch::Tensor` 的 map，已有名字 |

**解决方案**：EpModel 内部维护一个**有序名字列表**，`BuildGraphInputs()` 按名字从对应数据源取数据。

```cpp
class EpModel : public CausalLM {
    // load_model() 后从 GetGEGraph() 解析得到，按 index 排序
    std::vector<std::string> input_names_;   // 如 ["input_ids", "k_cache[0]", "v_cache[0]", "sparse_embedding", ...]
    std::vector<std::string> output_names_;  // 如 ["beam_sequence_group", "out_logprobs"]
};
```

**BuildGraphInputs() 实现**：
```cpp
std::vector<gert::Tensor> EpModel::BuildGraphInputs(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
    
    std::vector<gert::Tensor> inputs;
    for (const auto& name : input_names_) {
        if (IsTokenIds(name)) {
            inputs.push_back(TorchToGe(tokens));
        } else if (IsPositions(name)) {
            inputs.push_back(TorchToGe(positions));
        } else if (auto kv = ParseKVCacheName(name)) {
            inputs.push_back(TorchToGe(GetKVCacheTensor(kv_caches, *kv)));
        } else {
            // 自定义输入：从 MMData 按名字取
            inputs.push_back(TorchToGe(params.multimodal.mm_data.get(name)));
        }
    }
    return inputs;
}
```

**优点**：
- ✅ **ModelInputParams 零改动**：不新增任何字段
- ✅ **Pipeline 零改动**：不需要感知图的结构，标准输入走函数参数，自定义输入走 MMData（现有流程已覆盖）
- ✅ **职责归位**：名字匹配是 EpModel 的事，Pipeline 层不感知图结构
- ✅ **简洁**：一个 `vector<string>` + 一个循环，约 30 行代码

### 2.3 KVCache 原地更新机制

**核心设计**：GE 图模式下 KV Cache 由**图内部原地更新**，Host 不管理 paged attention / block table。

**机制**：
- KV Cache tensor 作为图输入传入（EpModel 内部从 `kv_caches` 参数按名字取）
- Graph 内部算子直接修改这些 tensor 的 device memory 内容
- `gert::Tensor` 引用 device memory 指针，Graph 内部原地写入，无需额外输出 tensor
- Host 侧不需要 `StoppingChecker`、不需要逐步 `append_token()`

**实现机制**：
```cpp
// EpModel::BuildGraphInputs() 内部，按 input_names_ 中的 KV Cache 名字从 kv_caches 取
// 例如 input_names_ = [..., "k_cache[0]", "v_cache[0]", ...]
// EpModel 解析名字得到 layer_index，从 kv_caches[layer_index] 取对应 tensor
// 转为 gert::Tensor 后传入图
// gert::Tensor 的 addr 指向 torch::Tensor 的 data_ptr()（零拷贝引用）
// Graph 执行后 KV Cache tensor 内容已被原地更新
```

**关键点**：
- **调用方职责**：不处理 KV Cache 更新逻辑，KV Cache tensor 通过 `forward()` 参数传入
- **EpModel 职责**：将 torch::Tensor 转为 gert::Tensor（引用 device memory），传入图执行
- **Graph 职责**：内部算子原地修改 KV Cache tensor 内容
- **内存管理**：KV Cache tensor 由 xLLM 的 KVCacheManager 管理，EpModel 不持有所有权
- **性能优化**：gert::Tensor 直接引用 torch::Tensor 的 device memory，零拷贝

### 2.4 多卡场景架构

**设计决策**：
- **GEInitializeV2**：进程级别，在 EpModel::load_model() 中首次调用时初始化（使用静态变量）
- **EpModel**：每个 device 独立的 EpModel 实例，由调用方创建
- **EpairModelLoader**：通过 `session_device_id` 指定真正的 deviceId

**实现**：
```cpp
void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
    // 1. GE 初始化（进程级别，只初始化一次）
    static bool ge_initialized = false;
    static std::mutex ge_mutex;
    
    if (!ge_initialized) {
        std::lock_guard<std::mutex> lock(ge_mutex);
        if (!ge_initialized) {
            if (ge::GEInitializeV2() != ge::SUCCESS) {
                LOG(ERROR) << "Failed to initialize GE";
                return;
            }
            ge_initialized = true;
        }
    }
    
    // 2. 加载 epair 文件
    // 3. CompileAndLoad(device_id_, nullptr)
}
```

**架构图**：
```
进程级别（静态变量）
  │
  └─> GEInitializeV2()  // 首次调用时初始化

每个 Worker/Engine
  │
  ├─> 创建 EpModel(device_id=0)
   │    └─> load_model() -> loader_->CompileAndLoad(options, nullptr)
  │
  ├─> 创建 EpModel(device_id=1)
   │    └─> load_model() -> loader_->CompileAndLoad(options, nullptr)
  │
  └─> 创建 EpModel(device_id=N)
        └─> load_model() -> loader_->CompileAndLoad(options, nullptr)

执行链路
Worker -> Engine -> GraphExecutor
                    └─> EpModel.forward()
                          └─> loader_->RunModelWithStreamAsync()
```

### 2.5 Stream 管理

**设计决策**：
- **复用 Worker Stream**：通过 `c10_npu::getCurrentNPUStream(device_id)` 获取
- **Init 阶段**：传 `nullptr`（编译阶段与运行时流无关）
- **forward 阶段**：传入真正的 stream（执行阶段）

**流程**：
```cpp
// Worker 执行前设置 stream
c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();

// EpModel::load_model() - 编译阶段
void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
    std::string epair_path = loader->model_weights_path() + "/model.epair";
    // 构造 EpairModelLoader（解析 epair + 构建 GE Graph）
    loader_ = std::make_unique<td::EpairModelLoader>(epair_path.c_str());
    // 编译（stream 传 nullptr）
    std::map<ge::AscendString, ge::AscendString> options = {
        {ge::AscendString("ge.session_device_id"),
         ge::AscendString(std::to_string(device_id_).c_str())}
    };
    loader_->CompileAndLoad(options, nullptr);
}

// EpModel::forward() - 执行阶段
ModelOutput EpModel::forward(const ModelInputParams& params) {
    // 获取当前 thread 的 stream
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    
    // 执行 Graph
    loader_->RunModelWithStreamAsync(stream, inputs, outputs);
}
```

### 2.6 Tensor 桥接与内存管理

#### 2.6.1 桥接方案概述

`RunModelWithStreamAsync` 的输入输出都是 `std::vector<gert::Tensor>`，xLLM 内部全部用 `torch::Tensor`。采用**拷贝方式**进行桥接。

| 方向 | 场景 | 方式 |
|---|---|---|
| torch → gert | 图输入（tokens, positions, kv_cache 等） | aclrtMemcpy D2D 拷贝 |
| gert → torch | 图输出（beam_sequence_group 等） | aclrtMemcpy D2D 拷贝 |

#### 2.6.2 输入方向：torch::Tensor → gert::Tensor

```cpp
// 在 EpModel::forward() 内部，按 input_names_ 从数据源构造 gert::Tensor vector
static gert::Tensor TorchToGe(const torch::Tensor& t) {
    gert::Tensor ge;
    
    // 1. 设置 shape
    gert::StorageShape shape;
    for (auto dim : t.sizes()) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    ge.GetShape() = shape;
    
    // 2. 设置 format 和 dtype
    ge.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    ge.SetDataType(TorchDtypeToGe(t.scalar_type()));
    
    // 3. 分配 device 内存并拷贝
    size_t bytes = t.nbytes();
    void* dev_ptr = nullptr;
    aclrtMalloc(&dev_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dev_ptr, bytes, t.data_ptr(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    
    // 4. 绑定（manager=nullptr，调用方负责释放 dev_ptr）
    ge.SetData(gert::TensorData(dev_ptr, nullptr, bytes, gert::kOnDeviceHbm));
    return ge;
}

// dtype 映射表
static ge::DataType TorchDtypeToGe(torch::ScalarType st) {
    switch (st) {
        case torch::kFloat32: return ge::DT_FLOAT;
        case torch::kFloat16: return ge::DT_FLOAT16;
        case torch::kBFloat16: return ge::DT_BF16;
        case torch::kInt32:   return ge::DT_INT32;
        case torch::kInt64:   return ge::DT_INT64;
        case torch::kBool:    return ge::DT_BOOL;
        default: CHECK(false) << "Unsupported dtype: " << st;
    }
}
```

**输入内存管理**：每次 forward 分配的 `dev_ptr` 需要在推理完成后 `aclrtFree`。

#### 2.6.3 输出方向：gert::Tensor → torch::Tensor

```cpp
static torch::Tensor GeToTorch(const gert::Tensor& ge, torch::Device device) {
    // 1. 从 gert::Tensor 读 shape 和 dtype
    auto& origin = ge.GetShape().GetOriginShape();
    std::vector<int64_t> dims(origin.GetDimNum());
    for (size_t i = 0; i < origin.GetDimNum(); ++i) dims[i] = origin.GetDim(i);
    
    auto options = torch::TensorOptions()
        .device(device)
        .dtype(GeDtypeToTorch(ge.GetDataType()));
    
    // 2. 创建 torch::Tensor（device 上）
    torch::Tensor out = torch::empty(dims, options);
    
    // 3. 拷贝数据 D2D
    aclrtMemcpy(out.data_ptr(), out.nbytes(),
                ge.GetAddr(), out.nbytes(),
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    return out;
}
```

#### 2.6.4 输出预分配策略

`RunModelWithStreamAsync` 要求调用方传入预分配的 `std::vector<gert::Tensor>& outputs`。

**策略**：在 `load_model()` 阶段从 `GetGEGraph()` 解析输出 TensorDesc，区分固定/动态 shape。

`ParseGraphIO()` 遍历 Data 节点读 `"index"` 属性得到有序 `input_names_`，遍历 NetOutput 节点得到 `output_names_` 和输出 TensorDesc（shape/dtype）。详细实现见 10.12.5 节。

**预分配逻辑**：
- 固定 shape 输出：`load_model()` 时预分配 `preallocated_outputs_`，每次 forward 复用
- 动态 shape 输出：每次 forward 前根据实际 batch 计算 shape，重新分配

#### 2.6.5 内存生命周期总结

| 对象 | 分配时机 | 释放时机 | 分配方式 |
|---|---|---|---|
| 输入 gert::Tensor dev_ptr | 每次 forward() | forward() 返回前 | aclrtMalloc |
| 输出 gert::Tensor dev_ptr（固定 shape） | load_model() | ~EpModel() | aclrtMalloc |
| 输出 gert::Tensor dev_ptr（动态 shape） | 每次 forward() | forward() 返回前 | aclrtMalloc |
| KV Cache tensor | KVCacheManager 管理 | KVCacheManager 管理 | torch::Tensor |

## 3. 类设计

### 3.1 EpModel（继承 CausalLM）

**设计思路**：
- `EpModel` 直接继承 `CausalLM`（通用设计，不限模型类型）
- 使用 GE 图执行方式，而非传统的 PyTorch 模型执行
- `EpairModelLoader` 通过 `std::unique_ptr` 管理（无默认构造函数）
- 图输入/输出信息通过 `GetGEGraph()` 解析 GE Graph 获取

**职责**：
- 继承 `CausalLM` 接口，作为模型类使用
- 管理 epair 模型的生命周期（构造 → 编译 → 执行 → 析构）
- 实现 `forward()` 接口，通过 GE 图执行推理
- 实现 `load_model()` 接口，加载和编译 epair 模型
- 内部维护图输入/输出的有序名字列表，完成数据源到图输入的映射

**接口**：
```cpp
class EpModel : public CausalLM {
public:
    EpModel(const torch::TensorOptions& options);
    ~EpModel() override;
    
    ModelOutput forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params) override;
    
    torch::Tensor logits(const torch::Tensor& hidden_states,
                         const torch::Tensor& selected_idxes) override;
    
    void load_model(std::unique_ptr<ModelLoader> loader) override;
    torch::Device device() const override;
    const torch::TensorOptions& options() const override;
    
    void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
    void update_expert_weight(int32_t) override {}
    
    bool IsInitialized() const { return initialized_; }
    
private:
    void ParseGraphIO();
    std::vector<gert::Tensor> BuildGraphInputs(
        const torch::Tensor& tokens,
        const torch::Tensor& positions,
        std::vector<KVCache>& kv_caches,
        const ModelInputParams& params);
    ModelOutput ConvertOutputs(std::vector<gert::Tensor>& device_outputs);
    
    std::unique_ptr<td::EpairModelLoader> loader_;
    torch::TensorOptions options_;
    uint64_t device_id_ = 0;
    bool initialized_ = false;
    
    // 图输入/输出有序名字列表（从 GetGEGraph() 解析，按 index 排序）
    std::vector<std::string> input_names_;   // 如 ["input_ids", "k_cache[0]", "v_cache[0]", "sparse_embedding"]
    std::vector<std::string> output_names_;  // 如 ["beam_sequence_group", "out_logprobs"]
    
    // 预分配的输出 tensors（固定 shape 时复用）
    std::vector<gert::Tensor> preallocated_outputs_;
    std::vector<void*> output_dev_mems_;
};
```

**核心实现**：
```cpp
EpModel::EpModel(const torch::TensorOptions& options)
    : options_(options) {
    device_id_ = options.device().index();
}

EpModel::~EpModel() {
    for (auto* ptr : output_dev_mems_) {
        if (ptr) aclrtFree(ptr);
    }
}

void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
    // 1. GE 初始化（进程级别，只初始化一次）
    static std::once_flag ge_init_flag;
    std::call_once(ge_init_flag, []() {
        std::map<ge::AscendString, ge::AscendString> opts;
        ge::GEInitializeV2(opts);
    });
    
    // 2. 从 model_weights_path() 拼接 epair 路径
    std::string epair_path = loader->model_weights_path() + "/model.epair";
    
    // 3. 构造 EpairModelLoader（解析 epair + 构建 GE Graph）
    loader_ = std::make_unique<td::EpairModelLoader>(epair_path.c_str());
    
    // 4. 编译和加载（编译阶段 stream 传 nullptr）
    std::map<ge::AscendString, ge::AscendString> options = {
        {ge::AscendString("ge.session_device_id"),
         ge::AscendString(std::to_string(device_id_).c_str())}
    };
    if (loader_->CompileAndLoad(options, nullptr) != td::SUCCESS) {
        LOG(ERROR) << "Failed to compile and load model: " << epair_path;
        return;
    }
    
    // 5. 从 GetGEGraph() 解析图输入/输出名字（按 index 排序）
    ParseGraphIO();
    
    // 6. 预分配固定 shape 的输出 tensors
    PreallocateFixedOutputs();
    
    initialized_ = true;
}

ModelOutput EpModel::forward(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& params) {
    if (!initialized_) {
        LOG(ERROR) << "EpModel not initialized";
        return ModelOutput();
    }
    
    // 1. 按 input_names_ 顺序从数据源构造 gert::Tensor 输入
    std::vector<gert::Tensor> graph_inputs = BuildGraphInputs(tokens, positions, kv_caches, params);
    
    // 2. 获取 Worker stream
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_id_).stream();
    
    // 3. 准备输出 tensors（固定 shape 复用预分配，动态 shape 按需分配）
    std::vector<gert::Tensor> device_outputs = PrepareOutputs();
    
    // 4. 执行 Graph
    auto status = loader_->RunModelWithStreamAsync(stream, graph_inputs, device_outputs);
    if (status != td::SUCCESS) {
        LOG(ERROR) << "RunModelWithStreamAsync failed: " << status;
    }
    
    // 5. 同步
    aclrtSynchronizeStream(stream);
    
    // 6. 转换输出 gert::Tensor → torch::Tensor
    ModelOutput result = ConvertOutputs(device_outputs);
    
    // 7. 清理输入 gert::Tensor 的临时内存
    for (auto& t : graph_inputs) {
        if (t.GetAddr()) aclrtFree(const_cast<void*>(t.GetAddr()));
    }
    
    return result;
}

std::vector<gert::Tensor> EpModel::BuildGraphInputs(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
    
    std::vector<gert::Tensor> inputs;
    inputs.reserve(input_names_.size());
    
    for (const auto& name : input_names_) {
        if (IsTokenIds(name)) {
            inputs.push_back(TorchToGe(tokens));
        } else if (IsPositions(name)) {
            inputs.push_back(TorchToGe(positions));
        } else if (auto kv = ParseKVCacheName(name)) {
            inputs.push_back(TorchToGe(GetKVCacheTensor(kv_caches, *kv)));
        } else {
            // 自定义输入：从 MMData 按名字取
            auto tensor = params.multimodal.mm_data.get(name);
            CHECK(tensor.defined()) << "Missing input tensor in MMData: " << name;
            inputs.push_back(TorchToGe(tensor));
        }
    }
    return inputs;
}
```

### 3.2 ModelInputParams

**无需修改**。GE 图模式复用现有字段：
- 标准输入（tokens、positions、kv_caches）通过 `forward()` 函数参数传入
- 自定义输入（如 sparse_embedding）通过 `params.multimodal.mm_data` 传入（MMData 本身就是 `string → torch::Tensor` 的 map）
- EpModel 内部按 `input_names_` 从对应数据源取数据，不需要额外的映射表

### 3.3 GeGraphExecutorImpl

**设计思路**：
- EpModel 继承 CausalLM，可以作为模型类传入 Executor
- Executor 持有 CausalLM 指针（实际是 EpModel）
- 直接调用 `model_->forward()` 执行推理

**职责**：
- 实现 `ExecutorImpl::run()` 接口
- 持有 `CausalLM`（实际是 `EpModel`）实例
- 调用 `model_->forward()` 执行推理

**接口**：
```cpp
class GeGraphExecutorImpl : public ExecutorImpl {
public:
    GeGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);
    
    ~GeGraphExecutorImpl() override;
    
    ForwardInput prepare_inputs(Batch& batch) override;
    
    ModelOutput run(const torch::Tensor& tokens,
                   const torch::Tensor& positions,
                   std::vector<KVCache>& kv_caches,
                   const ModelInputParams& params) override;
    
private:
    CausalLM* model_;  // 不拥有，由 Worker 管理生命周期
    uint64_t device_id_;
    bool initialized_ = false;
    
    ModelArgs args_;
    torch::Device device_;
    runtime::Options options_;
};

REGISTER_EXECUTOR("ge", GeGraphExecutorImpl);
```

**核心实现**：
```cpp
GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
    
    if (model == nullptr) {
        LOG(ERROR) << "GeGraphExecutorImpl requires non-null model";
        return;
    }
    
    auto* ep_model = dynamic_cast<EpModel*>(model);
    if (ep_model == nullptr || !ep_model->IsInitialized()) {
        LOG(ERROR) << "Model must be an initialized EpModel";
        return;
    }
    
    device_id_ = device.index();
    initialized_ = true;
}

ModelOutput GeGraphExecutorImpl::run(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
    if (!initialized_) return ModelOutput();
    return model_->forward(tokens, positions, kv_caches, params);
}
```

## 4. 文件结构

```
xllm/core/framework/model/
├── ep_model.h                       # EpModel 类定义（继承 CausalLM）
├── ep_model.cpp                     # EpModel 实现
├── ge_tensor_utils.h                # TorchToGe / GeToTorch / dtype 映射工具
├── ge_tensor_utils.h                # TorchToGe / GeToTorch / TorchDtypeToGe 桥接工具
└── causal_lm.h                      # CausalLM 基类（已存在）

xllm/core/runtime/
├── ge_graph_executor_impl.h        # GeGraphExecutorImpl 定义
├── ge_graph_executor_impl.cpp      # GeGraphExecutorImpl 实现
└── executor_impl_factory.h         # 添加 REGISTER_EXECUTOR("ge", GeGraphExecutorImpl)
```

## 5. 依赖项

### 5.1 外部库

| 库 | 来源 | 用途 |
|---|---|---|
| `torch_delegate_backend` | torch-delegate 仓 | `td::EpairModelLoader`（epair 加载、编译、执行） |
| `graph` / `graph_base` | CANN SDK | `ge::Graph`、`ge::GNode`、`ge::TensorDesc` |
| `metadef` | CANN SDK | GE 算子定义、IR 注册 |
| `ge_runner_v2` | CANN SDK | `ge::GeSession`、`GEInitializeV2` |
| `acl_rt` | CANN SDK | `aclrtMalloc`、`aclrtMemcpy`、`aclrtStream` |
| `torch_npu` | PyTorch NPU 后端 | `c10_npu::getCurrentNPUStream` |

### 5.2 头文件（仅 EpModel 实现文件需要）
```cpp
#include "torch_delegate/epair_model_loader.h"  // td::EpairModelLoader
#include "torch_delegate/td_types.h"            // td::Status
#include <ge/ge_api_v2.h>                        // GEInitializeV2 / GEFinalizeV2
#include <graph/graph.h>                         // ge::Graph
#include <exe_graph/runtime/tensor.h>            // gert::Tensor
#include <acl/acl_rt.h>                          // aclrtMalloc / aclrtMemcpy
#include <torch_npu/csrc/core/npu/NPUStream.h>   // NPU Stream
```

**注意**：`model_input_params.h` 等公共头文件**不引入** CANN GE 头文件，`gert::Tensor` 仅在 `ep_model.cpp` 内部使用。

### 5.3 CMake 集成

#### 5.3.1 集成方式

torch-delegate 作为**可选依赖**，仅在 `USE_NPU=ON` 且显式启用时链接：

```cmake
# 顶层 CMakeLists.txt
option(USE_TORCH_DELEGATE "Enable torch-delegate GE graph executor" OFF)

if(USE_NPU AND USE_TORCH_DELEGATE)
  add_definitions(-DUSE_TORCH_DELEGATE)
  
  # 方式 A：find_package（推荐，torch-delegate 已安装）
  if(DEFINED ENV{TORCH_DELEGATE_ROOT})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_DELEGATE_ROOT}")
    find_package(TorchDelegate REQUIRED)
  else()
    message(FATAL_ERROR "USE_TORCH_DELEGATE=ON but TORCH_DELEGATE_ROOT not set")
  endif()
endif()
```

```cmake
# xllm/CMakeLists.txt（链接阶段）
if(USE_TORCH_DELEGATE)
  target_link_libraries(xllm PUBLIC TorchDelegate::torch_delegate_backend)
endif()
```

#### 5.3.2 需要关注的编译链接问题

**问题 1：`_GLIBCXX_USE_CXX11_ABI` 兼容性**

| 组件 | ABI 设置 | 来源 |
|------|----------|------|
| torch-delegate `libtorch_delegate_backend.so` | `_GLIBCXX_USE_CXX11_ABI=1` | `csrc/CMakeLists.txt:53` 硬编码 |
| xLLM 主程序 | 取决于 libtorch 编译选项 | 通常 ABI=1（PyTorch 默认） |
| Mooncake `ascend_transport` | `_GLIBCXX_USE_CXX11_ABI=0` | 独立编译 |

**风险**：如果 xLLM 使用的 libtorch 是 ABI=0 编译的，与 torch-delegate（ABI=1）链接会导致 `std::string` / `std::list` 等 STL 类型 ABI 不兼容，运行时 crash。

**验证方式**：
```bash
# 检查 xLLM 使用的 libtorch 的 ABI 设置
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

**解决方式**：确保 torch-delegate 和 xLLM 使用相同 ABI 的 libtorch 编译。如果 xLLM 使用 ABI=0，需要重新编译 torch-delegate 并修改 `csrc/CMakeLists.txt` 中的 ABI 定义。

**问题 2：`libruntime.so` 符号冲突**

xLLM 自身有 `libruntime.a`（静态库），同时 CANN 也有 `libruntime.so`（动态库）。xLLM 已经通过 `find_library(NPU_RUNTIME_SO ...)` 显式查找 CANN 的 `libruntime.so` 来避免冲突（`CMakeLists.txt:437-446`）。

torch-delegate 通过 `find_dependency(runtime)` 也会引入 CANN 的 `libruntime.so`。

**风险**：如果 `find_dependency(runtime)` 找到的路径与 xLLM 的 `NPU_RUNTIME_SO` 不一致，可能导致符号重复或版本不匹配。

**解决方式**：确保 `TORCH_DELEGATE_ROOT` 中的 `Findruntime.cmake` 模块与 xLLM 使用相同的 CANN 安装路径（`$ENV{ASCEND_TOOLKIT_HOME}`）。

**问题 3：`libtorch_cpu.so` 共享依赖**

torch-delegate 依赖 `libtorch_cpu`（用于 `caffe2::serialize::PyTorchStreamReader` 读取 epair 归档）。xLLM 通过 `torch_npu` 也间接依赖 libtorch。

**风险**：如果 torch-delegate 和 xLLM 链接不同版本的 `libtorch_cpu.so`，会导致 ODR 违规和运行时 crash。

**解决方式**：确保 torch-delegate 编译时使用的 `Torch_DIR` 与 xLLM 使用的 `PYTORCH_INSTALL_PATH` 指向同一个 PyTorch 安装。

**问题 4：头文件搜索顺序**

torch-delegate 的公共头文件依赖 CANN 头文件（`graph/graph.h`、`exe_graph/runtime/tensor.h`）。xLLM 已经通过 `include_directories(SYSTEM ...)` 引入了 CANN 头文件路径。

**解决方式**：torch-delegate 的头文件也应以 `SYSTEM` 方式引入，避免警告：
```cmake
if(USE_TORCH_DELEGATE)
  include_directories(SYSTEM ${TorchDelegate_INCLUDE_DIRS})
endif()
```

#### 5.3.3 完整 CMake 集成代码

```cmake
# ============================================
# 顶层 CMakeLists.txt（在 USE_NPU 块内追加）
# ============================================
if(USE_NPU)
  # ... 现有 NPU 配置 ...

  option(USE_TORCH_DELEGATE "Enable torch-delegate GE graph executor" OFF)
  if(USE_TORCH_DELEGATE)
    add_definitions(-DUSE_TORCH_DELEGATE)
    
    if(DEFINED ENV{TORCH_DELEGATE_ROOT})
      list(APPEND CMAKE_PREFIX_PATH "$ENV{TORCH_DELEGATE_ROOT}")
      find_package(TorchDelegate REQUIRED)
      include_directories(SYSTEM ${TorchDelegate_INCLUDE_DIRS})
      message(STATUS "torch-delegate enabled: ${TorchDelegate_INCLUDE_DIRS}")
    else()
      message(FATAL_ERROR "USE_TORCH_DELEGATE=ON requires TORCH_DELEGATE_ROOT env var")
    endif()
  endif()
endif()

# ============================================
# xllm/CMakeLists.txt（链接阶段追加）
# ============================================
if(USE_TORCH_DELEGATE)
  target_link_libraries(xllm PUBLIC TorchDelegate::torch_delegate_backend)
endif()
```

#### 5.3.4 条件编译

EpModel 相关代码通过 `#ifdef USE_TORCH_DELEGATE` 保护：

```cpp
// ep_model.h
#ifdef USE_TORCH_DELEGATE

#include "torch_delegate/epair_model_loader.h"

class EpModel : public CausalLM {
    // ...
};

#endif  // USE_TORCH_DELEGATE
```

```cpp
// executor_impl_factory.h
#ifdef USE_TORCH_DELEGATE
#include "ge_graph_executor_impl.h"
REGISTER_EXECUTOR("ge", GeGraphExecutorImpl);
#endif
```

这样不启用 `USE_TORCH_DELEGATE` 时，所有 GE Graph 代码被完全跳过，不影响现有构建。

## 6. 实现计划

### Phase 1：基础设施 + EpModel
1. CMake 集成 `torch-delegate`（`USE_TORCH_DELEGATE` 选项 + `find_package`）
2. 验证 ABI 兼容性（`_GLIBCXX_USE_CXX11_ABI` 一致）
3. 验证 `libruntime.so` / `libtorch_cpu.so` 无符号冲突
4. 实现 `ge_tensor_utils.h`（`TorchToGe` / `GeToTorch` / dtype 映射）
5. 实现 `EpModel`（继承 `CausalLM`，`#ifdef USE_TORCH_DELEGATE` 保护）
   - `load_model()`：GE 初始化 → 构造 `EpairModelLoader` → `CompileAndLoad` → `ParseGraphIO`
   - `forward()`：`BuildGraphInputs` → `RunModelWithStreamAsync` → `ConvertOutputs`
   - `BuildGraphInputs()`：按 `input_names_` 从函数参数 + MMData 取数据
6. 单元测试：EpModel load + forward PoC

### Phase 2：Executor + Worker/Engine Pipeline
1. 实现 `GeGraphExecutorImpl`（`REGISTER_EXECUTOR("ge", ...)`）
2. 实现 `GeGraphWorkerPipeline`
   - `prepare_work_before_execute()`：复用 Base（H2D + KV block swap）
   - `step()`：单次 `executor->forward()` + 构造 `ForwardOutput`
3. 实现 `GeGraphEnginePipeline`
   - `step()`：prepare → get_model_output → process_beam_sequence_group → finish
4. 集成测试：端到端推理

### Phase 3：Master + Scheduler Pipeline
1. 实现 `process_epair_inputs()`（通用输入校验）
2. 实现 `GeGraphMasterPipeline`
3. 实现 `GeGraphSchedulerPipeline`（KV Cache 分配）
4. 工厂方法扩展（`RecPipelineType::kGeGraphPipeline`）
5. 端到端集成测试

### Phase 4：扩展 + 优化
1. 名字匹配函数扩展（覆盖更多模型的命名模式）
2. 动态 shape 输出处理
3. 多卡 TP 场景验证
4. 性能优化（零拷贝、输出预分配复用）

## 7. 已确认事项与待确认事项

### 7.1 已确认事项

| # | 事项 | 决策 |
|---|---|---|
| 1 | Graph 输入节点命名和顺序 | 通过 `GetGEGraph()` 解析 Data 节点的 `"index"` 属性获取顺序，`GetName()` 获取名称 |
| 2 | Graph 输出 Shape | 通过 `GetGEGraph()` 解析 NetOutput 节点的 TensorDesc 获取；固定 shape 预分配复用，动态 shape 按需分配 |
| 3 | EpairModelLoader API | 构造函数传路径、`CompileAndLoad(options_map, stream)`、`RunModelWithStreamAsync()` |
| 4 | Tensor 桥接方式 | 拷贝方式（aclrtMemcpy D2D），torch::Tensor ↔ gert::Tensor |
| 5 | 图输入映射方式 | 有序名字列表（`vector<string>`），EpModel 内部按名字从函数参数 + MMData 取数据，ModelInputParams 零改动 |
| 6 | EpModel 继承 | 直接继承 `CausalLM`（通用设计，不限 REC） |
| 7 | KV Cache 管理 | 图内部原地更新，Host 不管理 |
| 8 | beam search | 在图内完成，Pipeline 不需要 Sampler / BeamSearcher |
| 9 | GE 图输出格式 | `beam_sequence_group`（最终结果），不是 logits |
| 10 | 多轮 decode | 图内完成，Pipeline 单次 forward |

### 7.2 错误处理策略

**决策**：宽松策略（LOG ERROR + 返回空 ModelOutput）

**理由**：
- GeGraphExecutor 用于在线服务
- 错误不中断推理，适合容错场景
- 调用方检查返回结果是否为空

## 8. 测试计划

### 8.1 单元测试
- `EpModel` 测试：GE 初始化、load_model、forward、BuildGraphInputs
- `GeGraphExecutorImpl` 测试：构造函数、run()
- `ParseGraphIO` 测试：输入/输出名字解析正确性

### 8.2 集成测试
- 端到端推理测试
- 多卡场景测试
- 多线程并发测试

### 8.3 性能测试
- H2D 拷贝优化验证
- Stream 复用验证
- 多卡负载均衡测试

## 9. 兼容性分析

### 9.1 设计原则

GE Graph 路径作为**新增执行路径**，与现有 ATB 路径**完全隔离**，不引入任何兼容性问题。

### 9.2 隔离机制

| 维度 | 现有 ATB 路径 | 新增 GE Graph 路径 | 隔离方式 |
|------|--------------|-------------------|----------|
| **Executor** | `BaseExecutorImpl` / `RecExecutorImpl` | `GeGraphExecutorImpl` | 通过 `REGISTER_EXECUTOR("ge", ...)` 注册，runtime 配置选择 |
| **Model** | `OneRecForConditionalGeneration` 等 | `EpModel` | 不同的模型类，通过 `model_type` 配置区分 |
| **Pipeline** | `OneRecWorkPipeline` 等 | `GeGraphWorkerPipeline` | 通过 `RecPipelineType::kGeGraphPipeline` 枚举区分 |
| **ModelInputParams** | 现有字段 | **零改动** | 不新增任何字段，图输入映射在 EpModel 内部完成 |
| **依赖库** | ATB / torch_npu | torch_delegate + GE | CMake 条件编译，可选依赖 |

### 9.3 配置驱动

GE Graph 路径通过配置启用，默认关闭，不影响现有部署：

```cpp
// 通过 model_type 或 runtime 配置选择执行路径
if (options.backend() == "ge" && options.enable_graph()) {
    // GE Graph 路径
    backend = "ge";
} else {
    // 现有 ATB 路径
    backend = options.backend();  // "llm", "rec", "npu" 等
}
```

### 9.4 零侵入验证清单

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| ModelInputParams 结构 | `git diff` 检查 | 无改动 |
| 现有 Pipeline 逻辑 | `git diff` 检查 | 无改动（仅新增 `kGeGraphPipeline` 枚举） |
| 现有 Executor 注册 | 运行现有测试 | 全部通过 |
| CMake 构建 | 不配置 `USE_TORCH_DELEGATE=ON` | 构建成功（GE 路径条件编译跳过） |
| ABI 兼容性 | `_GLIBCXX_USE_CXX11_ABI` 一致 | torch-delegate 与 xLLM 使用相同 ABI 的 libtorch |
| 符号冲突 | `ldd` 检查 `libruntime.so` / `libtorch_cpu.so` | 无重复符号，共享同一 CANN/PyTorch 安装 |
| 运行时选择 | 配置 `backend="rec"` | 走现有 ATB 路径，行为不变 |

### 9.5 回退策略

如果 GE Graph 路径出现问题，可通过配置快速回退：

```bash
# 回退到 ATB 路径
--backend=rec --enable_graph=false
```

无需代码改动，无需重新编译。

## 10. 维测设计

### 13.1 日志规范

#### 13.1.1 关键路径日志

```cpp
// EpModel::load_model()
LOG(INFO) << "EpModel loading epair: " << epair_path;
LOG(INFO) << "EpModel parsed " << input_names_.size() << " inputs, " 
          << output_names_.size() << " outputs";
LOG(INFO) << "EpModel input_names: " << input_names_;  // 打印完整列表
LOG(INFO) << "EpModel output_names: " << output_names_;

// EpModel::forward()
VLOG(1) << "EpModel forward: tokens=" << tokens.sizes() 
        << ", positions=" << positions.sizes();
VLOG(2) << "EpModel BuildGraphInputs: " << input_names_.size() << " inputs";

// 性能日志
Timer timer;
loader_->RunModelWithStreamAsync(stream, graph_inputs, device_outputs);
LOG(INFO) << "EpModel RunModelWithStreamAsync latency: " << timer.elapsed_ms() << " ms";
```

#### 13.1.2 错误日志

```cpp
// 加载失败
LOG(ERROR) << "EpModel failed to load epair: " << epair_path 
           << ", status=" << status;

// 输入缺失
LOG(ERROR) << "EpModel missing input tensor: " << name 
           << ", available in MMData: " << mm_data.keys();

// 执行失败
LOG(ERROR) << "EpModel RunModelWithStreamAsync failed: status=" << status
           << ", input_count=" << graph_inputs.size()
           << ", output_count=" << device_outputs.size();
```

#### 13.1.3 调试日志

```cpp
// 详细调试（VLOG 级别）
VLOG(3) << "EpModel BuildGraphInputs detail:";
for (size_t i = 0; i < input_names_.size(); ++i) {
    const auto& name = input_names_[i];
    if (IsTokenIds(name)) {
        VLOG(3) << "  [" << i << "] " << name << " -> tokens " << tokens.sizes();
    } else if (IsPositions(name)) {
        VLOG(3) << "  [" << i << "] " << name << " -> positions " << positions.sizes();
    } else if (auto kv = ParseKVCacheName(name)) {
        VLOG(3) << "  [" << i << "] " << name << " -> kv_caches[" << kv->layer << "]";
    } else {
        VLOG(3) << "  [" << i << "] " << name << " -> MMData";
    }
}
```

### 13.2 指标采集

#### 13.2.1 性能指标

```cpp
// 在 EpModel::forward() 中采集
struct EpModelMetrics {
    double build_inputs_ms = 0.0;      // BuildGraphInputs 耗时
    double run_graph_ms = 0.0;         // RunModelWithStreamAsync 耗时
    double convert_outputs_ms = 0.0;   // ConvertOutputs 耗时
    double total_ms = 0.0;             // 总耗时
    size_t input_count = 0;            // 输入 tensor 数量
    size_t output_count = 0;           // 输出 tensor 数量
};

// 上报到监控系统
MetricsReporter::report("ep_model_forward_latency", metrics.total_ms);
MetricsReporter::report("ep_model_graph_run_latency", metrics.run_graph_ms);
MetricsReporter::report("ep_model_input_count", metrics.input_count);
```

#### 13.2.2 错误指标

```cpp
// 错误计数
MetricsReporter::increment("ep_model_load_error_count");
MetricsReporter::increment("ep_model_forward_error_count");
MetricsReporter::increment("ep_model_missing_input_count");
```

### 13.3 调试工具

#### 13.3.1 Dump 输入/输出 Tensor

```cpp
// 通过环境变量或配置开关控制
if (FLAGS_dump_ep_model_tensors) {
    DumpTensors("ep_model_inputs", graph_inputs, input_names_);
    DumpTensors("ep_model_outputs", device_outputs, output_names_);
}

void DumpTensors(const std::string& prefix, 
                 const std::vector<gert::Tensor>& tensors,
                 const std::vector<std::string>& names) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        std::string filename = prefix + "_" + names[i] + ".bin";
        DumpTensorToFile(t, filename);
        LOG(INFO) << "Dumped tensor: " << names[i] 
                  << ", shape=" << t.GetShape() 
                  << ", dtype=" << t.GetDataType()
                  << ", file=" << filename;
    }
}
```

#### 13.3.2 对比验证工具

```cpp
// 对比 GE Graph 输出与 ATB 路径输出
if (FLAGS_compare_ge_vs_atb) {
    auto atb_output = RunATBPath(tokens, positions, kv_caches, params);
    auto ge_output = RunGEPath(tokens, positions, kv_caches, params);
    
    CompareOutputs("beam_sequence_group", 
                   atb_output.beam_sequence_group, 
                   ge_output.beam_sequence_group);
}
```

### 13.4 故障排查指南

| 现象 | 可能原因 | 排查方法 |
|------|----------|----------|
| `load_model` 失败 | epair 文件不存在或损坏 | 检查文件路径，验证 epair 完整性 |
| `CompileAndLoad` 失败 | GE 初始化失败或图编译错误 | 检查 GE 日志，确认 CANN 版本兼容 |
| `RunModelWithStreamAsync` 失败 | 输入 tensor shape/dtype 不匹配 | 开启 `VLOG(3)` 查看输入详情，dump tensor 对比 |
| 输出全零或异常 | KV Cache 未正确初始化 | 检查 `allocate_kv_cache` 是否被调用 |
| 性能低于 ATB | 图编译开销或内存拷贝 | 对比 `build_inputs_ms` / `run_graph_ms` / `convert_outputs_ms` |
| 名字匹配失败 | epair 输入命名不在已知模式 | 查看 `input_names_` 日志，扩展 `IsTokenIds` 等函数 |

### 13.5 监控大盘

建议配置以下监控面板：

```
┌─────────────────────────────────────────────────────────┐
│  GE Graph Executor 监控                                  │
├─────────────────────────────────────────────────────────┤
│  请求量:     [实时曲线]                                  │
│  成功率:     [百分比]                                    │
│  平均延迟:   [P50 / P95 / P99]                          │
│  错误分布:   [饼图: load_error / forward_error / ...]   │
│  输入数量:   [直方图]                                    │
│  输出数量:   [直方图]                                    │
└─────────────────────────────────────────────────────────┘
```

## 11. 客户端请求/响应示例

### 11.1 HTTP 请求示例

#### 11.1.1 基本推理请求

```bash
curl -X POST http://localhost:8080/v1/rec/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onerec-ge",
    "prompt": "用户搜索: 运动鞋",
    "max_tokens": 10,
    "beam_width": 5,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

#### 11.1.2 带 MMData 的请求

```bash
curl -X POST http://localhost:8080/v1/rec/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "onerec-ge",
    "prompt_tokens": [101, 2054, 3078, 102],
    "input_tensors": [
      {
        "name": "sparse_embedding",
        "shape": [4, 768],
        "dtype": "float",
        "contents": {
          "fp32_contents": [0.1, 0.2, ...]
        }
      },
      {
        "name": "decoder_context_embedding",
        "shape": [4, 768],
        "dtype": "float",
        "contents": {
          "fp32_contents": [0.3, 0.4, ...]
        }
      }
    ],
    "max_tokens": 10,
    "beam_width": 5
  }'
```

### 11.2 HTTP 响应示例

#### 11.2.1 成功响应

```json
{
  "id": "cmpl-12345",
  "object": "text_completion",
  "created": 1704067200,
  "model": "onerec-ge",
  "choices": [
    {
      "text": "推荐商品ID: 9876543210",
      "index": 0,
      "logprobs": {
        "tokens": [9876, 5432, 10],
        "token_logprobs": [-0.1, -0.2, -0.3]
      },
      "finish_reason": "length"
    },
    {
      "text": "推荐商品ID: 1234567890",
      "index": 1,
      "logprobs": {
        "tokens": [1234, 5678, 90],
        "token_logprobs": [-0.5, -0.6, -0.7]
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 10,
    "total_tokens": 14
  }
}
```

#### 11.2.2 错误响应

```json
{
  "error": {
    "message": "EpModel failed to load epair: /models/onerec-ge/model.epair",
    "type": "model_load_error",
    "code": "epair_load_failed"
  }
}
```

### 11.3 gRPC 请求示例

```protobuf
message RecCompletionRequest {
  string model = 1;                    // "onerec-ge"
  repeated int32 prompt_tokens = 2;    // [101, 2054, 3078, 102]
  repeated InferInputTensor input_tensors = 3;
  uint32 max_tokens = 4;               // 10
  uint32 beam_width = 5;               // 5
  float temperature = 6;               // 0.7
  float top_p = 7;                     // 0.9
}

message InferInputTensor {
  string name = 1;                     // "sparse_embedding"
  repeated int64 shape = 2;            // [4, 768]
  DataType data_type = 3;              // FLOAT
  TensorContents contents = 4;
}
```

### 11.4 配置示例

#### 11.4.1 启用 GE Graph 路径

```bash
# 启动参数
xllm serve /models/onerec-ge \
  --backend=ge \
  --enable_graph=true \
  --max_batch_size=8 \
  --max_tokens_per_batch=1024
```

#### 11.4.2 回退到 ATB 路径

```bash
# 回退配置
xllm serve /models/onerec \
  --backend=rec \
  --enable_graph=false
```

## 12. 总结

**核心优势**：
- ✅ **继承设计**：EpModel 继承 CausalLM，符合模型抽象，可无缝接入 Executor
- ✅ **简洁架构**：Model 与 Graph 直接绑定，无需单例管理图
- ✅ **职责清晰**：EpModel 管理自己的 epair 模型，Worker/Engine 负责输入准备
- ✅ **零侵入**：ModelInputParams 和 Pipeline 层无需改动，图输入映射在 EpModel 内部完成
- ✅ **性能优化**：避免 H2D 拷贝，复用 Worker Stream
- ✅ **易于扩展**：支持多种模型类型（参考 RecCausalLM 设计）

**执行链路**：
```
Worker -> Engine -> GraphExecutor -> EpModel.forward() -> EpairModelLoader.RunModelWithStreamAsync
```

**实现优先级**：
1. EpModel（直接继承 CausalLM，包含 GE 初始化）
2. ModelInputParams 扩展（输入映射）
3. GeGraphExecutorImpl（Executor 实现）
4. 单元测试 + 集成测试

**下一步**：
- 确认 epair 文件的节点命名规范
- 开始实现 EpModel（继承 CausalLM）

## 13. Pipeline 设计方案

### 13.1 设计动机

#### 13.1.1 现有 Pipeline 的问题

当前 `RecEnginePipeline` 和 `RecWorkPipeline` 的各实例（LlmRec、OneRec、OneRecXAttention、RecMultiRound）承担了大量本应属于 Model 层的计算逻辑：

| 职责 | 所在层 | 具体操作 |
|------|--------|---------|
| 多轮 decode 循环 | EnginePipeline / WorkPipeline | `for i in [0, kRecDecodeSteps)` 循环驱动多步推理 |
| Beam Search | WorkPipeline | `beam_searcher_->forward()`、`process_beam_search_output()` |
| Sampling | WorkPipeline | `sampler_->forward()`、`rec_sampler_->forward()` |
| 约束解码 | WorkPipeline | `prepare_filter_mask_async()`、`RecSampler` with `filter_mask` |
| KV Cache 轮次管理 | WorkPipeline | 每轮修改 `input_params`、`token_ids`、`positions`、`attn_metadata` |
| 输出后处理 | EnginePipeline | `process_sample_output()`、`process_beam_search_output()`、`process_beam_sequence_group()` |

这些操作在单算子执行模式下是合理的——Host 需要逐步驱动每个计算步骤。但在 **torch_delegate 图模式**下，GE 图已经将整个模型的计算（包括多轮 decode、beam search、sampling）封装为一次 `RunModelWithStreamAsync()` 调用。Pipeline 层再继续承担这些职责会导致：

1. **职责重叠**：图内部已完成 beam search + sampling，Pipeline 再做一遍是冗余
2. **接口不匹配**：图的输入/输出是 Tensor，不需要 `SamplingParameters`、`filter_mask` 等 Host 侧结构
3. **维护成本**：每新增一种图模式都需要适配复杂的 Pipeline 逻辑

#### 13.1.2 设计目标

为 torch_delegate 图模式新增专用的 `GeGraphEnginePipeline` 和 `GeGraphWorkerPipeline`，实现：

- **职责单一**：Pipeline 只负责 Batch ↔ Tensor 的转换，不承载模型计算逻辑
- **单次推理**：一次 `step()` 调用 = 一次 `executor->forward()` 调用，所有多轮逻辑在图内完成
- **最小依赖**：不需要 Sampler、BeamSearcher、filter_mask 等组件
- **输出对齐**：输出格式对齐现有 `ForwardOutput.beam_sequence_group`，复用 `Batch::process_beam_sequence_group()`

### 13.2 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RecEngine                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  GeGraphEnginePipeline : RecEnginePipeline                    │  │
│  │                                                                │  │
│  │  step(batches):                                               │  │
│  │    1. workers_[0]->prepare_inputs(batches[0])                 │  │
│  │    2. get_model_output(forward_inputs)                        │  │
│  │       └─ 所有 workers 异步 step_async                         │  │
│  │       └─ 取 rank 0 输出                                       │  │
│  │       └─ D2H: beam_sequence_group / out_logprobs → CPU        │  │
│  │    3. batches[0].process_beam_sequence_group(output)          │  │
│  │    4. batches[0].finish()                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RecWorkerImpl                                   │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  GeGraphWorkerPipeline : RecWorkPipeline                      │  │
│  │                                                                │  │
│  │  prepare_inputs(batch):                                       │  │
│  │    └─ batch.prepare_forward_input() → ForwardInput            │  │
│  │                                                                │  │
│  │  prepare_work_before_execute(inputs, processed_inputs):       │  │
│  │    ├─ H2D 传输 (复用 Base 逻辑)                               │  │
│  │    ├─ KV block swap                                           │  │
│  │    └─ 复用 Base（H2D + KV block swap）                       │  │
│  │                                                                │  │
│  │  step(input):                                                 │  │
│  │    ├─ executor->forward(tokens, positions, kv_caches, params) │  │
│  │    │   └─ GeGraphExecutorImpl::run()                          │  │
│  │    │       └─ EpModel::forward()                              │  │
│  │    │           └─ RunModelWithStreamAsync()                   │  │
│  │    └─ 构造 ForwardOutput (beam_sequence_group 格式)           │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**执行链路**：
```
RecEngine::step()
  └─> GeGraphEnginePipeline::step()
        ├─> prepare_inputs()
        ├─> GeGraphWorkerPipeline::step()
        │     ├─> prepare_work_before_execute()
        │     ├─> GeGraphExecutorImpl::run()
        │     │     └─> EpModel::forward()
        │     │           └─> RunModelWithStreamAsync()
        │     └─> 构造 ForwardOutput
        ├─> process_beam_sequence_group()
        └─> batch.finish()
```

### 13.3 GeGraphMasterPipeline 设计

#### 13.3.1 四层 Pipeline 全景

Rec 推理链路共有四层 Pipeline 抽象，GE 图模式需要在每一层都提供对应实现：

```
Layer 1: RecMasterPipeline    → 请求构造 (API 输入 → Request)
Layer 2: RecEnginePipeline    → 引擎编排 (Batch → ForwardInput → ForwardOutput)
Layer 3: RecWorkPipeline      → Worker 计算 (ForwardInput → executor→forward → ForwardOutput)
Layer 4: SchedulerPipeline     → 调度批处理 (Sequence → Batch)
```

各层现有实例与 GE 图模式的映射关系：

```
RecPipelineType              → MasterPipeline                    → EnginePipeline                     → WorkPipeline                      → SchedulerPipeline
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
kLlmRecDefault               → LlmRecMasterPipeline              → LlmRecEnginePipeline               → LlmRecWorkPipeline                → LlmRecSchedulerPipeline
kLlmRecWithMmData            → LlmRecWithMmDataMasterPipeline    → (复用 LlmRec)                      → LlmRecWithMmDataWorkPipeline      → (复用 LlmRec)
kLlmRecMultiRoundPipeline    → LlmRecMasterPipeline (复用)       → RecMultiRoundEnginePipeline        → LlmRecMultiRoundPipeline          → RecMultiRoundSchedulerPipeline
kOneRecDefault               → OneRecPrefillOnlyMasterPipeline   → OneRecPrefillOnlyEnginePipeline    → OneRecWorkPipeline                → OneRecSchedulerPipeline
kOneRecXAttentionPipeline    → OneRecXAttentionMasterPipeline    → OneRecXAttentionEnginePipeline     → OneRecXAttentionWorkPipeline      → OneRecXAttentionSchedulerPipeline
kGeGraphPipeline (新增)      → GeGraphMasterPipeline (新增)      → GeGraphEnginePipeline (新增)       → GeGraphWorkerPipeline (新增)      → GeGraphSchedulerPipeline (新增)
```

#### 13.3.2 RecMasterPipeline 的职责

`RecMasterPipeline` 是请求构造层的策略抽象，负责将外部 API 输入（prompt、token_ids、input_tensors、MMData）转换为内部 `Request` 对象。

**核心方法**：
```cpp
class RecMasterPipeline {
public:
    // prompt 输入方式
    virtual std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback);

    // raw 输入方式（token_ids + MMData）
    virtual std::shared_ptr<Request> generate_request(
        const std::vector<int>& prompt_tokens,
        std::optional<MMData> mm_data,
        const RequestParams& sp,
        OutputCallback callback);
};
```

**与 Engine/Worker Pipeline 的本质区别**：
- Master Pipeline 不参与模型执行，只负责请求构造
- 不涉及 `step()`、`prepare_inputs()`、`prepare_work_before_execute()` 等执行方法
- 输出是 `std::shared_ptr<Request>`，而非 `ForwardOutput`

#### 13.3.3 GE 图模式下的核心差异

GE 图内部自行完成 **sampling + beam search + 多轮 decode + 停止判断**，这导致 Master 层的 Request 构造与现有 Pipeline 存在以下关键差异：

| 维度 | 现有 Pipeline | GE 图模式 | 影响 |
|------|-------------|----------|------|
| **StoppingChecker** | `build_stop_checker=true`（LlmRec/XAttention）或 `false`（PrefillOnly），Host 逐步检查 | **必须 `false`**——图内部控制停止，Host 无法逐步检查 | `build_request_common()` 参数 |
| **停止条件传递** | 由 Host 侧 `StoppingChecker::check()` 每步执行 | 需要把 `max_tokens`、`eos_token_id`、`stop_token_ids` **作为图输入 Tensor** 传入 GE 图 | Request 需要携带这些元数据到图 |
| **Sampling 参数** | 聚合成 per-token 张量（`temperatures`、`top_p`、`top_k` 等 batch 张量） | 图内处理采样，需要的是**标量配置**而非 per-token 张量 | `SamplingParameters::init()` 的聚合逻辑对 GE 图是冗余的 |
| **beam_width / num_return_sequences** | Host 侧驱动 beam 扩展、`SequencesGroup::process_beam_search()` | 图内完成 beam search，Host **不应** 做 beam 扩展 | `Sequence` 生命周期管理 |
| **Sequence 生命周期** | `append_token()` → `finished()` → `process_beam_search()` 循环 | 图一次性返回完整结果，Sequence **不走逐步循环** | Scheduler/Engine 层的 Sequence 管理 |
| **max_tokens** | 用于 `StoppingChecker` + `seq_capacity` 计算 | 同时作为图的 **最大 decode 步数** 输入 | 需要额外传递到图 |

#### 13.3.4 StoppingChecker 的悖论

```
现有模式:
  Request → Sequence → 每步 append_token → StoppingChecker::check() → 决定是否终止

GE 图模式:
  Request → Sequence → 一次性 forward → 图内部循环 decode + 停止 → 返回完整结果
                                            ↑
                                   图怎么知道什么时候停？
                                   → 需要 max_tokens / eos_token_id / stop_token_ids 作为图输入
```

`StoppingChecker` 在 GE 图模式下有双重角色：
1. **不构建** Host 侧的 `StoppingChecker` 对象（`build_stop_checker=false`）
2. **但要保留** 停止条件的原始数据（`max_tokens`、`eos_token_id`、`stop_token_ids`），让它们能流到图输入

**当前问题**：`build_request_common()` 中，`max_tokens`、`eos_token_id`、`stop_token_ids` 在构建完 `StoppingChecker` 后就丢弃了。GE 图模式需要在 `RequestState` 中保留它们。

#### 13.3.5 数据流差异

```
现有模式:
  RequestParams
    ├─ max_tokens ──→ StoppingChecker (Host 逐步检查)
    ├─ beam_width ──→ RequestSamplingParam ──→ SamplingParameters (per-token 张量)
    ├─ temperature ─→ RequestSamplingParam ──→ SamplingParameters (per-token 张量)
    └─ stop_token_ids → StoppingChecker (Host 逐步检查)

GE 图模式:
  RequestParams
    ├─ max_tokens ──→ RequestState 保留 ──→ MMData["max_decode_steps"]
    ├─ beam_width ──→ RequestState 保留 ──→ MMData["beam_width"]
    ├─ temperature ─→ RequestState 保留 ──→ MMData["temperature"]
    ├─ eos_token_id → RequestState 保留 ──→ MMData["eos_token_id"]
    └─ stop_token_ids → RequestState 保留 ──→ MMData["stop_token_ids"]
```

#### 13.3.6 GeGraphMasterPipeline 类定义

```cpp
class GeGraphMasterPipeline final : public RecMasterPipeline {
public:
    explicit GeGraphMasterPipeline(RecMaster& master)
        : RecMasterPipeline(master) {}
    ~GeGraphMasterPipeline() override = default;

    std::shared_ptr<Request> generate_request(
        std::string prompt,
        std::optional<std::vector<int>> prompt_tokens,
        std::optional<std::vector<proto::InferInputTensor>> input_tensors,
        const RequestParams& sp,
        OutputCallback callback) override;
};
```

#### 13.3.7 generate_request() 实现

```cpp
std::shared_ptr<Request> GeGraphMasterPipeline::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp,
    OutputCallback callback) {

    std::vector<int32_t> local_prompt_tokens;
    MMData processed_mm_data;

    // 1. 通用输入校验（不复用 process_onerec_inputs，因为 epair 是模型无关的）
    if (!process_epair_inputs(prompt_tokens, input_tensors,
                              &local_prompt_tokens, &processed_mm_data, callback)) {
        return nullptr;
    }

    // 2. 构建 Request（build_stop_checker=false，图内部控制停止）
    return master_.build_request_common(
        std::move(prompt),
        std::move(local_prompt_tokens),
        std::move(processed_mm_data),
        sp,
        callback,
        /*build_stop_checker=*/false);
}
```

**设计要点**：
- `build_stop_checker=false`：GE 图内部控制停止，Host 不需要逐步检查
- 使用 `process_epair_inputs()` 而非 `process_onerec_inputs()`：epair 是模型无关的，不能硬编码 OneRec 的 tensor 名字和校验规则
- 复用 `build_request_common()` 的 `RequestState` 构建逻辑

#### 13.3.7a process_epair_inputs() 通用输入校验

**与 `process_onerec_inputs()` 的区别**：

| 校验项 | process_onerec_inputs | process_epair_inputs |
|---|---|---|
| prompt_tokens / input_tensors 互斥 | ✅ | ✅ |
| tensor 非空 | ✅ | ✅ |
| tensor 无重复 name | ✅ | ✅ |
| tensor 有 contents | ✅ | ✅ |
| shape 为正 | ✅ | ✅ |
| numel 与 data 一致 | ✅ | ✅ |
| **只允许 sparse_embedding / decoder_context_embedding** | ✅ | **❌ 不限名字** |
| **必须 FLOAT(fp32)** | ✅ | **❌ 不限 dtype** |
| **必须 2-D [len, hidden]** | ✅ | **❌ 不限维度** |
| **hidden == model_args.hidden_size()** | ✅ | **❌ 不校验** |
| **必须包含 sparse_embedding** | ✅ | **❌ 不要求** |
| **强制转 bfloat16** | ✅ | **❌ 保持原始 dtype** |

```cpp
bool process_epair_inputs(
    const std::optional<std::vector<int>>& prompt_tokens,
    const std::optional<std::vector<proto::InferInputTensor>>& input_tensors,
    std::vector<int32_t>* local_prompt_tokens,
    MMData* processed_mm_data,
    OutputCallback callback) {
    
    if (prompt_tokens.has_value() && input_tensors.has_value()) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "prompt_tokens and input_tensors cannot both be set");
        return false;
    }

    if (prompt_tokens.has_value()) {
        local_prompt_tokens->assign(prompt_tokens.value().begin(),
                                    prompt_tokens.value().end());
    }

    if (input_tensors.has_value()) {
        if (input_tensors->empty()) {
            CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                                "input_tensors cannot be empty");
            return false;
        }

        MMDict mm_dict;
        mm_dict.reserve(input_tensors->size());

        for (const auto& tensor : input_tensors.value()) {
            const auto& name = tensor.name();
            
            if (mm_dict.find(name) != mm_dict.end()) {
                CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                                    "Duplicate input tensor: " + name);
                return false;
            }
            if (!tensor.has_contents()) {
                CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                                    "Input tensor '" + name + "' has no contents");
                return false;
            }
            if (tensor.shape_size() < 1) {
                CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                                    "Input tensor '" + name + "' must have at least 1-D");
                return false;
            }

            try {
                mm_dict[name] = util::convert_rec_tensor_to_torch(tensor);
            } catch (const std::exception& e) {
                CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                                    "Failed to parse input tensor '" + name + "': " + e.what());
                return false;
            }
        }

        *processed_mm_data = MMData(MMType::EMBEDDING, mm_dict);
    }

    if (local_prompt_tokens->empty() && !processed_mm_data->valid()) {
        CALLBACK_WITH_ERROR(StatusCode::INVALID_ARGUMENT,
                            "Requires prompt_tokens or input_tensors");
        return false;
    }

    return true;
}
```

#### 13.3.8 RequestState 扩展

GE 图模式需要在 `RequestState` 中保留停止条件和采样配置的原始数据，以便后续流转到图输入：

```cpp
struct RequestState {
    // ... 已有字段 ...

    // 新增：GE 图模式需要的原始配置数据
    // 这些数据当前仅在构建 StoppingChecker 时使用，之后被丢弃
    // GE 图模式需要将它们保留并传递到图输入
    uint32_t max_tokens = 0;                    // 最大生成 token 数 → 图输入 max_decode_steps
    int32_t eos_token_id = -1;                  // EOS token ID → 图输入 eos_token_id
    std::unordered_set<int32_t> stop_token_ids; // 停止 token 集合 → 图输入 stop_token_ids
    std::vector<std::vector<int32_t>> stop_sequences; // 停止序列 → 图输入 stop_sequences

    // 采样配置标量（当前仅在 SamplingParameters::init() 中聚合为张量）
    // GE 图模式需要标量形式作为图输入
    float temperature = 0.0;                    // → 图输入 temperature
    float top_p = 1.0;                          // → 图输入 top_p
    int64_t top_k = -1;                         // → 图输入 top_k
    float repetition_penalty = 1.0;             // → 图输入 repetition_penalty
};
```

**`build_request_common()` 修改**：

```cpp
// 在 build_request_common() 中，构建 RequestState 时保留原始数据
RequestState state;
// ... 已有字段赋值 ...

// 新增：保留 GE 图模式需要的原始数据
state.max_tokens = max_tokens;
state.eos_token_id = eos_token_id;
state.stop_token_ids = stop_tokens;
state.stop_sequences = stop_sequences;
state.temperature = sp.temperature;
state.top_p = sp.top_p;
state.top_k = sp.top_k;
state.repetition_penalty = sp.repetition_penalty;
```

#### 13.3.9 GeGraphSchedulerPipeline

GE 图模式的 SchedulerPipeline，负责 Batch 构造和 KV Cache 分配。

```cpp
class GeGraphSchedulerPipeline final : public SchedulerPipeline {
public:
    explicit GeGraphSchedulerPipeline() = default;

    std::vector<Batch> create_batches(FixedStepsScheduler& scheduler,
                                       BatchFactory* batch_factory) override;

    bool requires_kv_cache() const override { return true; }

    bool allocate_kv_cache(KVCacheManager* kv_cache_manager,
                            Sequence* sequence) override;
};
```

**与现有 SchedulerPipeline 的差异**：

| 维度 | OneRecSchedulerPipeline | GeGraphSchedulerPipeline |
|---|---|---|
| `requires_kv_cache()` | `false`（OneRec 不用 KV Cache） | **`true`**（GE 图需要） |
| Sequence 生命周期 | 逐步循环（每步 append_token） | **一次性完成**（图内完成全部 decode） |
| Batch 构造 | 类似 | 类似（复用 BatchFactory） |
| KV Cache 分配 | 不涉及 | 需要为每个 Sequence 分配 KV Cache blocks |

**`create_batches()` 实现要点**：
- 逻辑与 `LlmRecSchedulerPipeline` 类似（从 waiting_queue 取 Sequence，组装 Batch）
- 每个 Batch 包含一组 Sequence，每个 Sequence 对应一个请求
- GE 图模式下 Sequence 不经历逐步的 `append_token()` → `finished()` 循环
- 一次 `step()` 后 Sequence 直接标记为 FINISHED

**`allocate_kv_cache()` 实现要点**：
- 为 Sequence 分配 KV Cache blocks（复用 KVCacheManager）
- KV Cache tensor 作为图输入传入，图内部原地更新
- 分配的 block 数量需满足 `seq_capacity`（prompt_len + max_tokens）
- GE 图的 KV Cache 布局由 epair 导出时决定，Host 侧只需按标准 paged attention 格式分配

#### 13.3.10 工厂映射扩展

**RecMaster::create_pipeline**：
```cpp
std::unique_ptr<RecMasterPipeline> RecMaster::create_pipeline(
    RecPipelineType type, RecMaster& master) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<LlmRecMasterPipeline>(master);
        case RecPipelineType::kLlmRecWithMmData:
            return std::make_unique<LlmRecWithMmDataMasterPipeline>(master);
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecPrefillOnlyMasterPipeline>(master);
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionMasterPipeline>(master);
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphMasterPipeline>(master);
        default:
            LOG(FATAL) << "Unknown pipeline type";
            return nullptr;
    }
}
```

**FixedStepsScheduler::create_pipeline**：
```cpp
std::unique_ptr<SchedulerPipeline> FixedStepsScheduler::create_pipeline(
    RecPipelineType type) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
            return std::make_unique<LlmRecSchedulerPipeline>();
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecSchedulerPipeline>();
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionSchedulerPipeline>();
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<RecMultiRoundSchedulerPipeline>();
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphSchedulerPipeline>();
        default:
            LOG(FATAL) << "Unknown pipeline type";
            return nullptr;
    }
}
```

#### 13.3.11 对 Sequence 生命周期的影响

GE 图模式下，Sequence 的生命周期与现有模式有本质区别：

**现有模式**：
```
Sequence 创建
  → Scheduler 加入 Batch
  → Engine step() → Worker forward() → 返回 1 个 token
  → Sequence::append_token()
  → StoppingChecker::check() → 未完成则继续
  → 循环直到完成
```

**GE 图模式**：
```
Sequence 创建
  → Scheduler 加入 Batch
  → Engine step() → Worker forward() → 图内完成全部 decode
  → 返回完整结果 (beam_sequence_group)
  → Batch::process_beam_sequence_group() 一次性写入所有 token
  → Sequence 直接标记为 FINISHED
```

**关键影响**：
- Sequence 不经历逐步的 `append_token()` → `finished()` 循环
- `StoppingChecker` 不参与 Host 侧检查
- `SequencesGroup::process_beam_search()` 不执行（图内已完成）
- Scheduler 不需要在多个 step 之间维护 Sequence 状态

### 13.4 GeGraphEnginePipeline 设计

#### 13.4.1 职责

`GeGraphEnginePipeline` 是 `RecEnginePipeline` 的子类，负责 Engine 层的推理编排。其职责仅限于：

1. 初始化本地 Worker（复用 `OneRecLocalEnginePipeline` 的 Worker 管理逻辑）
2. 准备 Batch 输入
3. 触发 Worker 执行推理
4. 处理输出并回写 Batch

**不负责的职责**（与现有 Pipeline 的关键区别）：
- 不驱动多轮 decode 循环（图内完成）
- 不执行 beam search 后处理（图内完成，输出已是最终结果）
- 不执行 sampling（图内完成）

#### 13.4.2 类定义

```cpp
class GeGraphEnginePipeline final : public RecEnginePipeline {
public:
    explicit GeGraphEnginePipeline(RecEngine& engine);
    ~GeGraphEnginePipeline() override = default;

    void setup_workers() override;
    void process_group_test() override;
    bool init_model_workers(const std::string& model_path) override;
    int64_t estimate_min_available_memory() override;
    bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;
    int64_t minimal_kv_cache_blocks() const override { return 0; }
    
    ForwardOutput step(std::vector<Batch>& batches) override;
    
    std::vector<int64_t> get_active_activation_memory() const override;
    size_t num_workers() const override;

private:
    ForwardInput prepare_inputs(std::vector<Batch>& batches);
    ForwardOutput get_model_output(const ForwardInput& forward_inputs);

private:
    RecEngine& engine_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unique_ptr<ProcessGroup> process_group_;
};
```

#### 13.4.3 step() 流程

```
GeGraphEnginePipeline::step(batches)
│
├─ 1. prepare_inputs(batches)
│     └─ workers_[0]->prepare_inputs(batches[0])
│          └─ WorkerImpl::prepare_inputs(batch)
│               └─ model_executor_->prepare_inputs(batch)
│                    └─ batch.prepare_forward_input(args)
│
├─ 2. get_model_output(forward_inputs)
│     ├─ 所有 workers_ 异步 step_async(model_inputs)
│     ├─ folly::collectAll(futures).get()
│     ├─ 取 results.front() (rank 0 的输出)
│     └─ D2H: beam_sequence_group → CPU
│             beam_search_output.out_logprobs → CPU
│             Device::synchronize_default_stream()
│
├─ 3. batches[0].process_beam_sequence_group(output)
│
└─ 4. batches[0].finish()

返回: output
```

**与 RecMultiRoundEnginePipeline 的对比**：

| 步骤 | RecMultiRound | GeGraph |
|------|--------------|---------|
| 输入准备 | `workers_[0]->prepare_inputs()` | 相同 |
| 推理调用 | `get_model_output()` → Worker 内部多轮 | `get_model_output()` → Worker 内部单次 forward |
| 输出处理 | `process_beam_sequence_group()` | 相同 |
| finish | `batch.finish()` | 相同 |
| Worker 内部 | 多轮 decode 循环 + sampling + beam search | 单次 `executor->forward()`，图内完成一切 |

#### 13.4.4 Worker 管理

Worker 初始化逻辑复用 `OneRecLocalEnginePipeline` / `RecMultiRoundEnginePipeline` 的本地 Worker 模式：

```cpp
void GeGraphEnginePipeline::setup_workers() {
    // 空操作（本地 Worker 在 init_model_workers 中创建）
}

bool GeGraphEnginePipeline::init_model_workers(const std::string& model_path) {
    // 1. 创建 ProcessGroup
    //    单卡 NPU: create_process_group(rank=0, world_size=1, rank_size=1)
    //    多卡 NPU: create_npu_process_groups()
    //    非 NPU:   create_local_process_groups()
    
    // 2. 创建 Worker (WorkerType::REC)
    for (int rank = 0; rank < world_size; ++rank) {
        workers_.push_back(std::make_unique<Worker>(...));
    }
    
    // 3. 异步初始化模型
    std::vector<folly::SemiFuture<bool>> futures;
    for (auto& worker : workers_) {
        futures.push_back(worker->init_model_async(model_path));
    }
    auto results = folly::collectAll(futures).get();
    return std::all_of(results.begin(), results.end(), 
                       [](auto& r) { return r.value(); });
}
```

### 13.5 GeGraphWorkerPipeline 设计

#### 13.5.1 职责

`GeGraphWorkerPipeline` 是 `RecWorkPipeline` 的子类，负责 Worker 层的推理执行。其职责仅限于：

1. 将 `ForwardInput` 传递给 EpModel（EpModel 内部按 `input_names_` 构造图输入）
2. 调用 `executor->forward()` 执行一次推理
3. 将 `ModelOutput` 转换为 `ForwardOutput`（对齐 `beam_sequence_group` 格式）

**不负责的职责**（与现有 WorkPipeline 的关键区别）：
- 不需要 Sampler（图内完成采样）
- 不需要 BeamSearcher（图内完成 beam search）
- 不需要 filter_mask / RecSampler（图内完成约束解码）
- 不需要多轮循环（图内完成多轮 decode）
- 不需要每轮修改 `input_params` / `token_ids` / `positions`

#### 13.5.2 类定义

```cpp
class GeGraphWorkerPipeline final : public RecWorkPipeline {
public:
    explicit GeGraphWorkerPipeline(RecPipelineRuntime& runtime);
    ~GeGraphWorkerPipeline() override = default;

    ForwardInput prepare_inputs(Batch& batch) override;
    
    void prepare_work_before_execute(const ForwardInput& inputs,
                                      ForwardInput& processed_inputs) override;
    
    std::optional<ForwardOutput> step(const ForwardInput& input) override;
};
```

#### 13.5.3 prepare_inputs()

复用基类 `RecWorkPipeline::prepare_inputs()` 逻辑，将 Batch 转换为 `ForwardInput`：

```cpp
ForwardInput GeGraphWorkerPipeline::prepare_inputs(Batch& batch) {
    return RecWorkPipeline::prepare_inputs(batch);
}
```

#### 13.5.4 prepare_work_before_execute()

直接复用 Base 逻辑（H2D + KV block swap），**无需额外处理**：

```cpp
void GeGraphWorkerPipeline::prepare_work_before_execute(
    const ForwardInput& inputs, ForwardInput& processed_inputs) {
    
    // 复用 Base 逻辑：H2D 传输 + KV block swap
    // 自定义输入（如 sparse_embedding）已在 MMData 中，随 ForwardInput 正常流转
    RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
}
```

**设计要点**：
- Pipeline **不感知图结构**，不需要知道图要什么输入
- 标准输入（tokens、positions、kv_caches）通过 `forward()` 函数参数传递
- 自定义输入通过 `MMData`（`params.multimodal.mm_data`）传递，现有流程已覆盖
- EpModel 内部按 `input_names_` 从对应数据源取数据，Pipeline 层完全解耦

#### 13.5.5 step()

单次 forward，构造 `ForwardOutput`：

```cpp
std::optional<ForwardOutput> GeGraphWorkerPipeline::step(const ForwardInput& input) {
    auto& mutable_input = const_cast<ForwardInput&>(input);
    
    // 1. 单次 forward 调用（图内完成所有计算）
    auto model_output = runtime().executor->forward(
        mutable_input.token_ids,
        mutable_input.positions,
        runtime().worker.kv_caches_,
        mutable_input.input_params);
    
    // 2. 构造 ForwardOutput
    ForwardOutput output;
    
    // 从 ModelOutput 中提取 beam_sequence_group 格式的输出
    // GE 图的输出已经是 beam search 的最终结果
    if (model_output.beam_sequence_group.defined()) {
        output.beam_sequence_group = model_output.beam_sequence_group;
    }
    if (model_output.beam_search_output.out_logprobs.defined()) {
        output.beam_search_output = model_output.beam_search_output;
    }
    
    // 3. 设置 sampling 标志（从输入参数中透传）
    output.do_sample = mutable_input.sampling_params.do_sample;
    output.logprobs = mutable_input.sampling_params.logprobs;
    output.max_top_logprobs = mutable_input.sampling_params.max_top_logprobs;
    
    return output;
}
```

#### 13.5.6 ForwardInput 字段消费矩阵

| 字段 | GeGraph 使用方式 | 对比 Base RecWorkPipeline |
|------|-----------------|--------------------------|
| `token_ids` | READ → `forward()` 参数 → EpModel 内部匹配 | 相同（READ → executor） |
| `positions` | READ → `forward()` 参数 → EpModel 内部匹配 | 相同（READ → executor） |
| `input_params` | READ → `multimodal.mm_data` 传递给 EpModel | 相同（READ → executor） |
| `input_params.multimodal` | READ（仅 kCustom 类，由 Model override） | **忽略** |
| `sampling_params` | READ（仅透传标志字段到 output） | READ（用于 sampler） |
| `sampling_params.selected_token_idxes` | **忽略**（图内处理） | READ（用于 logits/sampler） |
| `sampling_params.use_beam_search` | **忽略**（图内处理） | READ（用于 beam_search kernel） |
| `sampling_params.acc_logprob` | **忽略**（图内处理） | READ（用于 beam_searcher） |
| `decoder_sampling_params` | **忽略** | 忽略 |
| `step_decode` / `step_meta()` | **忽略**（多轮在图内） | 忽略 |
| `transfer_kv_infos` | **忽略** | READ |
| `onerec_params` / `llmrec_params` | **忽略** | 各 Pipeline 按需 READ |

#### 13.5.7 与现有 WorkPipeline 的对比

| 维度 | Base RecWorkPipeline | OneRecXAttention | LlmRecMultiRound | **GeGraph** |
|------|---------------------|------------------|------------------|-------------|
| executor->forward() 次数 | 1 | 每轮 1~2，N 轮 | 每轮 1，N 轮 | **1** |
| Sampler | `sampler_` | `rec_sampler_` + filter_mask | `rec_sampler_` | **无** |
| Beam Search | `beam_searcher_` | 图内 | 图内 | **图内** |
| 多轮循环 | 无 | Worker 内 N 轮 | Worker 内 N 轮 | **无（图内）** |
| input_params 修改 | 无 | 拷贝+改 flag | 每轮原地修改 | **无** |
| 输出格式 | `sample_output` | `beam_sequence_group` | `beam_sequence_group` | **`beam_sequence_group`** |

### 13.6 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                GeGraphEnginePipeline::step()                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────── 输入组装 ──────────────┐                           │
│  │                                       │                           │
│  │  Batch (sequences)                    │                           │
│  │    │                                  │                           │
│  │    ├─ workers_[0]->prepare_inputs()   │                           │
│  │    │   └─ batch.prepare_forward_input │                           │
│  │    │       ├─ tokens, positions       │                           │
│  │    │       ├─ kv_cache_info           │                           │
│  │    │       └─ sampling_params         │                           │
│  │    │                                  │                           │
│  │    └─ → ForwardInput                 │                           │
│  │                                       │                           │
│  └───────────────────────────────────────┘                           │
│                       │                                               │
│                       ▼                                               │
│  ┌────────────── Worker 执行 ───────────┐                           │
│  │                                       │                           │
│  │  GeGraphWorkerPipeline::step()        │                           │
│  │    │                                  │                           │
│  │    ├─ prepare_work_before_execute()   │                           │
│  │    │   ├─ H2D 传输                    │                           │
│  │    │   ├─ KV block swap              │                           │
│  │    │   ├─ Schema 驱动填充标准输入     │                           │
│  │    │   └─ Model override 填充 Custom  │                           │
│  │    │                                  │                           │
│  │    ├─ executor->forward()             │                           │
│  │    │   └─ GeGraphExecutorImpl::run()  │                           │
│  │    │       └─ EpModel::forward()      │                           │
│  │    │           └─ RunModelWithStreamAsync()                       │
│  │    │               │                  │                           │
│  │    │               │  ┌──────────────────────────────────┐       │
│  │    │               │  │  GE Graph (epair)                │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Model Forward          │      │       │
│  │    │               │  │  │ (Transformer Layers)   │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  │              ▼                    │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Sampling / TopK / TopP │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  │              ▼                    │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Beam Search            │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  │              ▼                    │       │
│  │    │               │  │  ┌────────────────────────┐      │       │
│  │    │               │  │  │ Multi-round Decode     │      │       │
│  │    │               │  │  │ (loop in graph)        │      │       │
│  │    │               │  │  └───────────┬────────────┘      │       │
│  │    │               │  └──────────────┼───────────────────┘       │
│  │    │               │                  │                           │
│  │    │               ▼                  ▼                           │
│  │    │           ModelOutput                                        │
│  │    │           (beam_sequence_group + out_logprobs)               │
│  │    │                                  │                           │
│  │    └─ → ForwardOutput                │                           │
│  │                                       │                           │
│  └───────────────────────────────────────┘                           │
│                       │                                               │
│                       ▼                                               │
│  ┌────────────── 输出处理 ──────────────┐                           │
│  │                                       │                           │
│  │  D2H: beam_sequence_group → CPU       │                           │
│  │       out_logprobs → CPU              │                           │
│  │                                       │                           │
│  │  process_beam_sequence_group(output)  │                           │
│  │    ├─ 解析 [groups, beam_width, rounds]                           │
│  │    │   token 矩阵                      │                           │
│  │    └─ seq->set_beam_result()          │                           │
│  │                                       │                           │
│  │  batch.finish()                       │                           │
│  │                                       │                           │
│  └───────────────────────────────────────┘                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.7 RecPipelineType 扩展

#### 13.7.1 新增枚举值

```cpp
enum class RecPipelineType : uint8_t {
    kLlmRecDefault = 0,
    kLlmRecWithMmData = 1,
    kOneRecDefault = 2,
    kLlmRecMultiRoundPipeline = 3,
    kOneRecXAttentionPipeline = 4,
    kGeGraphPipeline = 5,  // 新增：torch_delegate GE 图模式
};
```

#### 13.7.2 工厂方法扩展

**RecEngine::create_pipeline**：
```cpp
std::unique_ptr<RecEnginePipeline> RecEngine::create_pipeline(
    RecPipelineType type, RecEngine& engine) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
            return std::make_unique<LlmRecEnginePipeline>(engine);
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<RecMultiRoundEnginePipeline>(engine);
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecPrefillOnlyEnginePipeline>(engine);
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionEnginePipeline>(engine);
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphEnginePipeline>(engine);
        default:
            LOG(FATAL) << "Unknown pipeline type: " << static_cast<int>(type);
            return nullptr;
    }
}
```

**RecWorkerImpl::create_pipeline**：
```cpp
std::unique_ptr<RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type, RecPipelineRuntime& runtime) {
    switch (type) {
        case RecPipelineType::kLlmRecDefault:
            return std::make_unique<LlmRecWorkPipeline>(runtime);
        case RecPipelineType::kOneRecDefault:
            return std::make_unique<OneRecWorkPipeline>(runtime);
        case RecPipelineType::kLlmRecMultiRoundPipeline:
            return std::make_unique<LlmRecMultiRoundPipeline>(runtime);
        case RecPipelineType::kOneRecXAttentionPipeline:
            return std::make_unique<OneRecXAttentionWorkPipeline>(runtime);
        case RecPipelineType::kGeGraphPipeline:          // 新增
            return std::make_unique<GeGraphWorkerPipeline>(runtime);
        default:
            LOG(FATAL) << "Unknown pipeline type: " << static_cast<int>(type);
            return nullptr;
    }
}
```

#### 13.7.3 Pipeline 类型选择逻辑

```cpp
RecPipelineType get_rec_pipeline_type(RecModelKind kind, 
                                       const ModelArgs& args,
                                       bool use_ge_graph) {
    if (use_ge_graph) {
        return RecPipelineType::kGeGraphPipeline;
    }
    
    // 原有逻辑...
    switch (kind) {
        case RecModelKind::kLlmRec:
            return RecConfig::max_decode_rounds() > 0 
                ? RecPipelineType::kLlmRecMultiRoundPipeline
                : RecPipelineType::kLlmRecDefault;
        case RecModelKind::kOneRec:
            return RecConfig::max_decode_rounds() > 0
                ? RecPipelineType::kOneRecXAttentionPipeline
                : RecPipelineType::kOneRecDefault;
        // ...
    }
}
```

### 13.8 ModelOutput 扩展

#### 13.8.1 问题

当前 `ModelOutput` 的字段是 `hidden_states`（不是 `logits`），由 WorkPipeline 中的 Sampler 和 BeamSearcher 处理后生成 `ForwardOutput`。但在 GE 图模式下，图输出已经是最终的 beam search 结果，`ModelOutput` 需要能够承载这些结果。

#### 13.8.2 扩展方案

在 `ModelOutput` 中新增 GE 图输出字段：

```cpp
struct ModelOutput {
    // 原有字段
    torch::Tensor hidden_states;       // [num_tokens, hidden_size]
    torch::Tensor residual;            // [num_tokens, hidden_size]
    torch::Tensor aux_hidden_states;   // [num_tokens, ...]
    
    // 新增：GE 图模式的直接输出
    torch::Tensor beam_sequence_group; // [groups, beam_width, rounds] token 矩阵
    BeamSearchOutput beam_search_output; // beam search 元数据 (out_logprobs 等)
};
```

#### 13.8.3 EpModel::forward() 输出处理

```cpp
ModelOutput EpModel::ConvertOutputs(std::vector<gert::Tensor>& device_outputs) {
    ModelOutput result;
    
    for (size_t i = 0; i < device_outputs.size() && i < output_specs_.size(); ++i) {
        torch::Tensor torch_output = GeToTorch(device_outputs[i], options_.device());
        const auto& name = output_specs_[i].name;
        
        if (name == "beam_sequence_group") {
            result.beam_sequence_group = torch_output;
        } else if (name == "out_logprobs") {
            result.beam_search_output.out_logprobs = torch_output;
        } else if (name == "hidden_states" || name == "logits") {
            result.hidden_states = torch_output;
        }
    }
    
    return result;
}
```

### 13.9 与现有方案的对比总结

#### 13.9.1 四层 Pipeline 职责对比

**Master 层（请求构造）**：

| 职责 | LlmRec | OneRecPrefill | OneRecXAttention | **GeGraph** |
|------|--------|--------------|------------------|-------------|
| 输入方式 | prompt + tokenizer | prompt + input_tensors | prompt + input_tensors | **prompt + tokenizer** |
| build_stop_checker | true | false | true | **false** |
| MMData 处理 | 无 | sparse_embedding 等 | sparse_embedding 等 | **无** |
| 停止条件保留 | 仅 StoppingChecker | 不保留 | 仅 StoppingChecker | **RequestState 保留原始数据** |
| 采样配置保留 | 仅 SamplingParam | 仅 SamplingParam | 仅 SamplingParam | **RequestState 保留标量** |

**Engine 层（推理编排）**：

| 职责 | LlmRec | OneRecPrefill | OneRecXAttention | MultiRound | **GeGraph** |
|------|--------|--------------|------------------|------------|-------------|
| Worker 初始化 | DistManager 远程 | 本地 PG + Worker | 本地 PG + Worker | 本地 PG + Worker | **本地 PG + Worker** |
| 输入准备 | DP 拆分 + prepare | prepare_inputs | prepare_inputs | prepare_inputs | **prepare_inputs** |
| forward 次数 | 动态 N 次 | 1+N 次 | N 轮 | N 轮 | **1 次** |
| 输出处理 | process_sample/beam | process_sample | process_beam_group | process_beam group | **process_beam_sequence_group** |

**Worker 层（设备计算）**：

| 职责 | LlmRec | OneRecPrefill | OneRecXAttention | MultiRound | **GeGraph** |
|------|--------|--------------|------------------|------------|-------------|
| Sampling | Worker 层 | Worker 层 | Worker 层 | Worker 层 | **图内** |
| Beam Search | Worker 层 | 无 | Worker 层 | Worker 层 | **图内** |
| 多轮 decode | 无 | Engine 层循环 | Worker 内循环 | Worker 内循环 | **图内** |
| Sampler 组件 | 需要 | 需要 | 需要 | 需要 | **不需要** |
| BeamSearcher 组件 | 需要 | 不需要 | 不需要 | 不需要 | **不需要** |

**Scheduler 层（调度批处理）**：

| 职责 | LlmRec | OneRec | OneRecXAttention | MultiRound | **GeGraph** |
|------|--------|--------|------------------|------------|-------------|
| requires_kv_cache | true | false | true | false | **true** |
| Sequence 生命周期 | 逐步循环 | 逐步循环 | 一次性完成 | 一次性完成 | **一次性完成** |

#### 13.9.2 代码复杂度对比

| 指标 | RecMultiRound WorkPipeline | **GeGraph WorkPipeline** |
|------|---------------------------|--------------------------|
| step() 行数 | ~140 行 | **~20 行** |
| prepare_work_before_execute() 行数 | ~130 行 | **~30 行** |
| 依赖组件 | Executor, RecSampler, KVCache manager | **Executor** |
| 需要理解的领域知识 | 多轮 decode、beam search、sampling、KV cache 轮次管理 | **Batch → Tensor 转换** |

### 13.10 文件结构

```
xllm/core/distributed_runtime/
├── rec_master.h                    # RecMasterPipeline 基类（已有）
├── rec_master.cpp                  # 新增 GeGraphMasterPipeline 实现
├── rec_engine.h                    # RecEnginePipeline 基类（已有）
├── rec_engine.cpp                  # 新增 GeGraphEnginePipeline 实现
└── ge_graph_engine_pipeline.h      # GeGraphEnginePipeline 类定义

xllm/core/runtime/
├── rec_worker_impl.h               # RecWorkPipeline 基类（已有）
├── rec_worker_impl.cpp             # 新增 GeGraphWorkerPipeline 实现
├── ge_graph_worker_pipeline.h      # GeGraphWorkerPipeline 类定义
├── executor_impl.h                 # ExecutorImpl 基类（已有）
├── ge_graph_executor_impl.h        # GeGraphExecutorImpl 定义
└── ge_graph_executor_impl.cpp      # GeGraphExecutorImpl 实现（已有）

xllm/core/scheduler/
├── fixed_steps_scheduler.h         # SchedulerPipeline 基类（已有）
└── fixed_steps_scheduler.cpp       # 新增 GeGraphSchedulerPipeline 实现

xllm/core/framework/model/
├── ep_model.h                      # EpModel（继承 CausalLM）
├── ep_model.cpp                    # EpModel 实现
└── ge_tensor_utils.h               # TorchToGe / GeToTorch / dtype 映射工具

xllm/core/framework/request/
└── request_state.h                 # RequestState（扩展 GE 图原始配置字段）

xllm/core/util/
└── rec_model_utils.h               # RecPipelineType 枚举（扩展 kGeGraphPipeline）
```

### 13.11 实现计划

#### Phase 1：EpModel 基础设施
1. 实现 `ge_tensor_utils.h`（`TorchToGe` / `GeToTorch` / dtype 映射）
2. 实现 `EpModel`（继承 `CausalLM`）
3. 实现 `ParseGraphIO()`（从 `GetGEGraph()` 解析 `input_names_` / `output_names_`）
4. 实现 `BuildGraphInputs()`（按名字从函数参数 + MMData 取数据）
5. 实现名字匹配辅助函数（`IsTokenIds` / `IsPositions` / `ParseKVCacheName`）

#### Phase 2：RequestState 扩展 + Master/Scheduler Pipeline
1. `RequestState` 新增 GE 图原始配置字段（`max_tokens`、`eos_token_id`、`stop_token_ids`、采样标量）
2. `build_request_common()` 修改：保留原始配置数据到 `RequestState`
3. 实现 `GeGraphMasterPipeline`
   - `generate_request()`：tokenize + `build_stop_checker=false`
4. 实现 `GeGraphSchedulerPipeline`
   - `create_batches()`：Batch 构造
   - `requires_kv_cache()`：返回 `true`
5. 工厂方法扩展：`RecMaster::create_pipeline()`、`FixedStepsScheduler::create_pipeline()`

#### Phase 3：基础 Pipeline 实现
1. 新增 `RecPipelineType::kGeGraphPipeline` 枚举值
2. 实现 `GeGraphWorkerPipeline`
   - `prepare_inputs()`：复用基类
    - `prepare_work_before_execute()`：复用 Base（H2D + KV block swap）
   - `step()`：单次 `executor->forward()` + 构造 `ForwardOutput`
3. 实现 `GeGraphEnginePipeline`
   - Worker 管理（复用本地 Worker 模式）
   - `step()`：prepare → get_model_output → process_beam_sequence_group → finish
4. `GeGraphExecutorImpl` 实现

#### Phase 4：ModelOutput 扩展
1. `ModelOutput` 新增 `beam_sequence_group` 和 `beam_search_output` 字段
2. `EpModel::forward()` 根据图输出节点名称解析结果

#### Phase 5：集成测试
1. 端到端推理测试（Request → Batch → GE 图 → beam_sequence_group → Batch → Response）
2. 与 RecMultiRoundEnginePipeline 的输出一致性对比
3. 多卡场景测试
4. 名字匹配正确性测试（覆盖 LLM / VLM / Rec 三种模型类型）
5. Sequence 生命周期验证（确认不走逐步 append_token 循环）

### 13.12 模型输入映射设计（有序名字列表）

#### 13.12.1 问题

不同模型的 GE 图输入节点差异很大：

```
LLM:  input_ids, position_ids, past_key_values[0].key, past_key_values[0].value, ...
VLM:  input_ids, position_ids, pixel_values, image_grid_thw, past_key_values[0].key, ...
Rec:  tokens, positions, encoder_tokens, encoder_positions, ...
```

需要将 xLLM 运行时数据映射到图的输入节点，且 `RunModelWithStreamAsync` 要求按精确顺序传入 `vector<gert::Tensor>`。

#### 13.12.2 方案选型

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **A: Pipeline 硬编码** | Pipeline 直接写死名字映射 | 简单直接 | 每新增模型改 Pipeline |
| **B: input_tensor_map** | ModelInputParams 新增 map，Pipeline 填充，EpModel 消费 | 解耦 | ModelInputParams 需改动，Pipeline 需感知图结构，数据搬两遍 |
| **C: 有序名字列表（选定）** | EpModel 内部维护 `vector<string>`，按名字从函数参数 + MMData 取数据 | ModelInputParams 零改动，Pipeline 零改动，代码量最小 | 名字匹配逻辑在 EpModel 内部 |

**选择方案 C**。核心洞察：数据已经在正确的位置了——标准输入是 `forward()` 函数参数，自定义输入在 `MMData` 中已带名字——不需要额外的映射表。

#### 13.12.3 核心设计

```
┌──────────────────────────────────────────────────────┐
│  EpModel                                              │
│  持有: vector<string> input_names_ (有序)             │
│  来源: 从 GetGEGraph() 解析 Data 节点 "index" 属性    │
└──────────────────────┬───────────────────────────────┘
                       │ BuildGraphInputs()
                       ▼
┌──────────────────────────────────────────────────────┐
│  按 input_names_ 遍历，按名字分类取数据：              │
│  - tokens/positions → forward() 函数参数              │
│  - k_cache[N]/v_cache[N] → kv_caches 参数            │
│  - 其他 → params.multimodal.mm_data 按名字取          │
└──────────────────────┬───────────────────────────────┘
                       │ TorchToGe() 拷贝转换
                       ▼
┌──────────────────────────────────────────────────────┐
│  vector<gert::Tensor> → RunModelWithStreamAsync()    │
└──────────────────────────────────────────────────────┘
```

#### 13.12.4 名字匹配辅助函数

EpModel 内部使用以下辅助函数将 `input_names_` 中的名字映射到数据源：

```cpp
// 判断是否为 token ids 输入
bool IsTokenIds(const std::string& name) {
    return name == "input_ids" || name == "tokens" || name == "token_ids";
}

// 判断是否为 positions 输入
bool IsPositions(const std::string& name) {
    return name == "position_ids" || name == "positions";
}

// 解析 KV Cache 名字，返回 {is_key, layer_index}
// 支持多种命名模式：
//   "past_key_values[N].key" / "past_key_values[N].value"
//   "k_cache[N]" / "v_cache[N]"
//   "key_cache[N]" / "value_cache[N]"
struct KVCacheMatch { bool is_key; int32_t layer; };
std::optional<KVCacheMatch> ParseKVCacheName(const std::string& name);
```

不匹配以上任何模式的名字视为**自定义输入**，从 `params.multimodal.mm_data` 按名字取。

#### 13.12.5 ParseGraphIO() 实现

EpModel 在 `load_model()` 中调用 `ParseGraphIO()`，从 `GetGEGraph()` 解析输入/输出名字：

```cpp
void EpModel::ParseGraphIO() {
    ge::Graph graph;
    loader_->GetGEGraph(&graph);
    
    // 解析输入：遍历 Data 节点，读 "index" 属性
    std::map<int64_t, std::string> indexed_inputs;
    for (const auto& node : graph.GetDirectNode()) {
        ge::AscendString type;
        node.GetType(type);
        if (type == "Data") {
            ge::AscendString name;
            node.GetName(name);
            int64_t idx = -1;
            node.GetAttr(ge::AscendString("index"), idx);
            indexed_inputs[idx] = name.GetString();
        }
    }
    // 按 index 排序存入 input_names_
    for (auto& [idx, name] : indexed_inputs) {
        input_names_.push_back(name);
    }
    
    // 解析输出：遍历 NetOutput 节点
    for (const auto& node : graph.GetDirectNode()) {
        ge::AscendString type;
        node.GetType(type);
        if (type == "NetOutput") {
            for (size_t i = 0; i < node.GetInputsSize(); ++i) {
                auto [src_node, src_port] = node.GetInDataNodesAndPortIndexs(i);
                ge::AscendString name;
                src_node->GetName(name);
                output_names_.push_back(name.GetString());
            }
        }
    }
}
```

#### 13.12.6 完整调用流程

```
RecWorkerImpl::prepare_inputs(batch)
  └─ batch.prepare_forward_input()              ← 模型无关，已有
       → ForwardInput (tokens, positions, attention, sampling, ...)
       → MMData 在 ForwardInput.input_params.multimodal.mm_data 中

GeGraphWorkerPipeline::prepare_work_before_execute()
  └─ Base: H2D + KV block swap                 ← 模型无关，已有
     （无需额外处理，MMData 随 ForwardInput 正常流转）

GeGraphExecutorImpl::run(tokens, positions, kv_caches, params)
  └─ EpModel::forward(tokens, positions, kv_caches, params)
       └─ BuildGraphInputs()
            ├─ "input_ids"     → TorchToGe(tokens)
            ├─ "positions"     → TorchToGe(positions)
            ├─ "k_cache[0]"    → TorchToGe(kv_caches[0].get_k_cache())
            ├─ "sparse_embedding" → TorchToGe(params.multimodal.mm_data.get("sparse_embedding"))
            └─ ...按 input_names_ 顺序组装 vector<gert::Tensor>
```

#### 13.12.7 不同模型类型的 input_names_ 示例

**LLM 模型**（如 Qwen2）：
```
input_names_ = ["input_ids", "position_ids", 
                "past_key_values[0].key", "past_key_values[0].value", ...]

BuildGraphInputs 匹配:
  "input_ids"              → IsTokenIds → tokens
  "position_ids"           → IsPositions → positions
  "past_key_values[0].key" → ParseKVCacheName → kv_caches[0].get_k_cache()
  ...
  无自定义输入 → 全部从函数参数取
```

**Rec 模型**：
```
input_names_ = ["tokens", "positions", "sparse_embedding",
                "k_cache[0]", "v_cache[0]", ...]

BuildGraphInputs 匹配:
  "tokens"            → IsTokenIds → tokens
  "positions"         → IsPositions → positions
  "sparse_embedding"  → 不匹配任何标准模式 → params.multimodal.mm_data.get("sparse_embedding")
  "k_cache[0]"        → ParseKVCacheName → kv_caches[0].get_k_cache()
  ...
```

#### 13.12.8 方案对比总结

| 维度 | A: Pipeline 硬编码 | B: input_tensor_map | **C: 有序名字列表** |
|------|-------------------|--------------------|--------------------|
| ModelInputParams 改动 | 无 | 新增字段 | **无** |
| Pipeline 改动 | 每模型改 Pipeline | 需 Schema 驱动填充 | **零改动** |
| 代码量 | Pipeline 膨胀 | ~200 行 | **~30 行** |
| 新增模型 | 改 Pipeline | 改 Model override | **自动适配（名字匹配）** |

### 13.13 已确认与待确认事项

#### 13.13.1 已确认

| # | 事项 | 决策 |
|---|---|---|
| 1 | GE 图输出格式 | beam search 在图内完成，输出为 `beam_sequence_group`（最终结果），Pipeline 不需要 Sampler/BeamSearcher |
| 2 | 输出 D2H 策略 | 使用异步 D2H 拷贝 + `Device::synchronize_default_stream()`，参考 `OneRecXAttentionEnginePipeline::get_model_output()` |
| 3 | 非 beam search 场景 | GE 图始终执行 beam search，非 beam 请求 beam_width=1 |
| 4 | EpairModelLoader 成员类型 | `std::unique_ptr<td::EpairModelLoader>`（无默认构造函数） |
| 5 | epair 路径获取 | `loader->model_weights_path() + "/model.epair"` |
| 6 | GE 初始化 | `std::call_once` + `GEInitializeV2()`，进程级别 |
| 7 | GE 终结 | `GEFinalizeV2()` 在进程退出时调用（atexit 或 ~EpModel 最后一个实例） |

#### 13.13.2 待确认

| # | 事项 | 影响 |
|---|---|---|
| 1 | 动态 shape 输出的具体处理 | 如果 epair 输出含动态维度（如 batch），需确认 GE 是否支持运行时 shape 变化 |
| 2 | 多卡 TP 场景下 GE 图的 HCCL 通信 | 确认 GE 图内部是否已包含 allreduce 等集合通信算子 |
| 3 | `CompileAndLoad` 的 stream 参数 | 编译阶段 stream 传 `nullptr` 是否对所有 epair 都安全 |
| 4 | 图输入统一为 MMData | 当前 `BuildGraphInputs` 分两路取数据（标准输入走 `forward()` 函数参数，自定义输入走 MMData）。如果后续将 tokens/positions/kv_caches 也统一塞进 MMData，`BuildGraphInputs` 可简化为只从 MMData 按名字取，消除 `IsTokenIds`/`IsPositions`/`ParseKVCacheName` 等判断逻辑。需要评估客户端请求如何组装、Pipeline 层如何传递 |