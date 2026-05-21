# DeepSeek V4 NPU GE 双图方案设计规格

> 版本：v2.0
> 日期：2026-05-21
> 状态：待实现

---

## 一、核心架构决策

### 1.1 Graph 包含完整推理流程（而非仅 DecoderLayers）

**关键修正**（基于 DS4 Python 实现和 GE Demo）：

| 维度 | 错误理解 | 正确理解 |
|------|---------|---------|
| **Graph 包含** | 只有中间 43 DecoderLayers | **完整推理流程 6 步** |
| **Graph 输入** | hidden_states（embedding 之后） | **input_ids（原始 tokens）** |
| **构建方式** | 逐层构建独立 Operation | **一个 EsGraphBuilder 构建整个 Model** |

**完整推理流程**：

```
Graph 包含：
├─ Step 1: Embedding lookup          (input_ids → hidden_states)
├─ Step 2: HyperConnection expansion ([batch,seq] → [batch,seq,hc_mult,hidden])
├─ Step 3: 43 DecoderLayers          (循环构建)
├─ Step 4: HyperConnection Head      (合并 hc_mult 维度)
├─ Step 5: Final RMSNorm             (hidden → normalized)
└─ Step 6: LM Head                   (hidden → logits)
```

**证据**：
- DS4 Python `modeling_deepseek.py:2011-2019`：`main_decode()` 包含完整 forward
- GE Demo `make_transformer_graph.cpp:50-86`：一个 builder 上直接调用算子

---

### 1.2 权重作为 Graph 输入（而非固化到 Graph）

**关键修正**：

| 维度 | 错误理解 | 正确理解 |
|------|---------|---------|
| **权重处理** | 固化到 Graph（编译时确定） | **作为 Graph 输入（每次执行传入约 565 个）** |
| **frozen_parameter** | 编译时优化 | **运行时优化（地址不变 → 缩短下发时间）** |
| **成员变量** | 编译后可释放 | **每次执行需要（不可释放）** |
| **Rolling Load** | 不讨论 | **不支持（地址不变要求 → 报错处理）** |

**原因**：GE ES C++ 没有 nn.Module 自动引用机制，必须显式传入权重。

**frozen_parameter=true 正确含义**：
- 适用对象：权重类输入（weights as inputs）
- 作用时机：运行时（Graph 执行），不是编译时
- 优化内容：权重地址不变 → 跳过地址检查 → 缩短 Graph 下发时间
- 前提条件：权重 tensor 地址不变 → 成员变量不可释放 → 不支持 Rolling Load

---

### 1.3 ATB vs GE ES 架构对比

| 维度 | ATB (DS2) | GE ES (DS4) |
|------|-----------|-------------|
| **构建模式** | 逐层构建 Operation | 一次性构建整个图 |
| **Graph 组成** | Operation 组合 | 直接包含所有算子 |
| **Graph 输入** | hidden_states | **input_ids** |
| **Graph 包含** | 只有中间 layers | **完整推理流程** |
| **权重处理** | Operation 内部引用 | **作为 Graph 输入传入** |
| **Layer 抽象** | 独立 Operation | **没有 Layer Operation** |

---

## 二、双图方案设计

### 2.1 核心流程

```
┌───────────────────────────────────────────────────────────────┐
│  Step 1: ES 构图（一次）                                       │
│                                                               │
│  build_model_graph()                                          │
│  ├─ 创建 EsGraphBuilder("ds4_model")                         │
│  ├─ 创建约 565 个权重输入 placeholder                           │
│  ├─ 构建完整推理流程（Embedding → 43 Layers → ... → LM Head） │
│  └─ BuildAndReset() → model_graph_                            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
                        ↓
          ┌─────────────┴─────────────┐
          ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│  Prefill Graph   │      │  Decode Graph    │
│  compile (≈5min) │      │  compile (≈5min) │
└──────────────────┘      └──────────────────┘
          ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│  -1 完全动态     │      │  dynamicDims分档 │
│  batch: {1..N}   │      │  batch: 1,4,8,...│
│  用于 PREFILL    │      │  用于 DECODE     │
└──────────────────┘      └──────────────────┘

总编译时间：约 10min（可控）
```

### 2.2 执行路径选择

```cpp
ModelOutput GeGraphExecutorImpl::run(...) {
  if (params.batch_forward_type.is_prefill()) {
    return run_prefill_graph(...);    // Prefill Graph
  }
  if (params.batch_forward_type.is_decode()) {
    return run_decode_graph(...);     // Decode Graph
  }
  // MIXED → eager mode fallback
  return model_->forward(...);
}
```

### 2.3 双图对比

| 维度 | Prefill Graph | Decode Graph |
|------|---------------|--------------|
| **ES 构建** | 同一个 graph（只构建一次） | 同一个 graph |
| **动态 shape** | `-1`（完全动态） | `dynamicDims`（分档） |
| **配置** | `ge.inputShape: "-1"` | `ge.inputShape: "-1"` + `ge.dynamicDims` |
| **编译时间** | ≈5min | ≈5min |
| **适用 ForwardType** | PREFILL / CHUNKED_PREFILL | DECODE |
| **batch 范围** | 动态（1-数千 tokens） | 固定分档（1,4,8,16,32,64,128） |

### 2.4 ForwardType 处理策略

| ForwardType | 执行方式 | Graph | 说明 |
|-------------|---------|-------|------|
| **PREFILL** | Prefill Graph | ✅ | 纯 Prefill batch |
| **CHUNKED_PREFILL** | Prefill Graph | ✅ | 分块 Prefill |
| **DECODE** | Decode Graph | ✅ | 纯 Decode batch |
| **MIXED** | Eager mode | ❌ | 包含 Prefill+Decode，无法用单一 graph |
| **EMPTY** | 直接返回 | ❌ | 空 batch |

**MIXED 不使用 Graph 的原因**：
- MIXED batch 中 Prefill sequences 和 Decode sequences 算子内部行为不同（如 Attention mask）
- Graph 无法实现 per-sequence 判断
- 只能 eager mode fallback

### 2.5 MIXED Eager 模式的理论 Bug

**核心问题**：`is_prefill` 是 binary batch-level flag，对于 MIXED batch，无论设为 `true` 还是 `false`，都无法对所有 sequence 正确。

**MIXED 的 `is_prefill` 设置**：
- `batch_forward_type.is_prefill()` 定义为 `value_ == PREFILL`（`batch_forward_type.h:51`）
- MIXED (value=3) → `is_prefill = false`
- 但 MIXED (value=3) → `is_chunked_prefill = true`（`attention_metadata_builder.cpp:98-100`）

**DS4 Attention 中 `is_prefill=false` 对 Prefill sequences 的 5 个错误**（`deepseek_v4_attention.cpp`）：

| Bug | 代码位置 | Prefill sequences 应有行为 | MIXED (is_prefill=false) 实际行为 | 影响 |
|-----|---------|--------------------------|--------------------------------|------|
| **1. KV 写入** | line 406-469 | 多 token 写入（ring buffer wrapping） | 单 token 写入 `slot_idx = (seqlen-1)%window_size_` | Prefill KV 丢失 |
| **2. KV 读取** | line 713-717 | 使用新鲜 KV `kv.unsqueeze(1)` | 使用 paged cache `k_cache`（Prefill 的 cache 因 Bug1 写入错误，数据无效） | Attention 输入错误 |
| **3. offsets** | line 597-603 | `offsets = query_lens`（per-sequence） | `offsets = torch::full_like(query_lens, window_size_)`（固定 decode 值） | 窗口索引错误 |
| **4. KV merge** | line 643-663 | 合并原始 KV + 压缩 KV | **跳过 merge**（no else branch） | 压缩 KV 数据丢失 |
| **5. block_table** | line 516-535 | `batch_offset` 直接偏移 | `block_tables.index(...)` decode mapping | KV cache 定位错误 |

**RoPE Bug**（`rotary_embedding.cpp:235-250`）：

```cpp
if (!is_prompt) {          // MIXED: is_prompt=false
  max_query_len = 1;       // ← 强制设为 1，Prefill sequences 有 query_len > 1
}
```

Prefill sequences 的 multi-token 位置编码被截断为单 token，RoPE 计算错误。

**根本原因**：

```
is_prefill 是 batch-level binary flag：
├─ is_prefill=true  → Prefill sequences 正确，Decode sequences 错误
├─ is_prefill=false → Decode sequences 正确，Prefill sequences 错误
└─ 任何值都无法同时正确处理 MIXED 中的两种 sequences
```

**部分缓解**：
- `is_chunked_prefill=true` 让标准 Attention（NPU/MLU/FlashInfer）路由到 chunked_prefill path，对 Decode sequences 基本正确
- 但 DS4 Attention 直接使用 `is_prefill`，绕过了 `is_chunked_prefill` 缓解
- Indexer/Compressor 有 per-sequence 处理（`query_lens`、`start_positions`），但结果被下游 batch-level 决策覆盖

**结论**：MIXED eager mode **理论上存在 bug**，Prefill sequences 在 DS4 Attention 中计算错误。当前框架假设 MIXED 频率低（≈10%），实际影响取决于 chunked prefill 配置。需在上线前验证，必要时通过 Scheduler 配置避免 MIXED。

---

## 三、权重输入设计

### 3.1 权重数量和类别

| 权重类别 | 数量 | 来源（成员变量） |
|---------|-----|----------------|
| **input_ids** | 1 | 用户输入 |
| **Embedding** | 1 | `embed_tokens_->weight()` |
| **43 Layers** | ~860 | `layers_[i]->get_all_weights()`（每层约 20 个，bf16 模式） |
| **Final Norm** | 1 | `norm_->weight()` |
| **LM Head** | 1 | `lm_head_->weight()` |
| **HC Head** | 3 | `hc_head_fn_, hc_head_base_, hc_head_scale_` |
| **总计** | **≈866** | bf16 ND + Data 输入（方案 1 起步） |

### 3.2 单层权重细分（约 13 个）

```
单层权重（bf16 模式）：
├─ HyperConnection pre-attention: ~3 个 (fn, base, scale)
├─ Attention: ~6 个 (norm, wq_a, wkv, wq_b, wo_a, wo_b)
├─ HyperConnection post-attention: ~3 个
├─ HyperConnection pre-FFN: ~3 个 (fn, base, scale)
├─ MoE: ~4 个 (norm, gate, gate_up/down_proj, shared_expert)
├─ HyperConnection post-FFN: ~3 个
└─ 单层总计：约 20 个
```

### 3.3 输入顺序约束：Graph 构建顺序 = 执行传入顺序（必须保序）

```
Graph 构建（按顺序创建 placeholder）：
  input_index=0:  CreateInput(0, "input_ids", DT_INT32, {-1})
  input_index=1:  CreateInput(1, "embed_weight", DT_BF16, {vocab_size, hidden_size})
  input_index=2-862: for 43 layers × ~20 weights (bf16 ND)
  input_index=863: CreateInput(863, "norm_weight", ...)
  input_index=864: CreateInput(864, "lm_head_weight", ...)
  input_index=865-867: HC Head (fn, base, scale)
  input_index=868+: KV Cache inputs + metadata inputs (frozen=0)

Graph 执行（按相同顺序传入 tensor）：
  inputs[0] = tokens
  inputs[1] = embed_tokens_->weight()
  inputs[2-862] = layers_[i]->get_all_weights() for each layer
  inputs[863] = norm_->weight()
  inputs[864] = lm_head_->weight()
  inputs[865-867] = hc_head_fn_, hc_head_base_, hc_head_scale_
  inputs[868+] = KV Cache + metadata (frozen=0)

顺序不一致 → 数据类型/shape 错误 → 推理失败
```

### 3.4 成员变量生命周期

| Phase | 作用 | 成员变量状态 |
|-------|------|------------|
| **Phase 1**: 权重加载 | `load_state_dict()` 加载权重到成员变量 | **必要** |
| **Phase 2**: Graph 构建 | 从成员变量提取 shape/dtype 创建 placeholder | **必要** |
| **Phase 3**: Graph 编译 | `frozen_parameter=true` → 地址必须不变 | **必要** |
| **Phase 4**: 每次执行 | **传入成员变量权重 tensor**（每次 RunGraph） | **必要** |

**结论**：成员变量在所有阶段都必要，**不可释放**。

### 3.5 Rolling Load 不支持

```cpp
void free_weights() {
  LOG(ERROR) << "Rolling Load not supported in GE Graph mode";
  CHECK(false) << "frozen_parameter=true requires address stability.";
}

void reload_weights() {
  LOG(ERROR) << "Rolling Load not supported in GE Graph mode";
  CHECK(false) << "frozen_parameter=true requires address stability.";
}
```

**原因**：
1. `frozen_parameter=true` 要求权重地址不变 → free_weights/reload_weights 导致地址变化
2. Graph 整体执行 → 无法中途更新权重输入
3. 默认配置 `enable_rolling_load=false`（大部分场景 HBM 充足）

**内存需求**：约 44 GB（所有权重常驻 HBM）

---

## 四、代码框架

### 4.1 核心类

#### GeGraphExecutorImpl（Runtime 层）

```cpp
// xllm/core/runtime/ge_graph_executor_impl.h

class GeGraphExecutorImpl : public ExecutorImpl {
 public:
  GeGraphExecutorImpl(CausalLM* model, const ModelArgs& args,
                      const torch::Device& device, const runtime::Options& options);
  ~GeGraphExecutorImpl() override;

  ForwardInput prepare_inputs(Batch& batch) override;
  ModelOutput run(const torch::Tensor& tokens, const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches, const ModelInputParams& params) override;

 private:
  CausalLM* model_;
  ge::Session* session_;
  uint32_t prefill_graph_id_;
  uint32_t decode_graph_id_;

  ModelOutput run_prefill_graph(const torch::Tensor& tokens, ...);
  ModelOutput run_decode_graph(const torch::Tensor& tokens, ...);
  std::vector<ge::Tensor> convert_inputs(...);
  ModelOutput convert_outputs(const std::vector<ge::Tensor>& ge_outputs);
};

REGISTER_EXECUTOR("npu_ge", GeGraphExecutorImpl);
```

#### GeTensorConverter

```cpp
// xllm/core/runtime/ge_tensor_converter.h

class GeTensorConverter {
 public:
  static ge::Tensor torch_to_ge(const torch::Tensor& torch_tensor);  // 零拷贝
  static torch::Tensor ge_to_torch(const ge::Tensor& ge_tensor);     // 零拷贝
 private:
  static ge::DataType map_dtype(torch::ScalarType torch_dtype);
  static ge::Format infer_format(const std::vector<int64_t>& sizes);
};
```

#### DeepseekV4GeModelImpl（Model 层）

```cpp
// xllm/models/llm/npu_ge/deepseek_v4.h

class DeepseekV4GeModelImpl : public torch::nn::Module {
 public:
  DeepseekV4GeModelImpl(const ModelContext& context);
  void load_state_dict(const StateDict& state_dict);
  void init_model();
  torch::Tensor forward(torch::Tensor& tokens, torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches, const ModelInputParams& params);

  // 权重收集（保序）
  std::vector<torch::Tensor> collect_all_weights();

 private:
  torch::nn::Embedding embed_tokens_{nullptr};
  std::vector<DeepseekV4GeDecoderLayer> layers_;
  RMSNorm norm_{nullptr};
  torch::nn::Linear lm_head_{nullptr};
  torch::Tensor hc_head_fn_, hc_head_base_, hc_head_scale_;

  ge::Session* session_;
  std::unique_ptr<ge::Graph> model_graph_;
  uint32_t prefill_graph_id_;
  uint32_t decode_graph_id_;

  void build_model_graph();
  void compile_prefill_graph();
  void compile_decode_graph();

  // Rolling Load 报错
  void free_weights();    // CHECK(false)
  void reload_weights();  // CHECK(false)
};

REGISTER_CAUSAL_MODEL(deepseek_v4_npu_ge, DeepseekV4GeForCausalLM);
```

### 4.2 关键实现逻辑

#### init_model() - 启动流程

```cpp
void DeepseekV4GeModelImpl::init_model() {
  auto& session_manager = ge::GeSessionManager::get_instance();
  session_ = session_manager.get_or_create_session();

  // Step 1: ES 构建（一次）
  build_model_graph();

  // Step 2: 编译 Prefill Graph
  compile_prefill_graph();  // -1 dynamic shape, ≈5min

  // Step 3: 编译 Decode Graph
  compile_decode_graph();   // dynamicDims: 1,4,8,16,32,64,128, ≈5min

  LOG(INFO) << "DS4 GE model initialized (dual graph, ≈10min compile)";
}
```

#### build_model_graph() - Graph 构建

```cpp
void DeepseekV4GeModelImpl::build_model_graph() {
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_model");

  // Step 1: 创建输入 placeholder（约 566 个，保序）
  int input_idx = 0;
  auto input_ids = builder->CreateInput(input_idx++, "input_ids", ge::DT_INT32, {-1});

  // 权重 placeholder（从 weight tensor 提取 shape/dtype）
  auto weights = collect_all_weights();
  std::vector<ge::es::EsTensorHolder> weight_inputs;
  for (auto& w : weights) {
    weight_inputs.push_back(builder->CreateInput(
        input_idx++, "", GeTensorConverter::map_dtype(w.scalar_type()), w.sizes().vec()));
  }

  // Step 2: Embedding lookup
  auto hidden = Embedding(input_ids, weight_inputs[0]);

  // Step 3: HyperConnection expansion
  hidden = Reshape(hidden, {-1, hc_mult_, hidden_size_});

  // Step 4: 43 DecoderLayers（直接在 builder 上构建）
  for (int layer = 0; layer < n_layers_; layer++) {
    auto layer_weights = get_layer_weights(weight_inputs, layer);
    hidden = build_decoder_layer_ops(builder, hidden, positions, layer_weights);
  }

  // Step 5: HyperConnection Head
  hidden = HyperConnectionHead(hidden, weight_inputs[563], weight_inputs[564], weight_inputs[565]);

  // Step 6: Final Norm
  hidden = RMSNorm(hidden, weight_inputs[561]);

  // Step 7: LM Head
  auto logits = MatMul(hidden, weight_inputs[562]);

  // Build graph
  model_graph_ = builder->BuildAndReset({logits});
}
```

#### compile_prefill_graph() / compile_decode_graph()

```cpp
void DeepseekV4GeModelImpl::compile_prefill_graph() {
  std::map<ge::AscendString, ge::AscendString> options = {
    {"ge.inputShape", "input_ids:-1"},  // 完全动态
    {"frozen_parameter", "true"}         // 权重地址不变
  };

  prefill_graph_id_ = session_manager.next_graph_id();
  session_->AddGraph(prefill_graph_id_, *model_graph_, options);
  session_->CompileGraph(prefill_graph_id_);  // ≈5min
}

void DeepseekV4GeModelImpl::compile_decode_graph() {
  std::map<ge::AscendString, ge::AscendString> options = {
    {"ge.inputShape", "input_ids:-1"},
    {"ge.dynamicDims", "1,4,8,16,32,64,128"},
    {"frozen_parameter", "true"}
  };

  decode_graph_id_ = session_manager.next_graph_id();
  session_->AddGraph(decode_graph_id_, *model_graph_, options);
  session_->CompileGraph(decode_graph_id_);  // ≈5min
}
```

#### run_prefill_graph() / run_decode_graph() - Graph 执行

```cpp
ModelOutput GeGraphExecutorImpl::run_prefill_graph(...) {
  // Step 1: 准备 GE inputs（保序：input_ids + 约 565 个权重）
  std::vector<ge::Tensor> inputs;
  inputs.push_back(GeTensorConverter::torch_to_ge(tokens));

  auto model = static_cast<DeepseekV4GeModelImpl*>(model_);
  auto weights = model->collect_all_weights();
  for (auto& w : weights) {
    inputs.push_back(GeTensorConverter::torch_to_ge(w));
  }

  // Step 2: 执行
  std::vector<ge::Tensor> outputs;
  auto stream = c10_npu::getCurrentNPUStream().stream();
  session_->RunGraphWithStreamAsync(prefill_graph_id_, stream, inputs, outputs);
  aclrtSynchronizeStream(stream);

  // Step 3: 转换输出
  return convert_outputs(outputs);
}
```

#### collect_all_weights() - 权重收集（保序）

```cpp
std::vector<torch::Tensor> DeepseekV4GeModelImpl::collect_all_weights() {
  std::vector<torch::Tensor> weights;

  weights.push_back(embed_tokens_->weight());
  for (int i = 0; i < layers_.size(); i++) {
    auto layer_weights = layers_[i]->get_all_weights();
    weights.insert(weights.end(), layer_weights.begin(), layer_weights.end());
  }
  weights.push_back(norm_->weight());
  weights.push_back(lm_head_->weight());
  weights.push_back(hc_head_fn_);
  weights.push_back(hc_head_base_);
  weights.push_back(hc_head_scale_);

  return weights;  // ≈565 个，顺序与 Graph placeholder 一致
}
```

### 4.3 Tensor 转换（零拷贝）

```cpp
ge::Tensor GeTensorConverter::torch_to_ge(const torch::Tensor& torch_tensor) {
  auto sizes = torch_tensor.sizes().vec();
  auto data_ptr = torch_tensor.data_ptr();
  ge::DataType ge_dtype = map_dtype(torch_tensor.scalar_type());

  ge::TensorDesc desc(ge::Shape(sizes), ge::FORMAT_ND, ge_dtype);
  ge::Tensor ge_tensor;
  ge_tensor.SetTensorDesc(desc);
  ge_tensor.SetData(reinterpret_cast<uint8_t*>(data_ptr),
                    torch_tensor.numel() * torch_tensor.element_size());
  return ge_tensor;
}

ge::DataType GeTensorConverter::map_dtype(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:  return ge::DT_FLOAT;
    case torch::kFloat16:  return ge::DT_FLOAT16;
    case torch::kBFloat16: return ge::DT_BF16;
    case torch::kInt32:    return ge::DT_INT32;
    case torch::kInt64:    return ge::DT_INT64;
    default: LOG(FATAL) << "Unsupported dtype"; return ge::DT_UNDEFINED;
  }
}
```

---

## 五、已创建的代码文件

### Runtime 层（xllm/core/runtime/）

| 文件 | 状态 |
|------|------|
| `ge_graph_executor_impl.h/.cpp` | ✅ 框架（核心逻辑 TODO） |
| `ge_tensor_converter.h/.cpp` | ✅ 框架（零拷贝 TODO） |
| `ge_session_manager.h/.cpp` | ✅ 框架 |

### Model 层（xllm/models/llm/npu_ge/）

| 文件 | 状态 |
|------|------|
| `deepseek_v4.h/.cpp` | ✅ 框架（graph 构建 TODO） |
| `CMakeLists.txt` | ✅ 注释状态 |

### Layer 层（xllm/core/layers/npu_ge/）

| 文件 | 状态 |
|------|------|
| `CMakeLists.txt` | ✅ 注释状态 |

### CMake 配置（待修改）

| 文件 | 修改内容 |
|------|---------|
| `xllm/CMakeLists.txt` | 添加 `USE_GE` option + GE 库依赖 |
| `xllm/core/runtime/CMakeLists.txt` | 添加 GE sources |
| `xllm/core/layers/CMakeLists.txt` | 添加 `add_subdirectory(npu_ge)` |
| `xllm/models/llm/CMakeLists.txt` | 添加 `add_subdirectory(npu_ge)` |

---

## 六、实施计划

### Phase 1: 辅助方法（P0）

1. `collect_all_weights()` - 收集所有权重（保序，约 866 个 bf16 ND）
2. `get_all_weights()` - 每个 layer 的权重提取
3. `GeTensorConverter::map_dtype()` - dtype 映射
4. ES wrapper for key DS4 ops (Phase 0: 2-3 个验证)

### Phase 2: GE ES API 实现（P1）

1. 全部 bf16 模式算子 ES wrapper（约 15 个，参考 TorchAir converter）
2. `build_model_graph()` - 创建权重 placeholder + 构建算子 + frozenInput 配置
3. `compile_prefill_graph()` - 编译 Prefill Graph
4. `compile_decode_graph()` - 编译 Decode Graph

### Phase 3: Graph 执行（P1）

1. `run_prefill_graph()` - 执行（传入 input_ids + 约 565 个权重）
2. `run_decode_graph()` - 执行（同上）
3. 权重 tensor 传入逻辑（保序）

### Phase 4: 测试验证（P2）

1. 验证权重输入顺序正确性
2. 验证 frozen_parameter=true 效果
3. 验证推理正确性（Prefill + Decode）
4. 验证 MIXED fallback

---

## 七、方案 A vs 方案 B（未来优化）

| 维度 | 方案 A（当前） | 方案 B（未来） |
|------|-------------|-------------|
| **权重合并** | 不合并（约 565 个独立输入） | 合并（约 6 个大 buffer 输入） |
| **优点** | 实现简单、权重管理清晰 | 输入少、Graph 构建简单、执行效率高 |
| **缺点** | 输入多、placeholder 创建复杂 | 需要 WeightBuffer 类、切片逻辑复杂 |
| **计划** | **打通功能优先** | 性能优化阶段 |

---

## 八、风险

| 风险 | 影响 | 应对 |
|------|------|------|
| GE ES API 不完全可用 | 编译/执行失败 | 先测试 GE Demo Transformer example |
| CompileGraph >5min | 启动时间过长 | 测试单层编译时间，评估分档策略 |
| GE ops 缺少 DS4 算子 | 推理失败 | 列出算子清单，与 GE 团队确认 |
| 权重输入顺序不一致 | 推理结果错误 | collect_all_weights 保序 + 单元测试 |
| MIXED 频率不可控 | 性能波动 | 监控 MIXED 比率，必要时强制分离 |

---

## 九、ES 构图可行性分析（参考 Python ge_graph 流程）

> 基于 `cann-recipes-infer/models/deepseek-v4/docs/deepseek_v4_execution_flow.md` 的 Python ge_graph 已跑通流程，
> 评估在 xLLM 中使用 ES 构图实现双图方案的可行性。

### 9.1 Python ge_graph 流程核心机制

```
model.decode() → torch.compile(dynamic=False, fullgraph=True, backend=torchair_npu_backend)
  ├─ Step 1: Dynamo 符号追踪 → FX Graph（自动捕获所有 nn.Parameter 引用）
  ├─ Step 2: AOTAutograd 参数提升 → 所有 nn.Parameter 变为 placeholder 输入
  ├─ Step 3: TorchAir 后端 → FX node → GE IR 算子（通过 converter registry）
  ├─ Step 4: frozen_parameter=True → Data → ConstPlaceHolder（注入 NPU 地址）
  ├─ Step 5: GE 引擎编译执行（整网下发）
  │
  ├─ Prefill: eager mode（不入图）
  └─ Decode: GE graph（torch.compile 编译后的缓存图执行）
```

**关键配置**（`runner_deepseek.py:162-180`）：
- `dynamic=False`：Decode 输入 shape 固定（单 token）
- `fullgraph=True`：整个函数在单个图内完成
- `frozen_parameter=True`：固定 nn.Parameter 内存地址（ConstPlaceHolder）
- `is_prefill=False` 作为常量：所有分支只走 Decode 路径
- `enable_npugraph_ex=False`：禁用 MLA 多流（ge_graph 模式下单 stream）
- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:False`：保证 NPU 地址不变

### 9.2 ES 构图 vs Python ge_graph 核心差异

| 维度 | Python (TorchAir) | ES 构图 (xLLM C++) |
|------|-------------------|-------------------|
| **构图方式** | `torch.compile` 自动追踪 → FX → TorchAir → GE IR | **手动**：`EsGraphBuilder` + 逐算子调用 |
| **权重入图** | AOTAutograd 自动 lift nn.Parameter → ConstPlaceHolder | **手动**：`CreateInput`（Data 输入）+ session `frozenInput` 选项 |
| **算子覆盖** | TorchAir converter registry（20+ DS4 converter） | **ES 内置仅基础算子**，DS4 专用需自定义 wrapper |
| **ConstPlaceHolder** | TorchAir 自动：Data → ConstPlaceHolder（注入地址） | ES 不暴露 ConstPlaceHolder API → 用 Data 输入 + frozenInput |
| **量化算子** | `npu_quant_matmul`/`npu_rms_norm_dynamic_quant` 等 converter 已有 | ES 无内置 → 需自定义 wrapper（GE IR 定义已有） |
| **HCCL 通信** | `torchair.patch_for_hcom()` 注册入图 | ES 有 `HcomAllGather/HcomAllReduce/HcomReduceScatter` ✅ |

### 9.3 DS4 Decode 算子清单（ES 缺失分析）

**核心利好**：Python 已跑通 → 所有 DS4 专用算子的 **GE op proto 已注册**（IR 定义存在）。
ES wrapper 可用 `CompliantNodeBuilder.OpType("已有op名")` 直接引用，无需从零定义 IR。

| 算子 | GE Op Type (TorchAir Converter) | ES 内置 | 需 Custom Wrapper | 出现频率 |
|------|-------------------------------|---------|-----------------|---------|
| Embedding | aten→GE | ❌ | ✅ | 1 次 |
| HC Pre | `HcPre` | ❌ | ✅ | 每层 2× |
| HC Post | `HcPost` | ❌ | ✅ | 每层 2× |
| RMSNorm | `npu_rms_norm` (torch_npu) | ❌ | ✅ (ES 有 `AddRmsNorm` 但不匹配) | 每层 2-3× |
| RmsNormDynamicQuant | `RmsNormDynamicQuant` | ❌ | ✅ (w8a8int8 模式) | 每层 1× |
| MatMul (bf16) | aten→GE | ✅ `MatMul` | - | 每层多× |
| QuantMatMul | `npu_quant_matmul` (torch_npu) | ❌ | ✅ (量化模式) | 每层 2-3× |
| TransposeBatchMatMul | `npu_transpose_batchmatmul` | ❌ | ✅ | 每层 1× (wo_a) |
| InplacePartialRotaryMul | `InplacePartialRotaryMul` | ❌ | ✅ | 每层 3-5× |
| ScatterNdUpdateAsc | **无 GE converter**（原地写入） | ❌ | ✅ (需特殊处理) | 每层 1-2× |
| KvCompressEpilog | `KvCompressEpilog` (float8) | ❌ | ✅ | 每层 1× (float8) |
| Compressor | `Compressor` | ❌ | ✅ | C4A/C128A 层 |
| QuantLightningIndexer | `QuantLightningIndexer` | ❌ | ✅ | C4A 层 |
| SparseAttnSharedkv | `SparseAttnSharedkv` | ❌ | ✅ | 每层 1× |
| Metadata (SparseAttn) | `SparseAttnSharedkvMetadata` | ❌ | ✅ | 每步 |
| Metadata (LI) | `QuantLightningIndexerMetadata` | ❌ | ✅ | 每步 |
| MoeGatingTopKHash | `MoeGatingTopKHash` | ❌ | ✅ | 每层 1× |
| GroupedMatMul | `npu_grouped_matmul` | ❌ | ✅ | 每层 2× (MoE) |
| DequantSwigluClampQuant | `DequantSwigluClampQuant` | ❌ | ✅ | 每层 1× |
| MC2 Dispatch/Combine | `npu_moe_distribute_dispatch/combine_v2` | ❌ | ✅ | 每层 1× |
| HCCL AllGather | hcom patch | ✅ | - | lm_head 等 |
| HCCL ReduceScatter | hcom patch | ✅ | - | wo_b 等 |
| Final Norm | `npu_rms_norm` | ❌ | ✅ | 1 次 |

**统计**：约 **20 个**算子需自定义 ES wrapper。

**bf16 起步模式简化**：
- 不需要 `QuantMatMul` / `RmsNormDynamicQuant` / `DequantSwigluClampQuant` wrapper（用标准 bf16 MatMul + RMSNorm 代替）
- 不需要 NZ 格式处理
- 不需要 weight_scale / smooth_scales 输入
- **bf16 起步 wrapper 约 15 个**

### 9.4 关键待解决问题

#### 1. ConstPlaceHolder / frozen_parameter

**bf16 起步方案**：权重用 Data 输入 + session `frozenInput` 选项

```cpp
// ES 构图：权重作为 Data 输入
auto weight_input = builder->CreateInput(input_index, "layer_0_wq_a_weight",
                                          ge::DT_BF16, {7168, 1536});

// Session 配置：frozenInput 选项
// frozenInput 格式："1,1,...,1,0,0,...0"（1=权重 frozen，0=动态输入）
std::map<ge::AscendString, ge::AscendString> options = {
    {"ge.inputShape", "input_ids:-1"},
    {"ge.dynamicDims", "1,4,8,16,32,64,128"},
    {"frozenInput", frozen_input_str}  // 约 865 个 1 + 几个 0
};

// RunGraph：每次传入全部权重
std::vector<ge::Tensor> inputs;
inputs.push_back(convert(tokens));         // 动态输入 (frozen=0)
for (auto& w : collect_all_weights()) {    // 权重 (frozen=1)
    inputs.push_back(convert(w));
}
session_->RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
```

#### 2. KV Cache 原地写入

- Python `scatter_nd_update_asc` 原地写入（无 GE converter）
- **bf16 起步方案**：KV Cache 作为 Graph 输入+输出（传入传出）
  - 每次 RunGraph 传入 KV Cache tensor（frozen=0，地址可能变化）
  - Graph 输出包含更新后的 KV Cache
- **量化优化阶段**：自定义 `ScatterNdUpdate` wrapper 或用 `kv_compress_epilog`

#### 3. Metadata 生成算子

- **方案 A**（推荐起步）：Graph 外部生成 metadata，作为 Data 输入传入
  - 与 xLLM 现有 `generate_kernel_metadata()` 流程一致
  - metadata 是每步动态生成的（frozen=0）
- **方案 B**（未来优化）：Graph 内包含 metadata 算子
  - metadata 输出直接作为后续算子输入
  - 减少 Host-Device 数据传输

#### 4. 量化参数（bf16 起步不涉及）

- bf16 模式：标准 `MatMul(hidden, weight)` 即可
- 量化优化阶段：需 `QuantMatMul` wrapper（weight + weight_scale + pertoken_scale）

#### 5. NZ 格式权重（bf16 起步不涉及）

- bf16 ND 格式权重：ES Data 输入直接可用
- 量化优化阶段：需 ConstPlaceHolder 设置 `storage_format=NZ(format 29)`

### 9.5 权重适配方案：bf16 ND + Data 输入 + frozenInput（起步）

**起步阶段决策**：

| 维度 | 方案 |
|------|------|
| **权重精度** | bf16（xLLM 现有流程，无量化预处理） |
| **权重格式** | ND（标准连续，ES Data 输入直接可用） |
| **权重入图方式** | Data 输入 + session `frozenInput` 选项 |
| **权重数量** | 约 865 个 Data 输入（43×~20 + Embedding + Norm + LM Head + HC Head） |
| **每次执行** | 传入全部权重 tensor（frozen=1 → 地址校验跳过） |
| **ES MatMul** | bf16 ND `MatMul`（内置，无需 wrapper） |

**权重输入顺序（保序）**：

```
input_index=0:     input_ids (DT_INT32, {-1})         ← frozen=0
input_index=1:     embed_tokens_weight (DT_BF16)       ← frozen=1
input_index=2-862: 43 layers × ~20 weights per layer    ← frozen=1
input_index=863:   norm_weight (DT_BF16)                ← frozen=1
input_index=864:   lm_head_weight (DT_BF16)             ← frozen=1
input_index=865-868: hc_head_fn/base/scale (DT_BF16)   ← frozen=1
input_index=869+:  KV Cache inputs (DT_BF16)            ← frozen=0
input_index=N:     metadata inputs (DT_INT32 etc)       ← frozen=0
```

**frozenInput 选项构造**：

```cpp
// 构造 frozenInput 字符串："1,1,...1,0,0,...0"
std::string frozen_input_str;
for (int i = 0; i < total_inputs; i++) {
    if (i == 0 || i >= kv_cache_start_index) {  // 动态输入
        frozen_input_str += "0";
    } else {                                     // 权重输入
        frozen_input_str += "1";
    }
    if (i < total_inputs - 1) frozen_input_str += ",";
}
```

**性能优化阶段升级路径**：

| 维度 | 起步 (bf16) | 优化 (w8a8int8) |
|------|------------|-----------------|
| **权重精度** | bf16 ND | int8 NZ + bf16 scale |
| **权重入图** | Data 输入 + frozenInput | ConstPlaceHolder（自定义 wrapper） |
| **MatMul** | ES `MatMul` (内置) | `QuantMatMul` (自定义 wrapper) |
| **预处理** | 无 | transpose + NZ format cast + scale dtype cast |
| **RunGraph 传权重** | 每次传入 ~865 个 tensor | 不传权重（ConstPlaceHolder 引用地址） |

### 9.6 推荐实施路径

**核心策略**：所有 DS4 专用算子的 GE IR 定义已注册 → 用 `CompliantNodeBuilder.OpType("已有op名")` 构建 → 参照 TorchAir converter 确认属性/输入输出规范。

**TorchAir converter 代码位置**：`cann-recipes-infer/ops/ascendc/torch_ops_extension/custom_ops/converter/*.py`（23 个文件）

**每个 ES wrapper 实现流程**：

```
1. GetRegisteredIrDef("SparseAttnSharedkv") → 查询 IR 定义
2. 参照 TorchAir converter (npu_sparse_attn_sharedkv.py) → 确认属性映射
3. CompliantNodeBuilder.OpType("SparseAttnSharedkv")
     .IrDefInputsV2(...)      → 定义输入
     .IrDefOutputsV2(...)     → 定义输出
     .IrDefAttrsV2(...)       → 定义属性
     .Build()
4. AddEdgeAndUpdatePeerDesc(...) → 连接边
5. node.SetAttr(...)           → 设置算子属性
```

**bf16 起步阶段实施计划**：

| Phase | 内容 | 预估 |
|-------|------|------|
| **Phase 0**: 验证 | 2-3 个关键算子 ES wrapper（SparseAttnSharedkv + HcPre + RMSNorm） | 5-7 天 |
| **Phase 1**: wrapper | ~15 个 bf16 模式算子 ES wrapper | 10-15 天 |
| **Phase 2**: 构图 | ES 构建完整 DS4 Decode Graph（bf16 权重 Data 输入 + frozenInput） | 5-7 天 |
| **Phase 3**: 编译执行 | 双图编译 + RunGraphWithStreamAsync + 输出转换 | 3-5 天 |
| **Phase 4**: 测试 | Prefill/Decode 正确性 + 性能对比 | 3-5 天 |
| **总计** | | **约 25-35 天** |

**量化优化阶段**（后续）：

| Phase | 内容 | 预估 |
|-------|------|------|
| 量化预处理 | transpose + NZ format cast + scale dtype cast | 5-7 天 |
| ConstPlaceHolder wrapper | 自定义 wrapper + 地址注入 | 3-5 天 |
| QuantMatMul wrapper | 自定义 wrapper | 3-5 天 |
| MC2 Dispatch/Combine wrapper | 自定义 wrapper | 5-7 天 |
| 集成测试 | 量化模式正确性验证 | 5-7 天 |

### 9.7 Python ge_graph 流程参考文档

| 文档 | 路径 | 内容 |
|------|------|------|
| **DS4 执行流程** | `cann-recipes-infer/models/deepseek-v4/docs/deepseek_v4_execution_flow.md` | ge_graph 模式端到端流程详解 |
| **TorchAir converter** | `cann-recipes-infer/ops/ascendc/torch_ops_extension/custom_ops/converter/*.py` | 23 个 DS4 专用算子 converter |
| **Runner** | `cann-recipes-infer/models/deepseek-v4/runner_deepseek.py` | `graph_compile()` (162-180), `_process_weight_after_loading()` (100-159) |
| **YAML 配置** | `cann-recipes-infer/models/deepseek-v4/config/ci_a3/deepseek_v4.yaml` | ge_graph 模式默认配置 |
| **ES Demo** | `ge/examples/es/transformer/cpp/src/make_transformer_graph.cpp` | ES 构图模式示例 |
| **Custom ES API** | `ge/examples/custom_es_api/` | 自定义 ES wrapper 模式（`CompliantNodeBuilder`） |
| **GE Session V2** | `ge/inc/external/ge/ge_api_v2.h` | `GeSession` API（CompileGraph + RunGraphWithStreamAsync） |
| **ES All Ops** | `ge/output/include/es_all_ops.h`（生成文件） | 所有已注册 op 的 ES API |

---

## 十、参考资料

| 来源 | 路径 | 关键内容 |
|------|------|---------|
| **DS4 Python** | `cann-recipes-infer/models/deepseek-v4/models/modeling_deepseek.py` | `get_cached_graph()` (1736-1760), `main_decode()` (2011-2019), `forward()` (1467-1552) |
| **GE Demo** | `ge/examples/es/transformer/cpp/src/make_transformer_graph.cpp` | `MakeTransformerGraphByEs()` (50-86) |
| **DS4 MLU Model** | `xllm/models/llm/deepseek_v4.h` | MLU 版 Model 定义 |
| **DS2 ATB Model** | `xllm/models/llm/npu/deepseek_v2.h` | ATB 版对比 |
| **MLU Executor** | `xllm/core/runtime/mlu_graph_executor_impl.cpp` | Graph/eager 分支 (235) |
| **Rolling Load 默认** | `xllm/core/common/global_flags.cpp:507-516` | `enable_rolling_load=false` |
| **BatchForwardType** | `xllm/core/framework/batch/batch_forward_type.h` | PREFILL/DECODE/MIXED 定义 |

---

**文档结束**