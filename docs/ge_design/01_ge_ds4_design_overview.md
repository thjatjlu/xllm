# DeepSeek V4 NPU GE 图模式设计方案（初稿）

> 文档版本：v1.0
> 创建日期：2025-01-19
> 状态：初稿（待评审）

---

## 一、方案概述

### 1.1 核心设计决策

**基于 MLU 成功实践的关键洞察**：
- ✅ **只编译 Decode Graph**（不编译 Prefill graph）
- ✅ **Prefill 使用 eager mode**（避免分钟级编译开销）
- ✅ **Decode 使用 GE dynamicDims**（优化高频 batch）
- ✅ **编译时间：约 5min**（单次 CompileGraph）

### 1.2 设计哲学

**为什么 Prefill 不需要 Graph？**

| 维度 | Prefill 特征 | Decode 特征 |
|-----|-------------|------------|
| **Batch 来源** | Prompt token 数量（动态极大） | 并发请求数（相对固定） |
| **Batch 范围** | 1 到数千 tokens | 1 到 max_seqs（128） |
| **执行频率** | 每请求 1 次 | 每 token 1 次（高频） |
| **计算量** | 大（prompt_len × hidden） | 小（batch × hidden） |
| **Graph 收益** | 编译开销 > 执行收益 | 编译开销 < 执行收益 |
| **最优方案** | ❌ Eager mode | ✅ GE Graph |

---

## 二、架构设计

### 2.1 启动流程

```
启动阶段（约 5min）：
  
  1. WorkerImpl::init_model()
     ↓
  2. DeepseekV4ModelImpl::load_state_dict()
     - 加载权重（秒级）
     ↓
  3. NpuGeDeepseekV4ModelImpl::init_model()
     
     【只编译 Decode Graph】（约 5min）
     - 构建 GE Graph（包含所有 43 layers）
     - 配置 dynamicDims：{1,4,8,16,32,64,128}
     - AddGraph + CompileGraph
     ↓
  4. 服务就绪
     打印："Model ready (decode graph compiled)"
```

### 2.2 运行流程

```
运行阶段：
  
  GeGraphExecutorImpl::run(tokens, positions, kv_caches, params)
    
    if (params.batch_forward_type.is_decode()) {
      // DECODE → 使用 decode_graph
      
      1. 准备输入 tensor（实际 batch）
      2. torch::Tensor → ge::Tensor 转换（零拷贝）
      3. RunGraphWithStreamAsync（GE 执行）
      4. aclrtSynchronizeStream（同步）
      5. ge::Tensor → torch::Tensor 转换
      6. 返回 ModelOutput
    }
    
    else {
      // PREFILL / CHUNKED_PREFILL / MIXED → eager mode
      return model_->forward(tokens, positions, kv_caches, params);
    }
```

---

## 三、关键配置

### 3.1 Decode Graph 配置

```cpp
// Decode graph dynamicDims 配置
std::map<ge::AscendString, ge::AscendString> graph_options = {
  {"ge.inputShape", 
   "hidden_states:-1,4096;"
   "positions:-1;"
   "block_tables:-1,128;"
   "new_cache_slots:-1"},
   
  {"ge.dynamicDims", 
   "1,4,8,16,32,64,128"}  // ← 7 个高频 batch sizes
};
```

### 3.2 ForwardType 处理策略

| ForwardType | 处理方式 | Graph 使用 |
|------------|---------|-----------|
| **PREFILL** | Eager mode | ❌ 不使用 |
| **CHUNKED_PREFILL** | Eager mode | ❌ 不使用 |
| **DECODE** | GE Graph | ✅ 使用 decode_graph |
| **MIXED** | Eager mode | ❌ 不使用 |
| **EMPTY** | 直接返回 | ❌ 不使用 |

---

## 四、技术细节

### 4.1 KV Cache 传递方式

**参考 MLU 设计**：每次 RunGraph 传递 KV Cache

```cpp
// 每次运行时传递 KV Cache（转换为 GE tensor）
std::vector<ge::Tensor> ge_inputs;

ge_inputs.push_back(GeTensorConverter::torch_to_ge(tokens));
ge_inputs.push_back(GeTensorConverter::torch_to_ge(positions));

// KV Cache 传递
for (auto& kv_cache : kv_caches) {
  ge_inputs.push_back(GeTensorConverter::torch_to_ge(kv_cache.k_cache));
  ge_inputs.push_back(GeTensorConverter::torch_to_ge(kv_cache.v_cache));
}

ge_inputs.push_back(GeTensorConverter::torch_to_ge(block_tables));
ge_inputs.push_back(GeTensorConverter::torch_to_ge(new_cache_slots));
```

### 4.2 Tensor 转换（零拷贝）

```cpp
// torch::Tensor → ge::Tensor（零拷贝）
ge::Tensor torch_to_ge(const torch::Tensor& torch_tensor) {
  // 1. 提取 tensor 属性
  auto sizes = torch_tensor.sizes();
  auto strides = torch_tensor.strides();
  auto dtype = torch_tensor.scalar_type();
  auto data_ptr = torch_tensor.data_ptr();
  
  // 2. 创建 ge::TensorDesc
  ge::Shape shape(sizes.vec());
  ge::DataType ge_dtype = map_dtype(dtype);
  ge::Format format = ge::FORMAT_ND;
  
  ge::TensorDesc desc(shape, format, ge_dtype);
  
  // 3. 创建 ge::Tensor（直接使用 NPU memory，零拷贝）
  ge::Tensor ge_tensor;
  ge_tensor.SetTensorDesc(desc);
  ge_tensor.SetData(
    reinterpret_cast<uint8_t*>(data_ptr),
    torch_tensor.numel() * torch_tensor.element_size()
  );
  
  return ge_tensor;
}
```

### 4.3 算子差异处理

**关键洞察：两张图（Prefill vs Decode）独立编译**

```cpp
// Prefill phase（eager mode）
model_->forward(tokens, positions, kv_caches, params);
  - Attention: 使用 causal mask
  - MoE: 标准 routing
  - Indexer/Compressor: 不使用

// Decode phase（GE graph）
run_decode_graph(tokens, positions, kv_caches, params);
  - Attention: 使用 decode mask（1 token vs all KV）
  - MoE: Hash routing（前 3 layers）
  - Indexer/Compressor: 使用（动态选择重要 KV）
```

**算子配置通过 params 传递**：
- Prefill: `params.is_prefill = true`
- Decode: `params.is_prefill = false`

---

## 五、文件结构规划

### 5.1 新增文件

```
xllm/models/llm/npu_ge/
  ├── deepseek_v4.h                    # Model 定义
  ├── deepseek_v4.cpp                  # Model 实现
  └── CMakeLists.txt                   # 编译配置

xllm/core/runtime/
  ├── ge_graph_executor_impl.h         # Executor 定义
  ├── ge_graph_executor_impl.cpp       # Executor 实现
  ├── ge_tensor_converter.h            # Tensor 转换工具
  ├── ge_session_manager.h             # Session 管理
  └── CMakeLists.txt                   # Runtime 编译配置

xllm/core/layers/npu_ge/
  └── CMakeLists.txt                        # Layer 编译配置（预留）
```

### 5.2 修改文件

```
xllm/CMakeLists.txt                    # 添加 GE 依赖
xllm/core/runtime/executor_impl_factory.h  # 注册 Executor
xllm/models/model_registry.h           # 注册 Model
xllm/xllm.cpp                          # 支持 --backend=npu_ge
```

---

## 六、编译依赖

### 6.1 GE 库依赖

```cmake
# GE 核心库
set(GE_LIBS
  ge_runner       # Session + RunGraph
  ge_compiler     # CompileGraph
  graph           # Graph 结构
  es_all          # ES API（自动生成）
  ascendcl        # ACL runtime
)

# 头文件路径
include_directories($ENV{ASCEND_HOME_PATH}/include/ge)
include_directories(${CMAKE_BINARY_DIR}/output/include)  # ES generated
```

### 6.2 ES API 生成

```cmake
# 生成 ES operator builders
find_package(GenerateEsPackage REQUIRED)

add_es_library(
  ES_LINKABLE_AND_ALL_TARGET es_all
  OUTPUT_PATH ${CMAKE_BINARY_DIR}/output
)
```

---

## 七、性能预估

### 7.1 编译时间

| 项目 | 预估时间 |
|-----|---------|
| **权重加载** | 30s |
| **Decode Graph 编译** | 5min |
| **总启动时间** | **约 5.5min** |

### 7.2 推理性能

| Phase | 执行方式 | 预估性能 |
|------|---------|---------|
| **Prefill** | Eager mode | 与 DS2 ATB 相当 |
| **Decode** | GE Graph | **优于 DS2 ATB**（GE graph 优化） |

---

## 八、风险与应对

### 8.1 主要风险

| 风险 | 影响 | 应对措施 |
|-----|------|---------|
| **GE dynamicDims 不支持某些算子** | 编译失败 | 测试关键算子支持性 |
| **KV Cache 传递开销** | 性能下降 | 零拷贝优化 |
| **Tensor 转换错误** | 推理失败 | 完善单元测试 |
| **CompileGraph 失败** | 服务无法启动 | 详细错误日志 |

### 8.2 Fallback 策略

```cpp
// 如果 decode graph 不可用，回退到 eager mode
if (!decode_graph_available_) {
  LOG(WARNING) << "Decode graph not available, fallback to eager mode";
  return model_->forward(tokens, positions, kv_caches, params);
}
```

---

## 九、与现有方案对比

### 9.1 对比表

| 维度 | DS2 ATB | MLU | GE（本方案） |
|-----|---------|-----|------------|
| **Prefill graph** | ✅ 有 | ❌ 无 | ❌ **无（参考 MLU）** |
| **Decode graph** | ✅ 有 | ✅ 有 | ✅ **有** |
| **Prefill 处理** | ATB GraphOp | Eager | **Eager** |
| **Decode 处理** | ATB GraphOp | MLUGraph | **GE Graph** |
| **编译时间** | 秒级（ATB） | 5min | **5min** |
| **算子生态** | ATB ops | MLU ops | **GE ops** |

### 9.2 核心优势

**相比 DS2 ATB**：
- ✅ 更强的 graph 优化（GE compiler 优化能力）
- ✅ 更好的内存管理（GE graph memory planning）

**相比 MLU**：
- ✅ 相同的架构设计（参考成功实践）
- ✅ 相似的编译时间（约 5min）

---

## 十、待确认事项

### 10.1 技术确认项

| 问题 | 状态 | 备注 |
|-----|------|------|
| GE dynamicDims 支持范围 | ✅ 已确认 | 支持 7 个 batch sizes |
| Prefill 是否需要 graph | ✅ 已确认 | 不需要（参考 MLU） |
| KV Cache 传递方式 | ✅ 已确认 | 每次 RunGraph 传递 |
| 算子差异处理 | ✅ 已确认 | 通过 params 传递配置 |
| MIXED ForwardType 处理 | ✅ 已确认 | Eager mode |

### 10.2 实施前需确认

- ❓ GE ops 库是否支持所有 DeepSeek V4 算子？
- ❓ CompileGraph 实际耗时是否约 5min？
- ❓ RunGraphWithStreamAsync API 是否可用？

---

## 附录

### A. 参考资料

- MLU Graph Executor 实现：`xllm/core/runtime/mlu_graph_executor_impl.cpp`
- DS2 ATB Layer 实现：`xllm/core/layers/npu/npu_deepseek_v2_decoder_layer_impl.cpp`
- GE Transformer Demo：`/home/lianghao/thj/code/ge/examples/es/transformer/`
- BatchForwardType 定义：`xllm/core/framework/batch/batch_forward_type.h`

### B. 关键代码位置

- **MLU eager/graph 选择**：`mlu_graph_executor_impl.cpp:235`
- **MLU Prefill mask 生成**：`mlu_graph_executor_impl.cpp:718-790`
- **DS2 ATB init_layer**：`npu_deepseek_v2_decoder_layer_impl.cpp:687-743`
- **BatchForwardType 定义**：`batch_forward_type.h:23-36`

---

**文档结束**