# DeepSeek V4 NPU GE 实现初稿总结

> 生成日期：2026-05-21
> 状态：代码初稿（框架搭建完成，具体实现为 TODO）

---

## 一、已创建的文件列表

### 1. Runtime 层（xllm/core/runtime/）

| 文件 | 状态 | 说明 |
|------|------|------|
| `ge_graph_executor_impl.h` | ✅ 已创建 | GE Executor 头文件，包含注册 `REGISTER_EXECUTOR("npu_ge", GeGraphExecutorImpl)` |
| `ge_graph_executor_impl.cpp` | ✅ 已创建 | GE Executor 实现，核心逻辑为 TODO |
| `ge_tensor_converter.h` | ✅ 已创建 | Tensor 转换工具头文件 |
| `ge_tensor_converter.cpp` | ✅ 已创建 | Tensor 转换工具实现，零拷贝逻辑为 TODO |
| `ge_session_manager.h` | ✅ 已创建 | GE Session 管理器头文件 |
| `ge_session_manager.cpp` | ✅ 已创建 | GE Session 管理器实现 |

### 2. Model 层（xllm/models/llm/npu_ge/）

| 文件 | 状态 | 说明 |
|------|------|------|
| `deepseek_v4.h` | ✅ 已创建 | DS4 GE Model 头文件，包含注册 `REGISTER_CAUSAL_MODEL(deepseek_v4_npu_ge, ...)` |
| `deepseek_v4.cpp` | ✅ 已创建 | DS4 GE Model 实现，forward 和 graph 构建为 TODO |
| `CMakeLists.txt` | ✅ 已创建 | Model 编译配置（注释状态） |

### 3. Layer 层（xllm/core/layers/npu_ge/）

| 文件 | 状态 | 说明 |
|------|------|------|
| `CMakeLists.txt` | ✅ 已创建 | Layer 编译配置（注释状态） |

---

## 二、已完成的注册

### 1. Executor Backend 注册

```cpp
// xllm/core/runtime/ge_graph_executor_impl.h:68
REGISTER_EXECUTOR("npu_ge", GeGraphExecutorImpl);
```

**说明**：
- Backend 名称：`npu_ge`（避免与现有 `npu` ATB backend 冲突）
- 用户可通过 `--backend=npu_ge` 选择 GE 模式

### 2. Model 注册

```cpp
// xllm/models/llm/npu_ge/deepseek_v4.h:76
REGISTER_CAUSAL_MODEL(deepseek_v4_npu_ge, DeepseekV4GeForCausalLM);
```

**说明**：
- Model 名称：`deepseek_v4_npu_ge`
- 用户可通过 `--model=deepseek_v4_npu_ge` 使用 DS4 GE 版本

---

## 三、需要修改的现有文件

### 1. Runtime CMakeLists.txt（待修改）

**文件**：`xllm/core/runtime/CMakeLists.txt`

**需要添加**：
```cmake
# TODO: Add GE executor sources when ready
# if(USE_GE)
#   list(APPEND RUNTIME_SOURCES
#     ge_graph_executor_impl.cpp
#     ge_tensor_converter.cpp
#     ge_session_manager.cpp
#   )
# endif()
```

### 2. Layers CMakeLists.txt（待修改）

**文件**：`xllm/core/layers/CMakeLists.txt`

**需要添加**：
```cmake
# TODO: Add GE layers when ready
# if(USE_GE)
#   add_subdirectory(npu_ge)
# endif()
```

### 3. Models CMakeLists.txt（待修改）

**文件**：`xllm/models/llm/CMakeLists.txt`（需要检查是否存在）

**需要添加**：
```cmake
# TODO: Add GE models when ready
# if(USE_GE)
#   add_subdirectory(npu_ge)
# endif()
```

### 4. 主 CMakeLists.txt（待修改）

**文件**：`xllm/CMakeLists.txt`

**需要添加**：
```cmake
# TODO: Add GE configuration option
# option(USE_GE "Enable GE graph mode for NPU" OFF)
# 
# if(USE_GE)
#   find_package(GE REQUIRED)
#   find_package(ACL REQUIRED)
#   
#   include_directories($ENV{ASCEND_HOME_PATH}/include/ge)
#   
#   # Add GE libraries
#   set(GE_LIBS
#     ge_runner
#     ge_compiler
#     graph
#     ascendcl
#   )
# endif()
```

---

## 四、核心代码逻辑（TODO 状态）

### 1. Executor::run() 核心逻辑

**文件**：`ge_graph_executor_impl.cpp:54-76`

```cpp
ModelOutput GeGraphExecutorImpl::run(...) {
  if (!params.batch_forward_type.is_decode()) {
    // PREFILL / CHUNKED_PREFILL / MIXED → eager mode
    return model_->forward(tokens, positions, kv_caches, params);
  }
  
  // DECODE → use decode graph
  return run_decode_graph(tokens, positions, kv_caches, params);
}
```

**TODO 项**：
- ✅ ForwardType 判断逻辑（框架已完成）
- ❌ run_decode_graph() 实现（需要 GE API）
- ❌ convert_inputs() 实现（需要 GE Tensor API）
- ❌ convert_outputs() 实现（需要 GE Tensor API）

### 2. Model::init_model() 核心逻辑

**文件**：`deepseek_v4.cpp:64-76`

```cpp
void DeepseekV4GeModelImpl::init_model() {
  // TODO: Initialize GE Session
  // auto& session_manager = ge::GeSessionManager::get_instance();
  // session_ = session_manager.get_or_create_session();

  // TODO: Build model graph (ES 构建，只一次)
  // 权重作为 Graph 输入（约 565 个，每次执行传入）
  // build_model_graph();

  // TODO: Compile two graphs (同一个 graph，不同的 dynamicDims 配置)
  // compile_prefill_graph();  // Prefill Graph (-1 dynamic shape)
  // compile_decode_graph();   // Decode Graph (dynamicDims)

  // 注意：权重作为输入，成员变量每次执行需要（不可释放）
  LOG(INFO) << "DeepSeek V4 GE model initialized (双图方案 pending)";
}
```

**TODO 项**：
- ❌ GE Session 初始化（需要 GE API）
- ❌ build_model_graph() 实现（需要 GE ES API）
- ❌ 权重输入 placeholder 创建（约 565 个）

**重要说明**：Decode Graph 包含完整推理流程（参考 DS4 Python 实现）：
- ✅ Embedding lookup（tokens → hidden_states）
- ✅ HyperConnection expansion
- ✅ 43 DecoderLayers
- ✅ HyperConnection Head
- ✅ Final Norm
- ✅ LM Head（hidden_states → logits）

**Graph 输入**：`input_ids`（原始 tokens），而非 `hidden_states`

**权重管理架构**（修正）：
- ✅ 权重作为 Graph 输入（每次执行传入约 565 个权重 tensor）
- ✅ frozen_parameter=true：权重地址不变（运行时优化）
- ✅ 成员变量每次执行需要（不可释放）
- ❌ **不支持 Rolling Load**（frozen_parameter 要求地址不变）

### 3. build_decode_graph() 完整流程（参考 DS4 Python）

**文件**：`deepseek_v4.cpp:76-117`

**完整推理流程**（参考 modeling_deepseek.py:1467-1552, 1976-1999）：

```cpp
void build_decode_graph() {
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_decode");
  
  // 输入：input_ids（原始 tokens）
  auto input_ids = builder->CreateInput(0, "input_ids", ge::DT_INT32, {-1});
  
  // Step 1: Embedding lookup
  auto hidden = Embedding(input_ids, embed_tokens_weight_);
  
  // Step 2: HyperConnection expansion
  hidden = Reshape(hidden, {-1, hc_mult_, hidden_size_});
  
  // Step 3: 43 DecoderLayers（直接在 builder 上构建）
  for (int layer = 0; layer < 43; layer++) {
    hidden = build_decoder_layer_ops(builder, hidden, ...);
  }
  
  // Step 4: HyperConnection Head
  hidden = HyperConnectionHead(hidden, hc_head_weights_);
  
  // Step 5: Final Norm
  hidden = RMSNorm(hidden, final_norm_weight_);
  
  // Step 6: LM Head
  auto logits = MatMul(hidden, lm_head_weight_);
  
  // Build graph（一次性构建完整推理流程）
  auto graph = builder->BuildAndReset({logits});
  session_->AddGraph(graph_id, *graph);
  session_->CompileGraph(graph_id);
}
```

**关键特征**：
- ✅ **一个 Graph 包含完整推理流程**（Embedding → Layers → HC Head → Norm → LM Head）
- ✅ **Graph 输入是 input_ids**（tokens），输出是 logits
- ✅ **不分层构建**：直接在一个 EsGraphBuilder 上构建所有算子
- ✅ **参考 DS4 Python torch.compile 实现**

**TODO 项**：
- ❌ Embedding 算子调用（需要 GE Embedding API）
- ❌ HyperConnection expansion 算子调用
- ❌ 43 DecoderLayers 算子调用
- ❌ HyperConnection Head 算子调用
- ❌ LM Head 算子调用

---

### 4. Layer 构建方式（澄清）

**正确理解**：
- ⚠️ **不需要独立的 Layer Builder 类**：所有算子在同一 EsGraphBuilder 上构建
- ⚠️ **直接在 build_model_graph() 中构建**：循环构建 43 个 DecoderLayer
- ⚠️ **可选辅助函数**：可使用 static function 组织算子调用逻辑

**代码组织**：
- 所有算子构建逻辑在 `deepseek_v4.cpp` 中
- 可选的辅助函数用于组织代码（如 `build_decoder_layer_ops()`）

---

## 五、目录结构（已创建）

```
xllm/
├── core/
│   ├── runtime/
│   │   ├── ge_graph_executor_impl.h         ✅
│   │   ├── ge_graph_executor_impl.cpp       ✅
│   │   ├── ge_tensor_converter.h            ✅
│   │   ├── ge_tensor_converter.cpp          ✅
│   │   ├── ge_session_manager.h             ✅
│   │   └ ge_session_manager.cpp             ✅
│   │   └ CMakeLists.txt                     📝（待添加 GE sources）
│   │
│   └── layers/
│       └── npu_ge/
│           └ CMakeLists.txt                         ✅（注释状态）
│
└── models/
    └── llm/
        └── npu_ge/
            ├── deepseek_v4.h                 ✅
            ├── deepseek_v4.cpp               ✅
            └ CMakeLists.txt                  ✅（注释状态）
```

---

## 六、下一步工作

### Phase 1: 基础设施搭建（优先级：P0）

1. **配置 GE 依赖**：
   - 修改主 CMakeLists.txt，添加 `USE_GE` option
   - 配置 GE 库依赖（ge_runner, ge_compiler, ascendcl）
   - 配置 GE 头文件路径

2. **实现 GeTensorConverter**：
   - 实现 dtype 映射表
   - 实现零拷贝 torch → GE tensor 转换
   - 添加单元测试

3. **实现 GeSessionManager**：
   - 实现 GE Session 创建和管理
   - 实现 Graph ID 分配

### Phase 2: Model Layer 实现（优先级：P1）

4. **实现 build_decode_graph()**：
   - 使用 GE ES API 构建 Decode Graph
   - 配置 dynamicDims：{1,4,8,16,32,64,128}
   - 实现 AddGraph + CompileGraph

5. **实现 DecoderLayer Builder**：
   - 实现 Attention graph 构建
   - 实现 MoE graph 构建
   - 实现 HyperConnection graph 构建

### Phase 3: Runtime Layer 实现（优先级：P1）

6. **实现 run_decode_graph()**：
   - 实现 RunGraphWithStreamAsync 调用
   - 实现 aclrtSynchronizeStream 同步
   - 实现输出转换

### Phase 4: 测试与验证（优先级：P2）

7. **单元测试**：
   - GeTensorConverter 测试
   - GeSessionManager 测试
   - GeGraphExecutor 测试

8. **集成测试**：
   - DS4 GE model 加载测试
   - Decode graph 执行测试
   - 输出正确性验证

---

## 七、风险提示

### 1. GE API 可用性风险

**风险**：GE ES API、Session API 可能不完全可用

**应对**：
- 先测试 GE Demo 中的 Transformer example
- 确认所有需要的 API 是否可用
- 如果不可用，考虑使用其他 GE API（非 ES）

### 2. 编译时间风险

**风险**：CompileGraph 可能超过 5min

**应对**：
- 先测试单层 CompileGraph 时间
- 评估是否需要分层编译
- 考虑异步编译策略

### 3. 算子支持性风险

**风险**：GE ops 库可能不支持 DeepSeek V4 的某些算子

**应对**：
- 列出所有需要的算子
- 与 GE 团队确认支持性
- 如果不支持，需要先实现算子

---

## 八、文件路径索引

### Runtime 层
- Executor 注册：`ge_graph_executor_impl.h:68`
- Executor 实现：`ge_graph_executor_impl.cpp`
- Tensor 转换：`ge_tensor_converter.cpp`
- Session 管理：`ge_session_manager.cpp`

### Model 层
- Model 注册：`deepseek_v4.h:76`
- Model 实现：`deepseek_v4.cpp`
- Graph 构建：`deepseek_v4.cpp:76-119`

### Layer 层
- Layer 构建逻辑：`deepseek_v4.cpp:build_model_graph()`（TODO）

### CMake 配置
- Runtime：`core/runtime/CMakeLists.txt`（待修改）
- Layers：`core/layers/CMakeLists.txt`（待修改）
- Models：`models/llm/CMakeLists.txt`（待修改）
- 主配置：`xllm/CMakeLists.txt`（待修改）

---

**总结文档结束**

---

## 附录：参考文档

### DS4 Python 实现（torch.compile）

**文件路径**：
`/home/lianghao/thj/code/cann-recipes-infer/models/deepseek-v4/models/modeling_deepseek.py`

**关键代码位置**：
- Graph 编译函数：`get_cached_graph()`（line 1736-1760）
- Decode 主函数：`main_decode()`（line 2011-2019）
- Model forward：`DeepseekV3Model.forward()`（line 1467-1552）
- LM Head：`forward_lm_head()`（line 1828）
- 执行路径：`decode()`（line 2031-2039）

**关键洞察**：
- 一个函数（`main_decode`）编译整个推理流程
- Graph 包含：Embedding + 43 Layers + HC Head + Norm + LM Head
- Graph 输入：`input_ids`（原始 tokens）
- torch.compile 使用 `fullgraph=True` 参数

### GE Demo（ES API）

**文件路径**：
`/home/lianghao/thj/code/ge/examples/es/transformer/cpp/src/make_transformer_graph.cpp`

**关键代码**：
- Graph 构建：`MakeTransformerGraphByEs()`（line 50-86）
- 直接调用算子：`Reshape()`, `MatMul()`, `Sigmoid()` 等
- BuildAndReset：一次性构建整个图