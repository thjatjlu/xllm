# DeepSeek V4 GE 方案 - 实施计划（初稿）

> 文档版本：v1.0
> 创建日期：2025-01-19

---

## 一、实施阶段划分

### Phase 1: 基础设施搭建（1-2周）

**目标**：搭建 GE runtime 基础框架

**任务清单**：
1. 创建 GeTensorConverter（torch ↔ ge tensor 转换）
2. 创建 GeSessionManager（Session 单例管理）
3. 创建 GeGraphExecutorImpl 框架（Executor 基础结构）
4. 配置 CMake 依赖（GE 库链接）

**新增文件**：
```
xllm/core/runtime/
  ├── ge_tensor_converter.h/.cpp
  ├── ge_session_manager.h/.cpp
  ├── ge_graph_executor_impl.h/.cpp
  └── CMakeLists.txt
```

**验收标准**：
- ✅ GE Session 成功初始化
- ✅ Tensor 转换功能测试通过
- ✅ CMake 编译通过

---

### Phase 2: Graph 构建框架（2-3周）

**目标**：实现 Decode Graph 构建流程

**任务清单**：
1. 实现 EsGraphBuilder 使用示例
2. 实现 `collect_all_weights()`（收集所有权重）
3. 实现 `build_model_graph()`（创建权重输入 placeholder）
4. 测试 CompileGraph 流程

**新增文件**：
```
xllm/models/llm/npu_ge/
  ├── deepseek_v4.h/.cpp
  └── CMakeLists.txt

xllm/core/runtime/
  ├── ge_graph_executor_impl.h/.cpp
  ├── ge_tensor_converter.h/.cpp
  └ ge_session_manager.h/.cpp
```
xllm/core/layers/npu_ge/
  └── CMakeLists.txt

xllm/models/llm/npu_ge/
  ├── deepseek_v4.h/.cpp
  └── CMakeLists.txt
```

**验收标准**：
- ✅ Decode Graph 成功构建
- ✅ CompileGraph 成功（约 5min）
- ✅ Graph structure 可 dump 查看

---

### Phase 3: 算子适配（3-4周）

**目标**：适配 DeepSeek V4 关键算子

**算子清单**：

| 算子名称 | GE 对应算子 | 优先级 | 状态 |
|---------|-----------|-------|------|
| **RMSNorm** | `RMSNorm` | P0 | 待实现 |
| **HyperConnectionPre** | `Sinkhorn + Split` | P0 | 待实现 |
| **MLA Attention** | `SparsePagedAttention` | P0 | 待实现 |
| **HyperConnectionPost** | `Add + Mul` | P0 | 待实现 |
| **FusedMoE** | `MoEGate + GroupGEMM` | P0 | 待实现 |
| **Indexer** | `TopK + Gather` | P1 | 待实现 |
| **Compressor** | `Softmax + WeightedSum` | P1 | 待实现 |
| **RoPE (DeepSeek Yarn)** | `ApplyRotaryEmbedding` | P0 | 待实现 |

**验收标准**：
- ✅ 单算子功能测试通过
- ✅ 算子集成到 graph 中
- ✅ 算子性能测试通过

---

### Phase 4: 集成测试（1-2周）

**目标**：完整模型推理测试

**测试场景**：

| 测试项 | 测试内容 | 预期结果 |
|-------|---------|---------|
| **单元测试** | 单算子 forward | 输出正确 |
| **集成测试** | 单层 DecoderLayer forward | 输出正确 |
| **系统测试** | 完整 model forward（prefill + decode） | 输出正确 |
| **性能测试** | 不同 batch sizes 性能对比 | 性能达标 |
| **压力测试** | 长时间运行稳定性 | 无错误 |

**验收标准**：
- ✅ Prefill eager mode 正常工作
- ✅ Decode graph 正常执行
- ✅ 输出与 MLU/DS2 一致（精度误差 < 1%）
- ✅ 性能满足要求（吞吐量、延迟）

---

### Phase 5: 优化与上线（1周）

**目标**：性能优化和生产环境准备

**优化项**：
1. Tensor 转换优化（零拷贝验证）
2. Graph memory planning 优化
3. 并行执行优化（stream 管理）
4. 错误处理完善

**上线准备**：
1. 配置文件完善
2. 日志和监控集成
3. 文档完善
4. 代码审查

---

## 二、CMake 配置

### 2.1 主 CMakeLists.txt

```cmake
# xllm/CMakeLists.txt（新增部分）

if(USE_NPU_GE)
  message(STATUS "Building with NPU GE backend")
  
  # GE 核心库
  set(GE_LIBS
    ge_runner       # Session + RunGraph
    ge_compiler     # CompileGraph  
    graph           # Graph structure
    graph_base      # Base utilities
    es_all          # ES API (generated)
    ascendcl        # ACL runtime
    c_sec           # Security lib
  )
  
  # GE 头文件路径
  include_directories($ENV{ASCEND_HOME_PATH}/include)
  include_directories($ENV{ASCEND_HOME_PATH}/include/ge)
  include_directories(${CMAKE_BINARY_DIR}/output/include)  # ES generated
  
  # ES API 生成（一次性）
  find_package(GenerateEsPackage REQUIRED)
  add_es_library(
    ES_LINKABLE_AND_ALL_TARGET es_all
    OUTPUT_PATH ${CMAKE_BINARY_DIR}/output
  )
  
  # 链接库
  list(APPEND COMMON_LIBS ${GE_LIBS})
  
  # 子目录
  add_subdirectory(core/layers/npu_ge)
endif()
```

### 2.2 Runtime CMakeLists.txt

```cmake
# xllm/core/runtime/CMakeLists.txt（新增部分）

cc_library(
  NAME ge_runtime
  HDRS
    ge_tensor_converter.h
    ge_session_manager.h
    ge_graph_executor_impl.h
  SRCS
    ge_tensor_converter.cpp
    ge_session_manager.cpp
    ge_graph_executor_impl.cpp
  DEPS
    ge_runner
    graph
    ascendcl
    framework_model
    torch_npu
)
```

---

## 三、测试计划

### 3.1 单元测试

```cpp
// xllm/core/runtime/ge_tensor_converter_test.cpp

TEST(GeTensorConverter, TorchToGe) {
  // Test 1: BF16 tensor conversion
  torch::Tensor torch_bf16 = torch::randn({128, 4096}, torch::kBFloat16)
                                 .to(torch::kNPU);
  
  ge::Tensor ge_tensor = GeTensorConverter::torch_to_ge(torch_bf16);
  
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetDataType(), ge::DT_BF16);
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetShape().GetDims(), 
            std::vector<int64_t>({128, 4096}));
  
  // Test 2: Zero-copy verification
  EXPECT_EQ(ge_tensor.GetData(), torch_bf16.data_ptr());
}

TEST(GeTensorConverter, GeToTorch) {
  // 反向转换测试
  ge::Tensor ge_input = ...;
  torch::Tensor torch_output = GeTensorConverter::ge_to_torch(ge_input);
  
  // 验证 dtype, shape, data pointer
}
```

### 3.2 集成测试

```cpp
// xllm/models/llm/npu_ge/deepseek_v4_test.cpp

TEST(DeepseekV4GeModel, CompileDecodeGraph) {
  ModelContext context = create_test_context();
  auto model = std::make_unique<DeepseekV4GeModelImpl>(context);
  
  // Load weights
  StateDict state_dict = load_test_weights();
  model->load_state_dict(state_dict);
  
  // Init model (compile graph)
  model->init_model();
  
  // Verify graph compiled
  EXPECT_TRUE(model->has_decode_graph());
}

TEST(DeepseekV4GeModel, DecodeForward) {
  // Test decode phase
  torch::Tensor tokens = torch::tensor({1, 2, 3}, torch::kInt32).to(torch::kNPU);
  torch::Tensor positions = torch::tensor({100, 101, 102}, torch::kInt32).to(torch::kNPU);
  
  ModelInputParams params;
  params.batch_forward_type = BatchForwardType::DECODE;
  
  KVCache kv_cache = create_test_kv_cache();
  
  auto output = model->forward(tokens, positions, {kv_cache}, params);
  
  // Verify output shape and correctness
  EXPECT_EQ(output.hidden_states.size(0), 3);
  EXPECT_EQ(output.hidden_states.size(1), 4096);
}
```

### 3.3 性能测试

```bash
# scripts/test_ge_performance.sh

#!/bin/bash

# Test different batch sizes
for batch in 1 4 8 16 32 64 128; do
  echo "Testing batch=$batch"
  
  # Measure decode latency
  ./xllm --backend=npu_ge \
         --model=/path/to/deepseek_v4 \
         --test-batch=$batch \
         --test-mode=decode
  
  # Expected: latency < 50ms for batch=1
  # Expected: latency < 100ms for batch=128
done

# Compare with MLU baseline
./scripts/compare_performance.sh --ge --mlu
```

---

## 四、风险管理

### 4.1 技术风险

| 风险 | 概率 | 影响 | 应对措施 |
|-----|-----|------|---------|
| **GE dynamicDims 不支持** | 低 | 高 | Fallback 到固定 buckets |
| **算子未实现** | 中 | 高 | 先确认 GE ops 库支持性 |
| **CompileGraph 失败** | 中 | 高 | 详细错误日志 + fallback |
| **性能不如预期** | 中 | 中 | 性能对比测试 + 优化 |
| **内存泄漏** | 低 | 高 | Memory profiling |

### 4.2 时间风险

| 风险 | 概率 | 影响 | 应对措施 |
|-----|-----|------|---------|
| **算子适配时间长** | 高 | 中 | 优先实现 P0 算子 |
| **测试发现问题多** | 中 | 中 | 预留 buffer time |
| **编译环境问题** | 低 | 中 | 提前验证编译环境 |

---

## 五、上线清单

### 5.1 功能验收

- ✅ Prefill eager mode 正常工作
- ✅ Decode graph 正常执行
- ✅ 所有 ForwardType 支持正确
- ✅ KV Cache 正常读写
- ✅ 输出精度正确（误差 < 1%）

### 5.2 性能验收

- ✅ CompileGraph 时间 < 10min
- ✅ Decode latency < 100ms（batch=128）
- ✅ Throughput > 1000 tokens/s
- ✅ Memory usage < 80% NPU memory

### 5.3 稳定性验收

- ✅ 连续运行 24h 无错误
- ✅ 异常情况正确处理
- ✅ 日志和监控正常

---

**文档结束**