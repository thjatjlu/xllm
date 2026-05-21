# DeepSeek V4 NPU GE 图模式设计文档（索引）

> 文档集合版本：v1.0
> 创建日期：2025-01-19
> 状态：初稿（待评审）

---

## 文档目录

### 01. 方案概述

**文件**：`01_ge_ds4_design_overview.md`

**内容概要**：
- 核心设计决策（只编译 Decode Graph）
- 架构设计（启动流程 + 运行流程）
- 关键配置（dynamicDims 配置）
- ForwardType 处理策略
- KV Cache 传递方式
- 文件结构规划
- 编译依赖配置
- 性能预估
- 风险与应对

---

### 02. 代码框架

**文件**：`02_ge_ds4_code_framework.md`

**内容概要**：
- 核心类设计（GeGraphExecutorImpl、GeTensorConverter）
- Executor::run() 实现逻辑
- Tensor 转换实现（零拷贝）
- Model 层实现（init_model、build_decode_graph）
- Graph 构建器（DecoderLayer Builder、Attention Builder）

---

### 03. 实施计划

**文件**：`03_ge_ds4_implementation_plan.md`

**内容概要**：
- 实施阶段划分（5 个 Phase）
- CMake 配置（主 CMake + Runtime CMake）
- 测试计划（单元测试 + 集成测试 + 性能测试）
- 风险管理（技术风险 + 时间风险）
- 上线清单（功能验收 + 性能验收 + 稳定性验收）

---

### 04. 关键决策记录

**文件**：`04_ge_ds4_key_decisions.md`

**内容概要**：
- 关键设计决策（Q1-Q9）
- 技术澄清（BatchForwardType、Chunked Prefill、dynamicDims vs `-1`）
- 待确认事项（GE ops 支持性、CompileGraph 耗时、API 可用性）
- 关键代码位置索引

---

### 05. Prefill vs Decode 算子差异分析（补充）

**文件**：`05_prefill_vs_decode_operator_diff.md`

**内容概要**：
- **关键发现**：算子执行流程完全相同！
- DecoderLayer forward 无 Prefill/Decode 分支
- Attention 内部差异详解（通过 `is_prefill` 参数控制）
- write_kv_to_cache 差异（批量 vs 单个）
- KV Merge 差异（Prefill 独有）
- 对 GE 方案的影响（不需要两个 graph）
- 最终设计决策确认（只编译 Decode graph）

---

### 06. MIXED 模式风险分析（双图方案）

**文件**：`06_mixed_mode_risk_analysis.md`

**内容概要**：
- MIXED 模式定义与触发条件
- **方案 A**：MIXED 使用 Eager Fallback（推荐方案）
  - 性能损失风险（中风险）
  - 输出一致性风险（高风险）
  - 内存管理风险（中风险）
  - 稳定性风险（中风险）
  - 频率不可控风险（中风险）
- **方案 B**：MIXED 使用 Prefill Graph 执行（不推荐方案）
  - `is_prefill` 参数冲突（高风险）
  - Attention 计算错误（高风险）
  - KV Cache 写入错误（高风险）
  - 推理结果完全错误（极高风险）
- 缓解策略与行动计划（P0/P1 任务）
- 监控指标与报警规则
- 最终方案推荐（方案 A）

---

### 07. GE ES 架构理解修正说明（新增）

**文件**：`07_ge_es_architecture_correction.md`

**内容概要**：
- **核心修正点**：Graph 包含完整推理流程（而非仅 43 Layers）
- **修正 1**：Graph 输入是 input_ids（原始 tokens）
- **修正 2**：一个 EsGraphBuilder 构建整个 Model（而非分层）
- **参考文档**：DS4 Python 实现、GE Demo 实际用法
- **更新的文档列表**：总结文档、代码框架、代码初稿
- **架构对比**：ATB vs GE ES 的关键差异

---

### 08. GE 双图方案设计说明（新增）

**文件**：`08_ge_dual_graph_design.md`

**内容概要**：
- **核心设计**：ES 构图只一次，编译两次（Prefill Graph + Decode Graph）
- **Prefill Graph**：-1 完全动态 shape（灵活性）
- **Decode Graph**：dynamicDims 分档优化（性能优化）
- **编译时间**：约 10min（两次编译，可控）
- **MIXED 处理**：eager mode fallback（正确性保证）
- **与 DS2 ATB 对比**：编译次数从 43+ 次降到 2 次
- **实现细节**：build_model_graph、compile_prefill_graph、compile_decode_graph

---

### 09. GE Graph 权重作为输入设计（新增）

**文件**：`10_ge_weight_as_input_design.md`

**内容概要**：
- **核心决策**：权重作为 Graph 输入，每次执行传入
- **与 DS4 Python 对比**：nn.Module 自动引用 vs 显式传入
- **frozen_parameter=true**：权重地址不变（运行时优化）
- **权重输入数量**：约 565 个（方案 A：不合并）
- **权重输入顺序**：Graph 构建顺序 = 执行传入顺序（保序）
- **权重参数提取**：从 weight tensor 提取 shape/dtype
- **成员变量必要性**：每次执行需要（不可释放）
- **Rolling Load**：不支持（frozen_parameter 要求地址不变）
- **内存需求**：约 44 GB（所有权重常驻 HBM）
- **方案 A vs 方案 B**：不合并（当前） vs 合并（未来优化）

---

## 核心设计要点总结

### 1. 双图方案架构（修正后）

```
✅ ES 构图：一次（构建完整推理流程）
✅ Graph 编译：两次（Prefill + Decode）
✅ Prefill Graph：-1 完全动态（灵活性）
✅ Decode Graph：dynamicDims 分档（性能优化）
✅ 总编译时间：约 10min（可控）
✅ MIXED 模式：报错处理（不支持）
✅ 权重管理：作为 Graph 输入（每次执行传入）
✅ 权重数量：约 565 个输入（方案 A：不合并）
✅ frozen_parameter=true：权重地址不变
✅ 成员变量：每次执行需要（不可释放）
✅ Rolling Load：不支持（地址不变要求）
```

### 2. 完整推理流程（6 步）

| 步骤 | 算子 | 输入 → 输出 |
|------|------|------------|
| **Step 1** | Embedding lookup | input_ids → hidden_states |
| **Step 2** | HyperConnection expansion | [batch, seq] → [batch, seq, hc_mult, hidden] |
| **Step 3** | 43 DecoderLayers | 循环构建所有层 |
| **Step 4** | HyperConnection Head | 合并 hc_mult 维度 |
| **Step 5** | Final Norm | hidden_states → normalized |
| **Step 6** | LM Head | hidden → logits |

### 3. ForwardType 处理

| ForwardType | 处理方式 | Graph 使用 |
|------------|---------|-----------|
| PREFILL | Eager mode | ❌ |
| CHUNKED_PREFILL | Eager mode | ❌ |
| DECODE | GE Graph | ✅ |
| MIXED | Eager mode | ❌ |
| EMPTY | 直接返回 | ❌ |

### 3. dynamicDims 配置

```
Decode Graph: {1,4,8,16,32,64,128}
- 7 个高频 batch sizes
- 参考 MLU/DS2 成功实践
- 平均 padding < 2%
```

### 4. Layer Builder 定位（澄清）

**重要说明**：
- ⚠️ **不是架构必需**：只是代码组织方式（辅助函数）
- ⚠️ **所有算子在同一 builder 上构建**：不分层创建独立 builder
- ⚠️ **可选设计**：也可以直接在 build_decode_graph() 中调用算子

**推荐方案**：
- 简化为辅助函数（static function）
- 用于组织算子调用逻辑
- 不需要独立的 Builder 类

---

## 关键文件路径

### Runtime 层

```
docs/ge_design/
├── README.md                                 # 索引文档（5.3KB）
├── 01_ge_ds4_design_overview.md              # 方案概述（9.9KB）
├── 02_ge_ds4_code_framework.md               # 代码框架（11KB）
├── 03_ge_ds4_implementation_plan.md          # 实施计划（8.3KB）
├── 04_ge_ds4_key_decisions.md                # 关键决策记录（9KB）
├── 05_prefill_vs_decode_operator_diff.md     # Prefill vs Decode 算子差异（9.5KB）
└── 06_mixed_mode_risk_analysis.md            # MIXED 模式风险分析（新增）
```

### Model 层

```
xllm/models/llm/npu_ge/
  ├── deepseek_v4.h/.cpp                  # Model 定义
```

### Layer 层

```
xllm/core/layers/npu_ge/
  └── CMakeLists.txt                      # 编译配置（预留）
```

---

## 参考资料

### MLU 实现

- `xllm/core/runtime/mlu_graph_executor_impl.cpp`（Executor）
- `xllm/models/llm/deepseek_v4.h`（Model 定义）
- `xllm/core/layers/mlu/deepseek_v4/*`（算子实现）

### DS2 ATB 实现

- `xllm/core/runtime/acl_graph_executor_impl.cpp`（Executor）
- `xllm/models/llm/npu/deepseek_v2.h`（Model 定义）
- `xllm/core/layers/npu/npu_deepseek_v2_decoder_layer_impl.cpp`（Layer）

### GE Demo

- `/home/lianghao/thj/code/ge/examples/es/transformer/cpp/`（Transformer demo）
- `/home/lianghao/thj/code/ge/inc/external/ge/ge_api.h`（Session API）

### xLLM Framework

- `xllm/core/framework/batch/batch_forward_type.h`（ForwardType 定义）
- `xllm/core/runtime/executor_impl_factory.h`（Executor 注册）

---

## 后续工作

### 立即开始

1. **Phase 1: 基础设施搭建**（1-2周）
   - 实现 GeTensorConverter
   - 实现 GeSessionManager
   - 创建 GeGraphExecutorImpl 框架

### 待确认事项

1. **GE ops 库支持性**（关键）
   - 确认所有 DeepSeek V4 算子是否已实现
   - 如果未实现，需要先实现算子

2. **CompileGraph 实际耗时**
   - 测试单次 CompileGraph 时间
   - 验证是否约 5min

3. **RunGraphWithStreamAsync API**
   - 验证 API 是否支持 dynamicDims
   - 验证异步执行是否可用

---

## 文档更新计划

- **v1.0**（当前）：初稿，基于讨论和分析
- **v2.0**（待更新）：补充算子适配细节
- **v3.0**（待更新）：补充测试结果和性能数据
- **v4.0**（待更新）：补充优化细节和上线经验

---

**文档索引结束**