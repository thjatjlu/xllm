# 文档和代码更新总结

> 更新日期：2026-05-21
> 架构修正：权重作为 Graph 输入（而非固化）

---

## 一、已完成的修改

### 1. 文档修改

| 操作 | 文件 | 说明 |
|------|------|------|
| **删除** | `09_ge_weight_management_design.md` | 废弃"权重固化"设计 |
| **新建** | `10_ge_weight_as_input_design.md` | "权重作为输入"完整设计 |
| **更新** | `README.md` | 更新架构设计要点和文档索引 |

---

### 2. 代码修改

| 文件 | 修改内容 |
|------|---------|
| **deepseek_v4.h** | ✅ 成员变量注释：每次执行需要，不可释放<br>✅ free_weights/reload_weights 报错：添加原因说明<br>✅ 新增：collect_all_weights() 方法声明 |
| **deepseek_v4.cpp** | ✅ load_state_dict 注释：每次执行需要<br>✅ build_model_graph 注释：权重作为输入（约 565 个）<br>✅ compile_prefill_graph 注释：frozen_parameter=true<br>✅ compile_decode_graph 注释：frozen_parameter=true |
| **ge_graph_executor_impl.cpp** | ✅ run_prefill_graph 注释：传入所有权重（约 565 个）<br>✅ run_decode_graph 注释：传入所有权重（约 565 个） |

---

## 二、架构修正对比

| 维度 | 之前错误理解 | 修正后正确理解 |
|------|-------------|---------------|
| **权重处理方式** | 固化到 Graph | **作为 Graph 输入** |
| **权重数量** | 不讨论 | **约 565 个** |
| **执行方式** | 只传 input_ids | **传入 input_ids + 565 个权重** |
| **成员变量** | 编译后可释放 | **每次执行需要（不可释放）** |
| **frozen_parameter** | 编译优化 | **运行时优化（地址不变）** |
| **Rolling Load** | 不支持 | **不支持（地址不变要求）** |

---

## 三、关键设计要点

### 1. 权重输入顺序规范

```
关键约束：Graph 构建顺序 = 执行传入顺序（必须保序）

构建阶段：
builder->CreateInput(0, "input_ids", ...)
builder->CreateInput(1, "embed_weight", ...)
builder->CreateInput(2, "layer_0_weight_0", ...)
...

执行阶段：
inputs.push_back(tokens);              → 对应 input 0
inputs.push_back(embed_weight);         → 对应 input 1
inputs.push_back(layer_0_weight_0);     → 对应 input 2
...

顺序不一致会导致：数据类型错误、shape 不匹配、推理错误
```

---

### 2. frozen_parameter=true 的作用

```
作用时机：运行时（Graph 执行）
优化内容：权重地址不变 → 跳过地址检查 → 缩短下发时间
前提条件：
  ├─ 权重 tensor 地址必须不变
  ├─ 成员变量不能释放
  └─ 不支持 Rolling Load
```

---

### 3. 成员变量生命周期

```
Phase 1: 权重加载 → 成员变量接收权重
Phase 2: Graph 构建 → 提取 weight shape/dtype
Phase 3: Graph 编译 → frozen_parameter=true
Phase 4: 每次执行 → **传入成员变量权重 tensor**

结论：成员变量在每个阶段都必要，不可释放
```

---

## 四、未实现的 TODO

### 1. 需要新增的方法

```cpp
// DeepseekV4GeModelImpl
std::vector<torch::Tensor> collect_all_weights();  // 收集所有权重

// DeepseekV4GeDecoderLayer
std::vector<torch::Tensor> get_all_weights();      // 获取 layer 权重

// GeTensorConverter
ge::DataType map_dtype(torch::ScalarType dtype);   // dtype 映射
```

---

### 2. 需要实现的 GE ES API

```cpp
// Graph 构建
builder->CreateInput(index, name, dtype, shape)
builder->BuildAndReset(outputs)

// Graph 编译
session_->AddGraph(graph_id, graph, options)
session_->CompileGraph(graph_id)

// Graph 执行
session_->RunGraph(graph_id, inputs, outputs)
```

---

### 3. 需要验证的设计

- 权重输入数量：约 565 个（是否准确）
- Graph 输入顺序：保序要求（是否正确）
- frozen_parameter=true：地址不变优化（是否生效）

---

## 五、文档索引

**所有设计文档**：
- `01_ge_ds4_design_overview.md` - 设计概览
- `02_ge_ds4_code_framework.md` - 代码框架
- `03_ge_ds4_implementation_plan.md` - 实现计划
- `04_ge_ds4_key_decisions.md` - 关键决策
- `05_prefill_vs_decode_operator_diff.md` - 算子差异
- `06_mixed_mode_risk_analysis.md` - MIXED 风险分析
- `07_ge_es_architecture_correction.md` - 架构修正说明
- `08_ge_dual_graph_design.md` - 双图方案设计
- **`10_ge_weight_as_input_design.md`** - 权重作为输入设计（新增）
- `README.md` - 文档索引

---

## 六、下一步工作

### Phase 1: 实现辅助方法（P0）

1. 实现 `collect_all_weights()`
2. 实现 `get_all_weights()`（每个 layer）
3. 实现 `GeTensorConverter::map_dtype()`

---

### Phase 2: 实现 GE ES API（P1）

1. 实现 `build_model_graph()`
2. 实现 `compile_prefill_graph()`
3. 实现 `compile_decode_graph()`

---

### Phase 3: 实现 Graph 执行（P1）

1. 实现 `run_prefill_graph()`
2. 实现 `run_decode_graph()`
3. 实现权重 tensor 传入逻辑

---

### Phase 4: 测试验证（P2）

1. 验证权重输入顺序
2. 验证 frozen_parameter=true
3. 验证推理正确性

---

**更新总结结束**