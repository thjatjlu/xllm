# GE Graph 权重作为输入设计（方案 A）

> 更新日期：2026-05-21
> 核心设计：权重作为 Graph 输入，每次执行传入，不固化到 Graph 内部

---

## 一、核心设计决策

### 1. 权重处理方式

```
权重作为 Graph 输入：
├─ Graph 构建：权重作为输入 placeholder（约 565 个）
├─ Graph 编译：frozen_parameter=true（权重地址不变）
├─ Graph 执行：传入所有权重 tensor（每次执行）
├─ 权重存储：成员变量（device 地址）
└─ 成员变量：每次执行需要（不可释放）
```

---

### 2. 与 DS4 Python 的对比

| 维度 | DS4 Python (torch.compile) | DS4 GE (权重作为输入) |
|------|---------------------------|---------------------|
| **权重存储** | nn.Module 成员变量 | 成员变量（相同） |
| **权重引用** | forward 内部自动引用 self.xxx | **显式传入 Graph 输入** |
| **torch.compile** | 编译 forward 函数（自动捕获权重） | N/A |
| **GE Graph** | N/A | 权重作为显式输入 |
| **执行方式** | 权重已在 nn.Module（不传入） | **传入权重 tensor** |
| **frozen_parameter** | 权重地址不变 | 权重地址不变（相同） |

**关键差异**：
- DS4 Python：nn.Module 自动管理权重引用（forward 内部自动访问 self.xxx）
- DS4 GE：C++ 没有 nn.Module 自动引用机制 → 必须显式传入权重

---

### 3. frozen_parameter=true 的正确含义

**文档原文**：
> "推理场景下，权重类输入的内存地址通常保持不变，可以开启本功能缩短图下发时间，提升下发性能。"

**正确理解**：

```
frozen_parameter=true 的作用：
├─ 适用对象：权重类输入（weights as inputs）
├─ 作用时机：运行时（Graph 执行）
├─ 优化内容：
│   ├─ 权重 tensor 地址不变（每次执行使用相同地址）
│   ├─ Runtime 可以跳过地址检查
│   └─ 缩短 Graph 下发时间
│
├─ 前提条件：
│   ├─ 权重 tensor 地址必须不变（不能 free_weights）
│   ├─ 成员变量必须保持权重（不能释放）
│   └─ 不支持 Rolling Load（地址会变化）
│
└─ 性能提升：Graph 下发性能提升（不是编译性能）
```

---

## 二、权重输入详细列表

### 1. 权重类别和数量

| 权重类别 | 数量 | 来源（成员变量） |
|---------|-----|----------------|
| **Input IDs** | 1 | `tokens`（用户输入） |
| **Embedding** | 1 | `embed_tokens_->weight()` |
| **43 Layers** | ~559 | `layers_[i]->weights()`（每层约 13 个） |
| **Final Norm** | 1 | `norm_->weight()` |
| **LM Head** | 1 | `lm_head_->weight()` |
| **HyperConnection Head** | 3 | `hc_head_fn_, hc_head_base_, hc_head_scale_` |
| **总计** | **~565** | 约 565 个输入 |

---

### 2. 每个 Layer 的权重细分（约 13 个）

```
单层权重数量估算：
├─ HyperConnection pre-attention: ~3 个
│   ├─ hc_attn_pre.hc_fn
│   ├─ hc_attn_pre.hc_base
│   └─ hc_attn_pre.hc_scale
│
├─ Attention weights: ~4 个
│   ├─ q_a_proj.weight
│   ├─ q_b_proj.weight
│   ├─ kv_a_proj.weight
│   └─ kv_b_proj.weight
│   ├─ o_proj.weight
│   └─ norm.weight
│
├─ MoE weights: ~5 个
│   ├─ gate_proj.weight
│   ├─ up_proj.weight
│   ├─ down_proj.weight
│   └─ shared_expert weights
│   └─ norm.weight
│
├─ HyperConnection post-MLP: ~3 个
│   ├─ hc_ffn_pre.hc_fn
│   ├─ hc_ffn_pre.hc_base
│   └─ hc_ffn_pre.hc_scale
│
└─ 单层总计：约 13 个权重
```

---

## 三、权重输入顺序规范

### 1. 关键约束：Graph 构建顺序 = 执行传入顺序

**必须保序**：输入的创建顺序必须与执行时传入顺序严格一致

```
权重输入顺序规范：
┌────────────────────────────────────────────────────────────┐
│  Graph 构建（按顺序创建 placeholder）：                       │
│                                                            │
│  input_index = 0:                                          │
│    builder->CreateInput(0, "input_ids", ...)              │
│                                                            │
│  input_index = 1:                                          │
│    builder->CreateInput(1, "embed_weight", ...)            │
│                                                            │
│  input_index = 2-560:                                      │
│    for (int layer = 0; layer < 43; layer++) {              │
│      for (int w = 0; w < 13; w++) {                        │
│        builder->CreateInput(input_index++, ...)            │
│      }                                                     │
│    }                                                       │
│                                                            │
│  input_index = 561:                                        │
│    builder->CreateInput(561, "norm_weight", ...)           │
│                                                            │
│  input_index = 562:                                        │
│    builder->CreateInput(562, "lm_head_weight", ...)        │
│                                                            │
│  input_index = 563-565:                                    │
│    builder->CreateInput(563, "hc_head_fn", ...)            │
│    builder->CreateInput(564, "hc_head_base", ...)          │
│    builder->CreateInput(565, "hc_head_scale", ...)         │
│                                                            │
└────────────────────────────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│  Graph 执行（按相同顺序传入 tensor）：                        │
│                                                            │
│  inputs.push_back(tokens);              → 对应 input 0     │
│  inputs.push_back(embed_tokens_->weight()); → 对应 input 1 │
│                                                            │
│  for (int layer = 0; layer < 43; layer++) {                │
│    auto layer_weights = get_layer_weights(layer);          │
│    for (auto& w : layer_weights) {                         │
│      inputs.push_back(convert(w));      → 对应 input 2-560 │
│    }                                                       │
│  }                                                         │
│                                                            │
│  inputs.push_back(norm_->weight());     → 对应 input 561   │
│  inputs.push_back(lm_head_->weight());  → 对应 input 562   │
│  inputs.push_back(hc_head_fn_);         → 对应 input 563   │
│  inputs.push_back(hc_head_base_);       → 对应 input 564   │
│  inputs.push_back(hc_head_scale_);      → 对应 input 565   │
│                                                            │
│  session_->RunGraph(graph_id, inputs, outputs);            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 2. 顺序不一致会导致的错误

```
如果顺序不一致：
├─ Graph 构建：CreateInput(0, "input_ids"), CreateInput(1, "embed_weight")
├─ Graph 执行：inputs.push_back(embed_weight), inputs.push_back(input_ids)
│
└─ 结果：
    ├─ embed_weight 被当作 input_ids 使用 → 数据类型错误
    ├─ input_ids 被当作 embed_weight 使用 → shape 不匹配
    └─ Graph 执行失败或推理结果错误
```

---

## 四、权重参数提取（从 torch::Tensor）

### 1. 提取 shape 和 dtype

```cpp
// 从 weight tensor 提取参数
torch::Tensor weight = embed_tokens_->weight();

// 提取 shape
std::vector<int64_t> shape = weight.sizes().vec();
// 例如：[vocab_size, hidden_size] = [129280, 4096]

// 提取 dtype
torch::ScalarType torch_dtype = weight.scalar_type();
ge::DataType ge_dtype = GeTensorConverter::map_dtype(torch_dtype);
// 例如：torch::kBFloat16 → ge::DT_BF16

// 创建 GE input placeholder
auto weight_input = builder->CreateInput(
    input_index,           // 输入序号
    "embed_weight",        // 输入名称
    ge_dtype,              // 数据类型
    shape                  // 形状
);
```

---

### 2. Dtype 映射表

```cpp
ge::DataType GeTensorConverter::map_dtype(torch::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case torch::kFloat32:   return ge::DT_FLOAT;
    case torch::kFloat16:   return ge::DT_FLOAT16;
    case torch::kBFloat16:  return ge::DT_BF16;
    case torch::kInt32:     return ge::DT_INT32;
    case torch::kInt64:     return ge::DT_INT64;
    default:
      LOG(FATAL) << "Unsupported dtype: " << torch_dtype;
      return ge::DT_UNDEFINED;
  }
}
```

---

## 五、权重映射策略

### 1. 映射原则：只要能够映射即可

**关键设计**：
- ✅ 不一定是 layers 结构
- ✅ 可以通过任意方式访问权重
- ✅ 只要保证顺序一致即可

---

### 2. 推荐的映射方案

**方案 1：按成员变量直接映射（推荐）**

```cpp
class DeepseekV4GeModelImpl {
 private:
  torch::nn::Embedding embed_tokens_{nullptr};
  std::vector<DeepseekV4GeDecoderLayer> layers_;
  RMSNorm norm_{nullptr};
  torch::nn::Linear lm_head_{nullptr};
  torch::Tensor hc_head_fn_;
  torch::Tensor hc_head_base_;
  torch::Tensor hc_head_scale_;
  
  // 收集所有权重（按顺序）
  std::vector<torch::Tensor> collect_all_weights() {
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
    
    return weights;
  }
};
```

---

### 3. 需要新增的辅助方法

```cpp
// DeepseekV4GeDecoderLayer 需要新增的方法
class DeepseekV4GeDecoderLayer {
 public:
  // 获取该 layer 的所有权重（按固定顺序）
  std::vector<torch::Tensor> get_all_weights() {
    std::vector<torch::Tensor> weights;
    
    weights.push_back(hc_attn_pre_->get_fn());
    weights.push_back(hc_attn_pre_->get_base());
    weights.push_back(hc_attn_pre_->get_scale());
    weights.push_back(attention_->get_q_a_proj_weight());
    weights.push_back(attention_->get_q_b_proj_weight());
    weights.push_back(attention_->get_kv_a_proj_weight());
    weights.push_back(attention_->get_kv_b_proj_weight());
    weights.push_back(attention_->get_o_proj_weight());
    weights.push_back(attention_->get_norm_weight());
    weights.push_back(moe_->get_gate_proj_weight());
    weights.push_back(moe_->get_up_proj_weight());
    weights.push_back(moe_->get_down_proj_weight());
    weights.push_back(moe_->get_norm_weight());
    weights.push_back(hc_ffn_pre_->get_fn());
    weights.push_back(hc_ffn_pre_->get_base());
    weights.push_back(hc_ffn_pre_->get_scale());
    
    return weights;
  }
};
```

---

## 六、成员变量生命周期管理

### 1. 生命周期时序

```
┌────────────────────────────────────────────────────────────┐
│  Phase 1: 权重加载                                           │
│                                                            │
│  load_state_dict()                                         │
│  ├─ embed_tokens_->load_state_dict(...)                   │
│  ├─ layers_[i]->load_state_dict(...)                      │
│  ├─ norm_->load_state_dict(...)                           │
│  └─ 权重加载到成员变量（device 地址）                        │
│                                                            │
│  成员变量状态：包含权重 tensor（device）                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│  Phase 2: Graph 构建                                         │
│                                                            │
│  build_model_graph()                                       │
│  ├─ 创建权重输入 placeholder（约 565 个）                    │
│  ├─ 从成员变量提取 weight shape/dtype                        │
│  ├─ 引用成员变量权重构建算子                                  │
│  └─ BuildAndReset() → model_graph_                         │
│                                                            │
│  成员变量状态：仍包含权重（用于 Graph 构建）                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│  Phase 3: Graph 编译                                         │
│                                                            │
│  compile_prefill_graph()                                   │
│  ├─ frozen_parameter=true → 权重地址必须不变                 │
│  └─ CompileGraph()                                         │
│                                                            │
│  compile_decode_graph()                                    │
│  ├─ frozen_parameter=true → 权重地址必须不变                 │
│  └─ CompileGraph()                                         │
│                                                            │
│  成员变量状态：仍包含权重（地址不能变）                        │
│                                                            │
└────────────────────────────────────────────────────────────┘
                     ↓
┌────────────────────────────────────────────────────────────┐
│  Phase 4: Graph 执行（每次执行）                              │
│                                                            │
│  run_prefill_graph(...)                                    │
│  ├─ 传入 input_ids                                          │
│  ├─ **传入成员变量权重 tensor**（每次执行需要）               │
│  │   ├─ embed_tokens_->weight()                            │
│  │   ├─ layers_[i]->get_all_weights()                      │
│  │   ├─ norm_->weight()                                    │
│  │   ├─ lm_head_->weight()                                 │
│  │   └─ hc_head_*                                          │
│  ├─ RunGraph(inputs, outputs)                              │
│  └─ frozen_parameter=true → 地址不变优化                    │
│                                                            │
│  成员变量状态：每次执行都需要（不可释放）                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

### 2. 成员变量必要性总结

| 成员变量 | Phase 1 (权重加载) | Phase 2 (Graph 构建) | Phase 3 (Graph 编译) | Phase 4 (每次执行) |
|---------|-------------------|--------------------|--------------------|--------------------|
| `embed_tokens_` | ✅ **必要** | ✅ **必要** | ✅ **必要** | ✅ **必要** |
| `layers_` | ✅ **必要** | ✅ **必要** | ✅ **必要** | ✅ **必要** |
| `norm_` | ✅ **必要** | ✅ **必要** | ✅ **必要** | ✅ **必要** |
| `lm_head_` | ✅ **必要** | ✅ **必要** | ✅ **必要** | ✅ **必要** |
| `hc_head_*` | ✅ **必要** | ✅ **必要** | ✅ **必要** | ✅ **必要** |

**结论**：
- ✅ 成员变量在所有阶段都必要
- ❌ **编译后不可释放**（与之前错误理解相反）
- ✅ 每次执行都需要传入权重 tensor

---

## 七、Rolling Load 不支持的原因

### 1. 根本原因：frozen_parameter=true 要求地址不变

```
Rolling Load 机制：
├─ free_weights() → 释放权重 → 地址改变
├─ reload_weights() → 加载新权重 → 新地址
├─ 地址变化 → frozen_parameter=true 失效
│
└─ 结果：
    ├─ frozen_parameter=true 要求权重地址不变
    ├─ Rolling Load 导致地址变化
    └─ Graph 模式不支持 Rolling Load
```

---

### 2. Graph 模式限制

```
Graph 执行粒度限制：
├─ Graph 整体执行（所有 43 layers + Embedding + LM Head）
├─ 无法中途更新权重输入
├─ RunGraph 是一次性执行
│
└─ Rolling Load 需要：
    ├─ 在 layer 间插入权重更新逻辑
    ├─ Graph 无法支持中途插入
    └─ Graph 模式不支持 Rolling Load
```

---

### 3. 报错处理

```cpp
// deepseek_v4.h
void free_weights() {
  LOG(ERROR) << "free_weights() not supported in GE Graph mode";
  CHECK(false) << "Rolling Load is not supported in GE Graph mode. "
               << "Weights are passed as inputs, frozen_parameter=true "
               << "requires address stability.";
}

void reload_weights() {
  LOG(ERROR) << "reload_weights() not supported in GE Graph mode";
  CHECK(false) << "Rolling Load is not supported in GE Graph mode. "
               << "Weights are passed as inputs, frozen_parameter=true "
               << "requires address stability.";
}
```

---

## 八、内存需求和适用场景

### 1. 内存需求计算

```
DeepSeek V4 权重大小估算：
├─ Embedding: vocab_size × hidden_size = 129280 × 4096 ≈ 0.5 GB
├─ 43 Layers:
│   ├─ 每层约 1 GB
│   └─ 43 × 1 = 43 GB
├─ Final Norm: hidden_size = 4096 ≈ 0.01 GB
├─ LM Head: hidden_size × vocab_size = 4096 × 129280 ≈ 0.5 GB
├─ HyperConnection Head: 约 0.01 GB
│
└─ 总计：约 44 GB
```

---

### 2. 适用场景

| 场景 | DS2 ATB | DS4 GE |
|------|---------|--------|
| **HBM 充足（≥ 44 GB）** | ✅ 支持（enable_rolling_load=false） | ✅ **支持** |
| **HBM 有限（< 44 GB）** | ✅ 支持（enable_rolling_load=true） | ❌ **不支持** |

**分工**：
- DS2 ATB：HBM 有限场景（支持 Rolling Load）
- DS4 GE：HBM 充足场景（权重常驻，性能优化）

---

## 九、方案 A vs 方案 B（未来优化）

### 方案 A：权重不合并（当前选择）

**优点**：
- ✅ 实现简单（直接传入成员变量权重）
- ✅ 权重管理清晰（每个权重独立）
- ✅ 打通功能优先

**缺点**：
- ⚠️ 输入数量多（约 565 个）
- ⚠️ Graph 构建复杂（需要创建 565 个 placeholder）
- ⚠️ 执行时需要传入 565 个 tensor

---

### 方案 B：权重合并（未来优化）

**合并策略**：
- 合并所有 43 layers 权重到一个大 buffer
- 减少输入数量：从 565 个 → 约 6 个

**优点**：
- ✅ 输入数量少（约 6 个）
- ✅ Graph 构建简单
- ✅ 执行效率高（减少 tensor 传入开销）

**缺点**：
- ⚠️ 需要额外实现 WeightBuffer 类
- ⚠️ 需要管理 layer_offsets
- ⚠️ 权重切片逻辑复杂

**计划**：
- 当前：方案 A（打通功能优先）
- 未来：方案 B（性能优化阶段）

---

## 十、架构设计总结

```
最终架构设计（方案 A）：
├─ 权重处理：作为 Graph 输入（每次执行传入）
├─ 权重数量：约 565 个独立输入（不合并）
├─ 权重映射：只要能映射即可（不一定是 layers）
├─ 输入顺序：Graph 构建顺序 = 执行传入顺序（保序）
├─ 权重参数：从 weight tensor 提取（shape, dtype）
├─ frozen_parameter=true：权重地址不变（运行时优化）
├─ 成员变量：每次执行需要（不可释放）
├─ Rolling Load：不支持（报错处理）
└─ 内存需求：约 44 GB（所有权重常驻 HBM）
```

---

## 十一、实现要点

### 1. Graph 构建要点

```cpp
void build_model_graph() {
  // 要点 1：按顺序创建所有输入 placeholder
  int input_index = 0;
  
  // 要点 2：从 weight tensor 提取 shape/dtype
  auto weights = collect_all_weights();
  
  // 要点 3：为每个 weight 创建 placeholder
  for (auto& w : weights) {
    builder->CreateInput(input_index++, "weight", 
                         map_dtype(w.scalar_type()), 
                         w.sizes().vec());
  }
  
  // 要点 4：构建算子（引用 weight placeholders）
  // ...
  
  // 要点 5：BuildAndReset
  model_graph_ = builder->BuildAndReset({logits});
}
```

---

### 2. Graph 执行要点

```cpp
void run_prefill_graph(...) {
  // 要点 1：按相同顺序收集所有权重
  auto weights = collect_all_weights();
  
  // 要点 2：转换为 GE tensor
  std::vector<ge::Tensor> inputs;
  inputs.push_back(convert(tokens));
  for (auto& w : weights) {
    inputs.push_back(convert(w));
  }
  
  // 要点 3：RunGraph
  session_->RunGraph(prefill_graph_id_, inputs, outputs);
}
```

---

### 3. frozen_parameter 配置要点

```cpp
std::map<ge::AscendString, ge::AscendString> options = {
  {"ge.inputShape", "input_ids:-1"},
  {"ge.dynamicDims", "1,4,8,16,32,64,128"},
  {"frozen_parameter", "true"}  // ← 权重地址不变
};
```

---

**设计文档结束**