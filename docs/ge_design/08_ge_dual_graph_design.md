# 双图方案设计说明（GE ES）

> 更新日期：2026-05-21
> 核心：ES 构图只一次，编译两次（Prefill Graph + Decode Graph）

---

## 一、双图方案核心设计

### 1. 核心流程

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: ES 构图（一次）                                      │
│                                                             │
│  auto builder = std::make_unique<ge::es::EsGraphBuilder>(); │
│  // 构建完整推理流程                                          │
│  auto graph = builder->BuildAndReset();                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                     ↓
       ┌─────────────┴─────────────┐
       ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│  Prefill Graph   │      │  Decode Graph    │
│  编译（约5min）   │      │  编译（约5min）   │
└──────────────────┘      └──────────────────┘
       ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│  -1 完全动态     │      │  dynamicDims分档 │
│  用于Prefill     │      │  用于Decode      │
└──────────────────┘      └──────────────────┘
```

### 2. 关键代码

```cpp
void DeepseekV4GeModelImpl::init_model() {
  // Step 1: ES 构建（一次）
  build_model_graph();

  // Step 2: 编译 Prefill Graph
  compile_prefill_graph();  // -1 dynamic shape

  // Step 3: 编译 Decode Graph
  compile_decode_graph();   // dynamicDims: 1,4,8,16,32,64,128
}

void build_model_graph() {
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_model");
  // 构建完整推理流程（一次）
  auto graph = builder->BuildAndReset();
  model_graph_ = std::move(graph);  // 保存 graph
}

void compile_prefill_graph() {
  std::map<ge::AscendString, ge::AscendString> prefill_options = {
    {"ge.inputShape", "input_ids:-1"}  // -1 完全动态
  };
  session_->AddGraph(prefill_graph_id_, *model_graph_, prefill_options);
  session_->CompileGraph(prefill_graph_id_);  // 编译（约5min）
}

void compile_decode_graph() {
  std::map<ge::AscendString, ge::AscendString> decode_options = {
    {"ge.inputShape", "input_ids:-1"},
    {"ge.dynamicDims", "1,4,8,16,32,64,128"}  // 分档
  };
  session_->AddGraph(decode_graph_id_, *model_graph_, decode_options);
  session_->CompileGraph(decode_graph_id_);  // 编译（约5min）
}
```

---

## 二、执行路径选择

```cpp
ModelOutput GeGraphExecutorImpl::run(...) {
  if (params.batch_forward_type.is_prefill()) {
    // PREFILL → Prefill Graph
    return run_prefill_graph(...);
  }

  if (params.batch_forward_type.is_decode()) {
    // DECODE → Decode Graph
    return run_decode_graph(...);
  }

  // MIXED → eager mode fallback
  return model_->forward(...);
}
```

---

## 三、双图方案对比表

| 维度 | Prefill Graph | Decode Graph |
|------|---------------|--------------|
| **ES 构建** | 同一个 graph（只构建一次） | 同一个 graph |
| **动态 shape** | `-1`（完全动态） | `dynamicDims`（分档） |
| **配置选项** | `ge.inputShape: "-1"` | `ge.inputShape: "-1"` + `ge.dynamicDims: "1,4,8,..."` |
| **编译时间** | 约 5min | 约 5min |
| **总编译时间** | **约 10min**（两次编译） | - |
| **适用场景** | PREFILL batches | DECODE batches |
| **输入 tokens** | 动态范围大（1-1000） | 固定分档（1,4,8,16,32,64,128） |
| **MIXED 处理** | ❌ 不使用 | ❌ 不使用（eager fallback） |

---

## 四、为什么使用双图方案？

### 1. Prefill Graph：-1 完全动态

**问题**：Prefill batch 的 tokens 数量动态范围大（1-1000）

**解决方案**：
- 使用 `-1` 完全动态 shape
- 编译一次，支持任意 tokens 数量
- 性能略低于分档，但灵活性高

**适用场景**：
- First prompt processing
- Chunked prefill（分块处理长 prompt）

---

### 2. Decode Graph：dynamicDims 分档

**问题**：Decode batch 的 batch_size 相对固定（常见：1,4,8,16,32,64,128）

**解决方案**：
- 使用 `dynamicDims` 分档优化
- 编译时针对常见 batch_size 优化
- 性能高于完全动态

**适用场景**：
- Token generation loop
- 批量 decode

---

### 3. 编译时间分析

| 方案 | 编译次数 | 编译时间 |
|------|---------|---------|
| **单图方案** | 1 次 | 约 5min |
| **双图方案** | 2 次 | **约 10min** |
| **多桶方案（DS2 ATB）** | 43+ 次 | 43+ 小时 |

**双图方案优势**：
- 编译时间可控（10min vs 43+ hours）
- 性能优化（分档 vs 完全动态）
- 灵活性（支持 Prefill 和 Decode）

---

## 五、与 DS2 ATB 的对比

| 维度 | DS2 ATB | DS4 GE ES |
|------|---------|-----------|
| **构建模式** | 每个 Layer 每个 bucket 一个 Operation | 一个 Graph，编译两次 |
| **编译次数** | 43 layers × N buckets | **2 次**（Prefill + Decode） |
| **编译时间** | 分钟级（ATB 构建快） | 分钟级（GE CompileGraph 约5min） |
| **优化程度** | ATB runtime优化 | **GE compiler深度优化** |
| **灵活性** | 多桶支持多种batch_size | 分档支持常见batch_size |

---

## 六、实现细节

### 1. Model Graph 构建（一次）

```cpp
// xllm/models/llm/npu_ge/deepseek_v4.cpp
void build_model_graph() {
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_model");

  // Step 1: Embedding lookup
  auto input_ids = builder->CreateInput(0, "input_ids", ge::DT_INT32, {-1});
  auto hidden = Embedding(input_ids, embed_tokens_->weight());

  // Step 2-8: 完整推理流程
  // ...

  // Step 9: Build graph（一次）
  model_graph_ = builder->BuildAndReset({logits});
}
```

### 2. Prefill Graph 编译

```cpp
void compile_prefill_graph() {
  std::map<ge::AscendString, ge::AscendString> prefill_options = {
    {"ge.inputShape", "input_ids:-1"}  // 完全动态
  };

  prefill_graph_id_ = session_manager.next_graph_id();
  session_->AddGraph(prefill_graph_id_, *model_graph_, prefill_options);

  ge::Status ret = session_->CompileGraph(prefill_graph_id_);
  CHECK(ret == ge::SUCCESS) << "Compile Prefill Graph failed: " << ret;
}
```

### 3. Decode Graph 编译

```cpp
void compile_decode_graph() {
  std::map<ge::AscendString, ge::AscendString> decode_options = {
    {"ge.inputShape", "input_ids:-1"},
    {"ge.dynamicDims", "1,4,8,16,32,64,128"}  // 分档
  };

  decode_graph_id_ = session_manager.next_graph_id();
  session_->AddGraph(decode_graph_id_, *model_graph_, decode_options);

  ge::Status ret = session_->CompileGraph(decode_graph_id_);
  CHECK(ret == ge::SUCCESS) << "Compile Decode Graph failed: " << ret;
}
```

---

## 七、MIXED 模式处理

**双图方案不处理 MIXED batch**：

| ForwardType | 执行模式 | 原因 |
|-------------|---------|------|
| **PREFILL** | Prefill Graph | ✅ 纯 Prefill batch |
| **DECODE** | Decode Graph | ✅ 纯 Decode batch |
| **MIXED** | Eager Mode | ⚠️ 包含 Prefill + Decode sequences，无法用单一 graph |

**MIXED fallback to eager**：
- Prefill sequences 和 Decode sequences 在同一 batch
- 算子内部需要 per-sequence 判断（graph 无法实现）
- 只能使用 eager mode

---

## 八、总结

### 核心设计

```
✅ ES 构图：一次（构建完整推理流程）
✅ Graph 编译：两次（Prefill + Decode）
✅ 总编译时间：约 10min（可控）
✅ Prefill Graph：-1 完全动态（灵活性）
✅ Decode Graph：dynamicDims 分档（性能优化）
✅ MIXED 模式：eager fallback（正确性保证）
```

### 优势

```
✅ 编译时间可控（10min vs 43+ hours）
✅ 性能优化（GE compiler 深度优化）
✅ 灵活性（支持 Prefill 和 Decode）
✅ 正确性（MIXED mode fallback）
✅ 与 DS4 Python 一致（torch.compile 双图模式）
```

---

**双图方案设计说明结束**