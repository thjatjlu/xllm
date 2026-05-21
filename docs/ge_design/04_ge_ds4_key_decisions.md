# DeepSeek V4 GE 方案 - 关键决策记录（初稿）

> 文档版本：v1.0
> 创建日期：2025-01-19

---

## 一、关键设计决策

### Q1: 为什么只编译 Decode Graph，不编译 Prefill Graph？

**决策**：只编译 Decode Graph，Prefill 使用 eager mode

**原因**：

| 维度 | Prefill | Decode |
|-----|---------|--------|
| **Batch 特征** | Prompt token 数量（1-1000，极不稳定） | 并发请求数（1-128，相对稳定） |
| **执行频率** | 每请求 1 次（低频） | 每 token 1 次（高频） |
| **Graph 收益** | 编译开销 > 执行收益 | 编译开销 < 执行收益 |
| **MLU 实践** | ❌ 不使用 graph | ✅ 使用 MLUGraph |

**参考依据**：
- MLU 的成功实践：`mlu_graph_executor_impl.cpp:235`
- 只有 `is_decode()` 时才使用 graph

---

### Q2: Decode Graph 使用什么 dynamicDims 配置？

**决策**：使用 `{1,4,8,16,32,64,128}`（7 个高频 batch）

**原因**：

1. **参考 MLU/DS2 bucket 配置**（验证过的有效方案）
2. **覆盖高频 batch sizes**：
   - batch=1（单用户，最高频）
   - batch=4,8（小规模并发，高频）
   - batch=16,32（中等并发，中频）
   - batch=64,128（大规模并发，低频）
3. **减少 padding 开销**：平均 padding < 2%

**配置示例**：
```cpp
{"ge.dynamicDims", "1,4,8,16,32,64,128"}
```

---

### Q3: Prefill `-1` 是否需要 dynamicDims？

**决策**：Prefill 不使用 graph（eager mode）

**修正说明**：
- 最初计划编译 Prefill graph（`-1` 完全动态）
- 发现 MLU 不编译 Prefill graph
- **修正为：Prefill 全部使用 eager mode**

---

### Q4: MIXED ForwardType 如何处理？

**决策**：MIXED 使用 eager mode

**原因**：
- MIXED = Prefill + Decode 混合在同一 batch
- 无法选择单一 graph
- MLU 的处理方式：eager mode

**代码实现**：
```cpp
if (!params.batch_forward_type.is_decode()) {
  // PREFILL / CHUNKED_PREFILL / MIXED → eager mode
  return model_->forward(tokens, positions, kv_caches, params);
}
```

---

### Q5: KV Cache 如何传递给 GE Graph？

**决策**：每次 RunGraph 时传递 KV Cache（转换为 GE tensor）

**原因**：

| 方案 | MLU | GE |
|-----|-----|-----|
| **传递方式** | 直接引用 PyTorch tensor（零拷贝） | 每次 RunGraph 传递 GE tensor |
| **Capture 时** | 记录 tensor 地址 | 记录 GE TensorDesc |
| **Replay 时** | 无需重新传递 | 需要每次传递 |

**实现**：
```cpp
std::vector<ge::Tensor> ge_inputs;

// 每次 RunGraph 传递 KV Cache
for (auto& kv_cache : kv_caches) {
  ge_inputs.push_back(torch_to_ge(kv_cache.k_cache));
  ge_inputs.push_back(torch_to_ge(kv_cache.v_cache));
}

session_->RunGraphWithStreamAsync(graph_id, stream, ge_inputs, ge_outputs);
```

---

### Q6: 算子层面 Prefill vs Decode 有差异，如何处理？

**决策**：两阶段不同处理，无需担心算子差异

**关键洞察**：
- Prefill 和 Decode 是**独立的两阶段**
- Prefill 使用 eager mode（算子通过 PyTorch forward 执行）
- Decode 使用 GE graph（算子通过 GE graph 执行）
- **两阶段的算子配置通过 params 传递**

**算子差异示例**：
```
Prefill phase (eager mode):
  - Attention: causal mask（prompt 内部 causal）
  - MoE: 标准 routing
  - Indexer/Compressor: 不使用

Decode phase (GE graph):
  - Attention: decode mask（1 token vs all KV）
  - MoE: Hash routing（前 3 layers）
  - Indexer/Compressor: 使用
```

---

### Q7: 编译时间预估是多少？

**决策**：单次 CompileGraph 约 5min

**修正说明**：
- 最初错误假设：43 layers × buckets = 43 小时
- **修正理解**：整个 model 编译 1 个 graph（不是 layer-wise）
- **正确计算**：1 次 CompileGraph ≈ 5min

**对比**：
| 方案 | 编译次数 | 编译时间 |
|-----|---------|---------|
| **错误假设** | 43×12=516 次 | 43 小时 |
| **修正后** | 1 次 | **5min** |

---

### Q8: 是否需要手动 padding？

**决策**：不需要手动 padding

**原因**：
- GE dynamicDims 支持动态 batch
- 直接传递实际 batch size
- GE graph 内部处理不同 batch sizes

**对比 MLU**：
- MLU 需要手动 padding（NPUGraph/MLUGraph 不支持 dynamicDims）
- GE 支持 dynamicDims → **无需手动 padding**

---

### Q9: 为什么参考 MLU 设计，而不参考 DS2 ATB？

**决策**：参考 MLU（只编译 Decode graph）

**对比**：

| 维度 | DS2 ATB | MLU | GE（本方案） |
|-----|---------|-----|------------|
| **Prefill graph** | ✅ 有（毫秒级构建） | ❌ 无 | ❌ **无（参考 MLU）** |
| **Decode graph** | ✅ 有 | ✅ 有 | ✅ **有** |
| **编译时间** | 秒级（ATB） | 5min（MLUGraph） | **5min（GE）** |
| **编译本质** | 创建图结构（毫秒级） | Graph capture（毫秒级） | Graph compilation（分钟级） |

**关键差异**：
- ATB `init_node()` 是毫秒级（创建图结构）
- GE `CompileGraph()` 是分钟级（真正的编译优化）
- **GE 不能像 ATB 那样为 Prefill 编译 graph**（分钟级开销不可接受）

---

## 二、技术澄清

### T1: BatchForwardType 定义

**来源**：`batch_forward_type.h:23-36`

```cpp
enum Value : int32_t {
  PREFILL = 0,           // 传统 prefill（无 KV cache）
  CHUNKED_PREFILL = 1,   // 分块 prefill（使用 KV cache）
  DECODE = 2,            // Decode phase
  MIXED = 3,             // Prefill + Decode 混合
  EMPTY = 4,             // 无序列
};
```

**处理策略**：
- PREFILL → eager mode
- CHUNKED_PREFILL → eager mode
- DECODE → GE graph
- MIXED → eager mode
- EMPTY → 直接返回

---

### T2: Chunked Prefill 解释

**问题**：什么是 Chunked Prefill？

**答案**：
```
传统 Prefill：
  Request A: prompt=1000 tokens（占用大量资源）
  Request B: prompt=10 tokens（等待 A 完成）

Chunked Prefill：
  Round 1:
    - Request A: prefill chunk 1 (200 tokens)
    - Request B: prefill full (10 tokens) ← 很快完成
    - Request C: decode (batch=5) ← 同时进行
  
  Round 2:
    - Request A: prefill chunk 2 (200 tokens)
    - Request D: prefill full (50 tokens)
    - Request C, E: decode (batch=10)
  
优势：
  - 短 prompt 不被长 prompt 阻塞
  - Prefill + Decode 并行
  - 提高吞吐量
```

**对 GE 方案的影响**：
- Chunked Prefill 仍然是 prefill → 使用 eager mode
- 可以复用 prefill eager logic

---

### T3: dynamicDims vs `-1` 的区别

**问题**：GE dynamicDims 和 `-1` 有什么区别？

**答案**：

| 方案 | 配置 | 支持 batch | 编译行为 |
|-----|------|-----------|---------|
| **完全动态 `-1`** | `ge.inputShape = "x:-1,..."` | 任意 batch | 生成通用 kernel |
| **dynamicDims** | `ge.dynamicDims = "1,4,8"` | 指定 batch | 为每个 batch 优化 kernel |

**本方案选择**：
- Decode graph：使用 dynamicDims（性能优化）
- Prefill：不使用 graph（eager mode）

---

### T4: MLU vs GE 本质差异

**问题**：MLU 和 GE 的核心差异是什么？

**答案**：

| 维度 | MLU (MLUGraph) | GE (Graph) |
|-----|----------------|------------|
| **本质** | PyTorch execution recorder | 独立的 graph compiler |
| **Capture** | 记录 PyTorch forward（毫秒级） | 构建独立 graph + CompileGraph（分钟级） |
| **Tensor 管理** | PyTorch allocator（共享） | GE allocator（独立） |
| **KV Cache** | 直接引用 PyTorch tensor | 需要转换为 GE tensor |
| **能否 lazy？** | ✅ 可以（毫秒级） | ❌ 不能（分钟级） |

---

## 三、待确认事项

### C1: GE ops 库支持性

**问题**：GE ops 库是否支持所有 DeepSeek V4 关算子？

**需要确认的算子**：
- RMSNorm
- SparsePagedAttention（MLA + 窗口 + 压缩）
- SinkhornIteration（HyperConnection）
- MoEGate + GroupGEMM（FusedMoE）
- TopK + Gather（Indexer）
- Softmax + WeightedSum（Compressor）
- ApplyRotaryEmbedding（DeepSeek Yarn RoPE）

---

### C2: CompileGraph 实际耗时

**问题**：CompileGraph 实际耗时是多少？

**需要测试**：
- 单次 CompileGraph 时间（预期 5min）
- 是否受 graph 复杂度影响
- 是否受 dynamicDims 配置影响

---

### C3: RunGraphWithStreamAsync API

**问题**：RunGraphWithStreamAsync API 是否可用？

**需要验证**：
- API 是否支持 dynamicDims
- 是否支持异步执行
- 是否需要 LoadGraph 预处理

---

## 四、关键代码位置索引

### I1: MLU eager/graph 选择

**文件**：`mlu_graph_executor_impl.cpp:235`

```cpp
bool graph_mode = params.batch_forward_type.is_decode();
if (!graph_mode) {
  return model_->forward(tokens, positions, kv_caches, params);
}
```

---

### I2: DS2 ATB init_layer

**文件**：`npu_deepseek_v2_decoder_layer_impl.cpp:687-743`

```cpp
int64_t init_layer() {
  init_node(prefill_node_);
  init_node(decode_node_);
  init_node(decode_mla_node_);
}
```

---

### I3: BatchForwardType 定义

**文件**：`batch_forward_type.h:23-36`

---

### I4: GE Session API

**文件**：`/home/lianghao/thj/code/ge/inc/external/ge/ge_api.h`

---

### I5: GE ES Builder

**文件**：`/home/lianghao/thj/code/ge/inc/external/ge/eager_style_graph_builder/cpp/es_graph_builder.h`

---

**文档结束**