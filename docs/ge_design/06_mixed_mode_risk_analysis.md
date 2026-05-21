# DeepSeek V4 GE 双图方案 - MIXED 模式风险分析

> 文档版本：v1.0
> 创建日期：2025-01-19
> 状态：风险分析文档

---

## 一、MIXED 模式定义与触发条件

### 1.1 MIXED 模式定义

**BatchForwardType 定义**（`batch_forward_type.h:33`）：
```cpp
MIXED = 3  // Mixed prefill and decode in one batch when doing chunked prefill
```

**含义**：Batch 中同时包含 Prefill/Chunked Prefill sequences 和 Decode sequences。

---

### 1.2 MIXED 触发条件

**代码位置**：`batch.cpp:93-119`

**三种触发场景**：

| 当前 Batch 类型 | 新加入 Sequence 类型 | 结果 |
|---------------|-------------------|------|
| **PREFILL** | CHUNKED_PREFILL | CHUNKED_PREFILL |
| **PREFILL** | DECODE | **MIXED** ← 触发 |
| **CHUNKED_PREFILL** | DECODE | **MIXED** ← 触发 |
| **DECODE** | PREFILL/CHUNKED_PREFILL | **MIXED** ← 触发 |
| **MIXED** | 任意 | MIXED（保持） |

**典型场景**：
```
Chunked Prefill + Decode 并行：
  Round 2:
    Request A: prefill chunk 2 (200 tokens)  → SequenceStage::CHUNKED_PREFILL
    Request B: decode (1 token)              → SequenceStage::DECODE
    
  Batch 检测到混合 → 设置为 MIXED
```

---

## 二、方案 A：MIXED 使用 Eager Fallback（推荐方案）

### 2.1 设计方案

**执行逻辑**：
```cpp
ModelOutput GeGraphExecutorImpl::run(...) {
  
  switch (params.batch_forward_type.value()) {
    
    case BatchForwardType::PREFILL:
    case BatchForwardType::CHUNKED_PREFILL:
      // Prefill Graph（优化）
      return run_prefill_graph(...);
    
    case BatchForwardType::DECODE:
      // Decode Graph（优化）
      return run_decode_graph(...);
    
    case BatchForwardType::MIXED:
      // ← Eager Fallback（风险点）
      return model_->forward(...);  // Eager mode
    
    case BatchForwardType::EMPTY:
      return ModelOutput();
  }
}
```

**关键决策**：
- ✅ PREFILL / CHUNKED_PREFILL → Prefill Graph（-1）
- ✅ DECODE → Decode Graph（dynamicDims）
- ⚠️ **MIXED → Eager Mode**（fallback）

---

### 2.2 风险分析（方案 A）

#### 风险 1：性能损失风险（中风险）

**问题描述**：
- MIXED 场景使用 Eager mode → 无 Graph 优化
- 如果 MIXED 频率高于预期 → 性能下降明显

**风险量化**：
```
假设场景：
  - Prefill Graph 性能提升：20%
  - MIXED 频率：预期 10%，实际可能 20-30%
  - MIXED 性能损失：20%（Eager vs Graph）
  
整体性能影响：
  - MIXED 频率 10%  → 性能损失 2%（可接受）
  - MIXED 频率 20%  → 性能损失 4%（需关注）
  - MIXED 频率 30%  → 性能损失 6%（需优化）
  - MIXED 频率 50%  → 性能损失 10%（不可接受）
```

**缓解措施**：
- 监控 MIXED 频率（添加 metrics counter）
- 如果频率过高 → 调整 chunked prefill 配置
- 长期优化：实现 MIXED Graph 或优化 Eager 性能

---

#### 风险 2：输出一致性风险（高风险）

**问题描述**：
- Eager mode 和 Graph mode 可能输出不一致
- 精度差异可能导致推理结果错误

**可能原因**：
```
精度差异来源：
  1. Graph 编译优化（算子融合）引入精度误差
  2. Eager mode 使用 PyTorch native 算子
  3. GE Graph 使用 GE ops（算子实现不同）
  4. 浮点运算顺序不同（累加顺序）
```

**验证方法**：
```cpp
// 测试代码
torch::Tensor prefill_eager = model_->forward(prefill_tokens, ...);
torch::Tensor prefill_graph = run_prefill_graph(prefill_tokens, ...);

float relative_error = 
    (prefill_eager - prefill_graph).abs().max() / prefill_eager.abs().max();

// 验证标准：relative_error < 0.01（< 1%）
```

**缓解措施**：
- **必须验证**：Prefill Graph vs Prefill Eager 输出一致性
- **必须验证**：Decode Graph vs Decode Eager 输出一致性
- 如果误差过大 → 调整 GE 编译选项（降低优化强度）

---

#### 风险 3：内存管理风险（中风险）

**问题描述**：
- Graph mode 使用 GE memory pool（预分配、复用）
- Eager mode 使用 PyTorch allocator（动态分配）
- 混用可能导致内存碎片或峰值增加

**风险场景**：
```
内存使用模式：
  
  Prefill Graph:
    - GE memory pool（预分配）
    - 内存峰值：固定
  
  Decode Graph:
    - GE memory pool（预分配）
    - 内存峰值：固定
  
  MIXED Eager:
    - PyTorch allocator（动态）
    - 内存峰值：可能高于 Graph
    - 内存碎片：可能累积
```

**缓解措施**：
- 监控内存使用（memory profiling）
- 验证 Eager mode 内存峰值是否超出 NPU memory limit
- 如果峰值过高 → 优化 Eager mode 内存使用

---

#### 风险 4：稳定性风险（中风险）

**问题描述**：
- Eager mode 和 Graph mode 混用可能引入稳定性问题
- 运行时切换模式可能有资源冲突

**可能问题**：
```
稳定性风险来源：
  1. Session 状态管理（Eager vs Graph 切换）
  2. Stream 管理（不同执行模式使用不同 stream）
  3. KV Cache 管理（Eager 写入 vs Graph 写入）
  4. 错误处理（Eager 和 Graph 错误处理不同）
```

**缓解措施**：
- 确保 Session 状态一致
- 确保 Stream 管理正确
- 完善 KV Cache 管理
- 完善错误处理

---

#### 风险 5：频率不可控风险（中风险）

**问题描述**：
- MIXED 频率取决于 Scheduler 配置
- 用户配置不当 → MIXED 频率高于预期

**配置参数**：
```cpp
FLAGS_enable_chunked_prefill = true;
FLAGS_max_chunk_tokens = 256;
FLAGS_max_seqs_per_batch = 128;
FLAGS_max_tokens_per_batch = 4096;
```

**频率影响因素**：
```
影响因素：
  1. Chunk size 配置（小 chunk → MIXED 频率低）
  2. Max tokens per batch（大 batch → MIXED 频率高）
  3. 请求 pattern（长短 prompt 混合 → MIXED 频率高）
  4. Scheduler 调度策略（激进调度 → MIXED 频率高）
```

**缓解措施**：
- 提供配置指南（推荐 chunked prefill 参数）
- 监控 MIXED 频率（如果过高 → 建议调整配置）
- 提供配置选项（禁用 chunked prefill → 无 MIXED）

---

### 2.3 风险汇总表（方案 A）

| 风险类型 | 严重程度 | 发生概率 | 影响 | 缓解措施 | 建议行动 |
|---------|---------|---------|------|---------|---------|
| **输出一致性** | **高** | 中 | 推理结果错误 | **必须验证** | P0（上线前必须测试） |
| **性能损失** | 中 | 中 | 性能下降 2-10% | 监控频率 | P1（上线后监控） |
| **内存管理** | 中 | 低 | 内存峰值增加 | Memory profiling | P1（测试阶段验证） |
| **稳定性** | 中 | 低 | 运行时错误 | 完善错误处理 | P1（测试阶段验证） |
| **频率不可控** | 中 | 中 | MIXED 频率过高 | 配置指南 + 监控 | P1（上线后监控） |

---

## 三、方案 B：MIXED 使用 Prefill Graph 执行（不推荐方案）

### 3.1 设计方案（假设）

**尝试使用 Prefill Graph（-1）处理 MIXED**：

```cpp
ModelOutput GeGraphExecutorImpl::run(...) {
  
  switch (params.batch_forward_type.value()) {
    
    case BatchForwardType::PREFILL:
    case BatchForwardType::CHUNKED_PREFILL:
    case BatchForwardType::MIXED:  // ← 尝试使用 Prefill Graph
      return run_prefill_graph(...);  // Prefill Graph
    
    case BatchForwardType::DECODE:
      return run_decode_graph(...);  // Decode Graph
  }
}
```

**假设理由**：
- Prefill Graph 使用 `-1` 动态 shape → 支持任意 batch size
- MIXED batch 的 tokens 数量动态 → `-1` 理论上可支持

---

### 3.2 关键问题：`is_prefill` 参数冲突

#### 问题分析

**Prefill Graph 编译时的参数**：
```cpp
// Prefill Graph 编译
build_prefill_graph() {
  // Attention 内部逻辑（attention.cpp:553）
  bool is_prefill = attn_metadata.is_prefill;  // ← 编译时固定
  
  // 如果 is_prefill=true（Prefill path）
  if (is_prefill) {
    offsets = query_lens;          // Prefill: offsets = 实际 query length
    kv_merge = merge_kv(...);      // Prefill: 合并 KV
  }
}
```

**MIXED batch 的参数设置**（`attention_metadata_builder.cpp:98-101`）：
```cpp
attn_metadata.is_prefill = params.batch_forward_type.is_prefill();

// ForwardType → is_prefill 映射：
//  PREFILL          → is_prefill=true
//  CHUNKED_PREFILL  → is_prefill=false
//  MIXED            → is_prefill=false  ← 关键！
//  DECODE           → is_prefill=false
```

---

#### 核心矛盾

**MIXED batch 使用 Prefill Graph 的后果**：

| Sequence Type | Prefill Graph 处理（is_prefill=true） | 结果 |
|--------------|--------------------------------------|------|
| **Prefill A (200 tokens)** | offsets=200，KV merge | ✅ 正确 |
| **Prefill B (50 tokens)** | offsets=50，KV merge | ✅ 正确 |
| **Decode C (1 token)** | offsets=1，KV merge | ❌ **错误**（Decode 不应 merge） |
| **Decode D (1 token)** | offsets=1，KV merge | ❌ **错误**（Decode 不应 merge） |

**问题详解**：

1. **Decode sequences 的 offsets 错误**：
   - Prefill path: `offsets = query_lens`（实际 tokens 数量）
   - Decode 应该: `offsets = window_size`（固定 128）
   - Prefill Graph 会为 Decode tokens 设置 offsets=1（错误）

2. **Decode sequences 不应 KV merge**：
   - Prefill path 会执行 KV merge（合并原始 KV + 压缩 KV）
   - Decode tokens 只生成 1 token，不需要 merge
   - Prefill Graph 会错误地为 Decode tokens 执行 merge

---

### 3.3 风险分析（方案 B）

#### 风险 1：Attention 计算错误（高风险）

**问题描述**：
- Prefill Graph 的 Decode sequences 使用错误的 offsets
- KV cache 选择错误（只选择最近 1 token，而不是 window_size）

**后果**：
```
MIXED batch 示例：
  Prefill A: query_len=200, 正确 offsets=200 → KV 选择正确
  Prefill B: query_len=50,  正确 offsets=50  → KV 选择正确
  Decode C:  query_len=1,   错误 offsets=1    → KV 只选 1 token（应选 128）
  
Attention 计算错误：
  - Decode C 的 KV cache 只包含最近 1 token（缺少历史上下文）
  - Attention 输出完全错误（丢失历史信息）
```

**严重性**：
- ❌ **推理结果完全错误**（不可接受）
- ❌ 无法通过后续修复（算子层面问题）

---

#### 风险 2：KV Cache 写入错误（高风险）

**问题描述**：
- Prefill path 的 KV cache 写入逻辑不适用于 Decode sequences

**代码差异**（`attention.cpp:416-468`）：

**Prefill KV写入**（多 tokens）：
```cpp
if (is_prefill) {
  // 写入多个 tokens（可能使用 ring buffer）
  reshape_paged_cache(kv_seq, slot_mapping);  // 写入 prompt tokens
}
```

**Decode KV写入**（单 token）：
```cpp
else {
  // 写入单个 token 到 ring buffer 位置
  int64_t slot_idx = (seqlen - 1) % window_size_;
  reshape_paged_cache(kv_seq, slot_mapping);  // 只写 1 token
}
```

**MIXED 使用 Prefill Graph 的后果**：
- Decode sequences 会错误地执行 Prefill KV写入逻辑
- 可能覆盖历史 KV cache（ring buffer 错误写入）

---

#### 风险 3：推理结果完全错误（极高风险）

**问题描述**：
- Decode sequences 的 Attention 计算错误 + KV cache 写入错误
- 推理输出完全不可用

**验证结果**：
```
如果尝试方案 B：
  
  验证测试：
    输入：MIXED batch（Prefill 200 tokens + Decode 1 token）
    Prefill Graph 输出：hidden_states
    
  对比基准：
    Eager mode 输出：hidden_states_correct
    
  验证结果：
    relative_error = (hidden_states - hidden_states_correct).abs().max()
    
  预期：
    relative_error > 50%（完全错误）
```

**结论**：
- ❌ **不可用**（推理结果完全错误）
- ❌ 无法通过精度调整修复
- ❌ 必须避免此方案

---

### 3.4 方案 B 的技术限制总结

| 限制 | 描述 | 严重性 |
|-----|------|-------|
| **is_prefill 固定** | Prefill Graph 编译时 `is_prefill=true`，无法动态切换 | 致命 |
| **offsets 错误** | Decode sequences 使用 query_lens 而非 window_size | 致命 |
| **KV merge 错误** | Decode sequences 不应 merge，但 Prefill Graph 会执行 | 致命 |
| **KV写入错误** | Prefill path 的写入逻辑不适用于 Decode tokens | 致命 |
| **推理结果错误** | Attention 计算和 KV cache 都错误 → 输出不可用 | 致命 |

**最终结论**：
- ❌ **方案 B 完全不可行**
- ❌ 技术限制无法解决（GE Graph 编译后参数固定）
- ✅ 必须使用方案 A（Eager fallback）

---

## 四、缓解策略与行动计划

### 4.1 Phase 1：验证阶段（上线前必须完成）

#### P0 任务（必须完成）

| 任务 | 验证内容 | 通过标准 | 文件位置 |
|-----|---------|---------|---------|
| **输出一致性测试** | Prefill Graph vs Prefill Eager | 精度误差 < 1% | `test_ge_prefill_consistency.cpp` |
| **输出一致性测试** | Decode Graph vs Decode Eager | 粯度误差 < 1% | `test_ge_decode_consistency.cpp` |
| **MIXED 功能测试** | Chunked prefill + decode 混合 | 输出正确 | `test_ge_mixed_forward.cpp` |

#### P1 任务（建议完成）

| 任务 | 验证内容 | 通过标准 |
|-----|---------|---------|
| **Memory profiling** | Prefill/Decode/MIXED 内存峰值 | < 80% NPU memory |
| **性能对比测试** | Prefill Graph vs Prefill Eager | 性能提升 > 10% |
| **MIXED 性能测试** | MIXED Eager vs 理论 MIXED Graph | 确认性能损失 |

---

### 4.2 Phase 2：监控阶段（上线后持续）

#### 监控指标

**代码实现**：
```cpp
// Metrics counters
COUNTER_DEFINE(num_prefill_graph_execution);      // Prefill Graph 执行次数
COUNTER_DEFINE(num_decode_graph_execution);       // Decode Graph 执行次数
COUNTER_DEFINE(num_mixed_eager_execution);        // ← MIXED Eager 执行次数

COUNTER_DEFINE(prefill_graph_latency_ms);         // Prefill Graph 延迟
COUNTER_DEFINE(decode_graph_latency_ms);          // Decode Graph 延迟
COUNTER_DEFINE(mixed_eager_latency_ms);           // ← MIXED Eager 延迟

COUNTER_DEFINE(npu_memory_peak_mb);               // NPU 内存峰值
```

**监控代码位置**：`ge_graph_executor_impl.cpp`

```cpp
ModelOutput GeGraphExecutorImpl::run(...) {
  
  switch (params.batch_forward_type.value()) {
    
    case BatchForwardType::PREFILL:
    case BatchForwardType::CHUNKED_PREFILL:
      COUNTER_INC(num_prefill_graph_execution);
      auto start = Timer::now();
      auto result = run_prefill_graph(...);
      COUNTER_SET(prefill_graph_latency_ms, Timer::elapsed_ms(start));
      return result;
    
    case BatchForwardType::DECODE:
      COUNTER_INC(num_decode_graph_execution);
      auto start = Timer::now();
      auto result = run_decode_graph(...);
      COUNTER_SET(decode_graph_latency_ms, Timer::elapsed_ms(start));
      return result;
    
    case BatchForwardType::MIXED:
      COUNTER_INC(num_mixed_eager_execution);  // ← 监控 MIXED 频率
      auto start = Timer::now();
      auto result = model_->forward(...);
      COUNTER_SET(mixed_eager_latency_ms, Timer::elapsed_ms(start));
      return result;
  }
}
```

---

#### 报警规则

| 监控项 | 正常范围 | WARNING阈值 | ERROR阈值 | 行动 |
|-------|---------|------------|----------|------|
| **MIXED 频率** | < 10% | > 20% | > 30% | 调整配置 |
| **MIXED latency** | < Prefill latency | > 2× Prefill | > 3× Prefill | 性能优化 |
| **内存峰值** | < 70% | > 80% | > 90% | 内存优化 |
| **输出误差** | < 1% | > 1% | > 5% | 停止服务 |

---

### 4.3 长期优化计划（可选）

#### 优化方向 1：实现 MIXED Graph（复杂，不推荐）

**技术方案**：
- 分拆 batch（Prefill tokens + Decode tokens）
- 分别执行 Prefill Graph + Decode Graph
- 合并输出

**工作量**：
- 需要实现 batch 分拆逻辑（复杂）
- 需要实现 KV Cache 分拆处理（复杂）
- 需要实现输出合并逻辑（复杂）

**建议**：
- ⚠️ 只在 MIXED 频率极高（> 30%）时考虑
- ⚠️ 工作量大，收益有限（MIXED 频率低）

---

#### 优化方向 2：优化 MIXED Eager 性能（推荐）

**优化方法**：
1. PyTorch JIT compilation（torch.compile）
2. 算子优化（优化 PyTorch forward）
3. 内存优化（减少内存峰值）

**预期收益**：
- 性能提升：10-20%
- 工作量：中等
- 风险：低

---

#### 优化方向 3：调整 Scheduler 配置（推荐）

**配置调整**：
```cpp
// 减少 MIXED 频率的配置
FLAGS_max_chunk_tokens = 128;        // 减小 chunk size
FLAGS_max_tokens_per_batch = 2048;   // 减小 batch tokens
FLAGS_enable_chunked_prefill = false; // 禁用 chunked prefill（彻底避免 MIXED）
```

**预期效果**：
- 禁用 chunked prefill → 无 MIXED batch
- 性能影响：长 prompt 吞吐量下降（但无 MIXED 风险）

---

## 五、最终方案推荐

### 5.1 方案对比总结

| 维度 | 方案 A（Eager fallback） | 方案 B（Prefill Graph） |
|-----|------------------------|----------------------|
| **技术可行性** | ✅ 完全可行 | ❌ 完全不可行 |
| **正确性** | ✅ 保证正确 | ❌ 推理结果错误 |
| **性能** | ⚠️ MIXED 有性能损失（可接受） | ❌ 无性能优化（且错误） |
| **风险** | 中风险（可控） | 高风险（不可控） |
| **工作量** | 低（Eager 已存在） | 高（无法实现） |
| **推荐指数** | ⭐⭐⭐⭐⭐ | ❌ 不推荐 |

---

### 5.2 最终设计决策

**采用方案 A：双图 + Eager fallback**

```
执行路径：
  PREFILL          → Prefill Graph（-1）
  CHUNKED_PREFILL  → Prefill Graph（-1）
  DECODE           → Decode Graph（dynamicDims）
  MIXED            → Eager Mode（fallback）← 推荐
  EMPTY            → 直接返回
```

**关键决策理由**：
1. ✅ **技术可行性**：Eager mode 已存在，无需额外实现
2. ✅ **正确性保证**：Eager mode 可正确处理 MIXED
3. ⚠️ **性能可接受**：MIXED 频率低（10%）→ 性能损失 < 2%
4. ✅ **风险可控**：通过验证和监控降低风险

---

### 5.3 关键行动清单

#### 上线前必须完成（P0）

- [ ] 验证 Prefill Graph vs Prefill Eager 输出一致性
- [ ] 验证 Decode Graph vs Decode Eager 输出一致性
- [ ] 测试 MIXED Eager 功能正确性
- [ ] Memory profiling（验证峰值 < 80% NPU memory）

#### 上线后持续监控（P1）

- [ ] 监控 MIXED 频率（如果 > 20% → WARNING）
- [ ] 监控 MIXED latency（如果 > 2× Prefill → WARNING）
- [ ] 监控内存峰值（如果 > 80% → WARNING）
- [ ] 提供配置指南（推荐 chunked prefill 参数）

---

## 六、代码位置索引

| 功能 | 文件位置 | 关键行号 |
|-----|---------|---------|
| **MIXED 定义** | `batch_forward_type.h` | 33 |
| **MIXED 生成逻辑** | `batch.cpp` | 93-119 |
| **is_prefill 设置** | `attention_metadata_builder.cpp` | 101 |
| **MLU执行路径选择** | `mlu_graph_executor_impl.cpp` | 235-248 |
| **DS2执行路径选择** | `acl_graph_executor_impl.cpp` | 993-1003 |
| **Attention is_prefill 分支** | `deepseek_v4_attention.cpp` | 599-603, 643-663 |
| **KV cache 写入差异** | `deepseek_v4_attention.cpp` | 416-468 |
| **Scheduler 调度** | `chunked_prefill_scheduler.cpp` | 72-146 |

---

## 七、总结

**关键结论**：
1. ✅ **方案 A（Eager fallback）是唯一可行方案**
2. ❌ **方案 B（Prefill Graph）完全不可行**（技术限制无法解决）
3. ⚠️ **MIXED 风险可控**（通过验证 + 监控 + 配置）
4. ✅ **工作量低**（Eager mode 已存在，无需额外实现）

**核心风险**：
- **方案 A**：性能损失风险（可控，MIXED 频率低）
- **方案 B**：推理结果错误风险（不可控，技术限制）

**推荐方案**：
- 双图方案（Prefill Graph + Decode Graph）
- MIXED 使用 Eager fallback
- 完成 P0 验证 + P1 监控

---

**文档结束**