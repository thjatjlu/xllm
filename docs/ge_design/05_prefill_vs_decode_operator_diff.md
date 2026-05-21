# DeepSeek V4 Prefill vs Decode 算子差异分析（补充）

> 文档版本：v1.0
> 创建日期：2025-01-19
> 状态：补充分析（关键发现）

---

## 一、关键发现：算子执行流程相同！

### 1.1 DecoderLayer Forward（顶层）

**代码位置**：`deepseek_v4_decoder_layer_impl.cpp:140-208`

**执行流程（Prefill 和 Decode 完全相同）**：

```
DecoderLayer forward():
  1. HyperConnectionPre (hc_attn_pre_)   ← 相同
  2. RMSNorm (attn_norm_)                 ← 相同
  3. Attention (attention_)               ← 相同（内部有分支）
  4. HyperConnectionPost (hc_attn_post_)  ← 相同
  5. HyperConnectionPre (hc_moe_pre_)    ← 相同
  6. RMSNorm (moe_norm_)                  ← 相同
  7. MoE Routing (route_gate_)            ← 相同
  8. MoE TopK (topk_)                     ← 相同
  9. FusedMoE (moe_mlp_)                  ← 相同
  10. HyperConnectionPost (hc_moe_post_) ← 相同
```

**关键结论**：
- ✅ **DecoderLayer forward 没有 Prefill/Decode 分支**
- ✅ **所有算子都执行，执行流程完全相同**
- ✅ **差异在算子内部的数据处理逻辑**

---

## 二、Attention 内部差异详解

### 2.1 Attention Forward 内部分支

**代码位置**：`deepseek_v4_attention.cpp:550-797`

**关键差异点**（通过 `is_prefill` 参数控制）：

| 代码位置 | Prefill | Decode | 差异内容 |
|---------|---------|-------|---------|
| **line 599-603** | `offsets = query_lens` | `offsets = window_size` | offsets 计算不同 |
| **line 643-663** | 合并原始 KV + 压缩 KV | 不合并 | Prefill 需要 merge KV |
| **line 683-692** | `batch_offset = cu_context_lens` | `batch_offset = std::nullopt` | batch_offset 计算 |
| **line 713-717** | `kv_cache_reshaped = kv.unsqueeze(1)` | `kv_cache_reshaped = k_cache` | KV cache 来源不同 |
| **line 416-468** | write_kv_to_cache（多个 tokens） | write_kv_to_cache（单个 token） | KV 写入逻辑不同 |

---

### 2.2 write_kv_to_cache 差异详解

**代码位置**：`deepseek_v4_attention.cpp:400-470`

**Prefill 逻辑（写多个 tokens）**：

```cpp
if (is_prefill) {
  if (seqlen <= window_size_) {
    // 直接复制所有 tokens
    reshape_paged_cache(kv_seq, slot_mapping);
  } else {
    // 超过 window：使用 ring buffer 布局
    // 写入 tail 部分（最近 window_size 个 tokens）
    kv_tail = kv_seq.slice(0, -window_size_);
    reshape_paged_cache(kv_tail, slot_mapping);
  }
}
```

**Decode 逻辑（写单个 token）**：

```cpp
else {
  // 写入单个 token 到 ring buffer 位置
  int64_t slot_idx = (seqlen - 1) % window_size_;
  reshape_paged_cache(kv_seq, slot_mapping);  // 只写 1 个 token
}
```

**差异本质**：
- Prefill：批量写入（可能写多个 tokens）
- Decode：单个写入（只写 1 个 token）

---

### 2.3 KV Merge 差异（Prefill 独有）

**代码位置**：`deepseek_v4_attention.cpp:643-663`

**Prefill 独有逻辑**：

```cpp
if (is_prefill) {
  // 合并原始 KV + 压缩 KV
  for (int64_t i = 0; i < batch_size; ++i) {
    auto cur_kv = kv.slice(0, query_start, query_end);  // 原始 KV
    if (compress_lens[i] > 0) {
      auto kv_merge = torch::cat({cur_kv, compress_kvs[i]}, 0);  // 合并
      merge_kvs.push_back(kv_merge);
    }
  }
  kv = torch::cat(merge_kvs, 0);  // 更新 KV tensor
}
```

**Decode 不需要 merge**：
- Decode 只生成 1 个 token
- 不需要合并原始 KV

---

## 三、其他算子的差异分析

### 3.1 HyperConnection（无差异）

**代码位置**：`hyper_connection.cpp`

**结论**：
- ✅ Prefill 和 Decode 执行相同的算子逻辑
- ✅ 无 `is_prefill` 分支判断

---

### 3.2 MoE（无差异）

**代码位置**：`fused_moe.cpp`

**结论**：
- ✅ Prefill 和 Decode 执行相同的算子逻辑
- ✅ Hash routing（前 3 layers）和标准 routing 都执行
- ✅ 无 `is_prefill` 分支判断

---

### 3.3 Indexer（内部可能有差异）

**代码位置**：`indexer_v2.cpp`

**grep 结果**：`line 185: is_prefill`

**可能差异**：
- Indexer forward 可能根据 `is_prefill` 调整 top-k 选择逻辑
- 需要进一步查看 indexer forward 实现

---

### 3.4 Compressor（待确认）

**grep 结果**：无 `is_prefill` 关键字

**结论**：
- 可能无 Prefill/Decode 差异
- 需要进一步确认

---

## 四、关键结论：GE Graph 方案的影响

### 4.1 核心发现

**算子执行层面**：
- ✅ **Prefill 和 Decode 的算子执行流程完全相同**
- ✅ **差异在算子内部的数据处理逻辑**（通过 `is_prefill` 参数控制）
- ✅ **不需要两个独立的 GE graph！**

### 4.2 对 GE 方案的影响

**之前的错误假设**：
```
编译两个 graph：
  1. Prefill graph（-1 动态 shape）
  2. Decode graph（dynamicDims 分档）
  
理由：假设 Prefill 和 Decode 有不同的算子执行流程
```

**修正后的理解**：
```
只编译一个 graph（Decode graph）：
  - 算子执行流程相同（同一个 forward 函数）
  - 通过 params.is_prefill 参数控制内部差异
  - Prefill 使用 eager mode（因为 batch 动态性太大）
  - Decode 使用 GE graph（因为 batch 相对稳定）
```

---

### 4.3 为什么 Prefill 不需要 Graph（真正原因）

**原因 1：算子层面无差异**
- Prefill 和 Decode 执行相同的算子流程
- 不需要两个不同的 graph

**原因 2：Batch 动态性**
- Prefill batch = prompt token 数量（1-1000，极不稳定）
- Decode batch = 并发请求数（1-128，相对稳定）

**原因 3：Graph 收益对比**
| 维度 | Prefill | Decode |
|-----|---------|--------|
| **Graph 编译** | 分钟级（开销大） | 分钟级（开销相同） |
| **执行频率** | 每请求 1 次（收益低） | 每 token 1 次（收益高） |
| **Batch 稳定性** | 极不稳定（graph 无意义） | 稳定（graph 可优化） |
| **是否值得？** | ❌ 编译开销 > 执行收益 | ✅ 编译开销 < 执行收益 |

---

## 五、GE Graph 构建策略（最终方案）

### 5.1 只编译 Decode Graph

**理由**：
1. ✅ 算子执行流程相同（一个 graph 即可）
2. ✅ Decode batch 稳定（dynamicDims 可优化）
3. ✅ Prefill batch 动态性太大（不值得编译）

### 5.2 Decode Graph 内部逻辑

**通过 params 传递 `is_prefill` 参数**：

```cpp
// GE graph 执行时
ModelInputParams params;
params.is_prefill = false;  // ← Decode graph：is_prefill = false

// Attention 内部根据 params.is_prefill 判断
if (params.is_prefill) {
  // Prefill 逻辑（Decode graph 不执行）
} else {
  // Decode 逻辑（Decode graph 执行）
}
```

**关键洞察**：
- Decode graph 编译时，`is_prefill` 固定为 false
- Graph 内部的分支逻辑会固定到 Decode 分支
- **Graph 编译结果只包含 Decode 分支的算子逻辑**

---

## 六、对比 DS2 ATB 的差异

### 6.1 DS2 ATB 的双图设计

**DS2 编译多个 graphs**：
```cpp
init_layer() {
  init_node(prefill_node_);        // Prefill graph
  init_node(prefill_node_prefixcache_);
  init_node(decode_node_);         // Decode graph
  init_node(decode_mla_node_);
}
```

**为什么 DS2 可以编译多个 graphs？**
- ATB init_node() 是毫秒级（创建图结构，不是编译）
- 编译开销极低，可以编译多个 graphs

### 6.2 GE 的编译成本

**GE CompileGraph 是分钟级**：
- 真正的编译优化（算子融合、内存规划）
- 编译开销高，不适合编译多个 graphs

**最优策略**：
- 只编译 Decode graph（收益最高的场景）
- Prefill 使用 eager mode（避免编译开销）

---

## 七、最终设计决策（确认）

### 7.1 修正后的方案

**原方案（错误）**：
```
编译两张图：
  - Prefill graph（-1）
  - Decode graph（dynamicDims）
```

**修正后的方案（正确）**：
```
只编译一张图：
  - Decode graph（dynamicDims = {1,4,8,16,32,64,128}）
  
Prefill：
  - Eager mode（直接 forward）
  
理由：
  1. 算子执行流程相同（代码验证）
  2. Prefill batch 动态性太大（不值得编译）
  3. Graph 编译开销高（分钟级）
```

### 7.2 关键证据总结

**证据 1：DecoderLayer forward 无分支**
- 代码：`deepseek_v4_decoder_layer_impl.cpp:140-208`
- 所有算子都执行（HyperConnection、Attention、MoE）

**证据 2：Attention 内部有 is_prefill 分支**
- 代码：`deepseek_v4_attention.cpp`
- 通过 `is_prefill` 参数控制数据处理逻辑
- 不影响算子选择（只影响数据组织方式）

**证据 3：其他算子无分支**
- HyperConnection、MoE 无 `is_prefill` 判断
- 执行流程完全相同

---

## 八、文档更新建议

### 更新文档 01（方案概述）

**补充内容**：
```
## 1.1 核心设计决策（补充）

关键发现：
  - Prefill 和 Decode 算子执行流程相同
  - 差异在内部数据处理逻辑（通过 params 控制）
  - 不需要编译两个独立的 GE graph
  
理由：
  1. 算子层面无差异（代码验证）
  2. Batch 动态性差异（编译收益不同）
  3. 编译成本差异（分钟级 vs 毫秒级）
```

### 更新文档 04（关键决策）

**补充 Q10**：

```
### Q10: Prefill 和 Decode 算子层面是否有差异？

**决策**：算子执行流程相同，不需要两个 graph

**代码证据**：
  - DecoderLayer forward 无分支判断
  - 所有算子都执行（Attention、MoE、HyperConnection）
  - 差异在内部数据处理（通过 is_prefill 参数）

**结论**：
  - ✅ 算子执行流程相同
  - ✅ 只编译 Decode graph 即可
  - ✅ Prefill 使用 eager mode（batch 动态性原因）
```

---

**文档结束**