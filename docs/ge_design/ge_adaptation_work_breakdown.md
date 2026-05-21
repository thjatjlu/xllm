# DeepSeek V4 GE ES 适配 — 工作分解、工作量评估与优先级

> 版本：v1.0
> 日期：2026-05-26
> 基于设计规格：`ge_dual_graph_spec.md` v2.0

---

## 一、当前状态

| 类别 | 状态 | 说明 |
|------|------|------|
| 设计文档 | ✅ 完成 | `ge_dual_graph_spec.md`（877行，架构/双图/权重/ES可行性全覆盖） |
| 代码骨架 | 🔶 TODO | 6个文件，类声明已有，实现全部注释/空函数 |
| CMake | 🔶 注释 | `npu_ge/CMakeLists.txt` 全注释，顶层无 `USE_GE` option |
| ES wrapper | ❌ 未开始 | 0个DS4专用wrapper |
| Graph 构建 | ❌ 未开始 | `build_model_graph()` 全注释 |
| 编译执行 | ❌ 未开始 | `compile/run` 全注释 |

---

## 二、适配范围

### 2.1 支持的 ForwardType

| ForwardType | 处理方式 | 说明 |
|-------------|----------|------|
| **DECODE** | Decode Graph（固定 batch=1，不用 dynamicDims） | 第一个打通的场景 |
| **PREFILL** | Prefill Graph（-1 动态 shape） | 与 Decode 共用同一张 ES 构建的图，编译两次 |
| **CHUNKED_PREFILL** | Prefill Graph（同上） | |
| **MIXED** | **报错拒绝** | `CHECK(false)`，要求 Scheduler 分离 prefill/decode |
| **EMPTY** | skip | 直接返回 |

### 2.2 核心简化决策

| 项目 | 最终设计 | 调试阶段简化 | 原因 |
|------|----------|-------------|------|
| **Decode batch** | dynamicDims {1,4,8,16,32,64,128} | **写死 batch=1** | 先打通单 batch，后续加 dynamicDims |
| **Prefill shape** | -1 完全动态 | **-1 完全动态**（不简化） | Prefill 输入长度天然可变，必须动态 |
| **frozen_parameter** | true + frozenInput ~866位 | **先不设**，每次传全部权重 | frozenInput 是性能优化，不影响功能 |
| **Eager fallback** | 无 | **无** | 不做 eager 保底，纯 Graph |
| **MIXED 处理** | Scheduler 分离 | **报错拒绝** | 理论bug，不在当前范围 |

### 2.3 架构核心

```
┌─────────────────────────────────────────────┐
│  ES 构图（一次）                               │
│  build_model_graph()                         │
│  ├─ 创建 ~866 个 Data 输入 placeholder（保序） │
│  ├─ Embedding → HC expand → 43 Layers →     │
│  │  HC Head → Norm → LM Head                │
│  └─ BuildAndReset() → model_graph_           │
└─────────────────────────────────────────────┘
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
  ┌──────────────┐  ┌──────────────┐
  │ Prefill Graph │  │ Decode Graph │
  │ compile       │  │ compile      │
  │ -1 动态shape  │  │ batch=1 固定 │
  │ ≈5min         │  │ ≈5min        │
  └──────────────┘  └──────────────┘
```

---

## 三、工作分解（6个Phase）

### P0: 环境搭建与 GE ES API 验证

| # | 工作项 | 详情 | 工作量 | 验收标准 |
|---|--------|------|--------|----------|
| P0-1 | CMake 集成 | 顶层 `option(USE_GE)` + `find_package(GE)` + runtime/models/layers 条件编译 | 1d | `USE_GE=ON` 可编译 |
| P0-2 | 编译验证 | GE ES 头文件 + `EsGraphBuilder` + `CompliantNodeBuilder` 可编译通过 | 0.5d | 空壳编译无报错 |
| P0-3 | ES Demo 运行 | 编译运行 `ge/examples/es/transformer`，验证 Session → AddGraph → CompileGraph → RunGraph 全链路 | 1d | Demo 输出正确 |
| P0-4 | 单算子验证 | 用 `CompliantNodeBuilder` 构建一个 `HcPre` 算子节点，编译运行验证 | 2d | 单算子推理正确 |

**Phase 总计**: **~4.5d** | **依赖**: GE库安装、NPU驱动 | **关键产出**: GE ES API 可用性确认

---

### P1: Runtime 基础设施

| # | 工作项 | 详情 | 工作量 | 验收标准 |
|---|--------|------|--------|----------|
| P1-1 | GeSessionManager | singleton + `get_or_create_session()` + `next_graph_id()` + Session options | 2d | 可创建/获取 Session |
| P1-2 | GeTensorConverter | `torch_to_ge()` 零拷贝 + `ge_to_torch()` + `map_dtype()` + `infer_format()` | 2d | dtype/shape 映射正确 |
| P1-3 | GeGraphExecutorImpl | `run()`：DECODE→Decode Graph, PREFILL→Prefill Graph, MIXED→CHECK(false), EMPTY→skip | 1d | ForwardType 路径正确 |
| P1-4 | REGISTER_EXECUTOR | `npu_ge` executor 注册，框架可加载 | 0.5d | `--backend npu_ge` 启动成功 |

**Phase 总计**: **~5.5d** | **依赖**: P0 | **关键产出**: ExecutorImpl 可框架加载

---

### P2: DS4 Model 权重基础设施

| # | 工作项 | 详情 | 工作量 | 验收标准 |
|---|--------|------|--------|----------|
| P2-1 | DeepseekV4GeDecoderLayer 权重类 | 定义子模块（hc_attn_pre/post, attn, hc_moe_pre/post, moe）+ `get_all_weights()` 保序收集，不实现 forward() | 2d | 单层权重收集数量和顺序正确 |
| P2-2 | Model weight collection | `collect_all_weights()`：Embedding → 43×~20 → Norm → LM Head → HC Head（~866个，保序） | 1d | 全量权重收集 ~866 个，顺序一致 |
| P2-3 | load_state_dict() | 权重加载到成员变量（frozen_parameter 要求地址不变） | 1d | 权重 tensor dtype/shape 加载正确 |
| P2-4 | Rolling Load 报错 | `free_weights/reload_weights` → CHECK(false) | 0d | 已实现 |

**Phase 总计**: **~4d** | **依赖**: P1 | **关键产出**: ~866 个权重保序可收集

---

### P3: ES Wrapper 开发（核心瓶颈）

每个 wrapper 实现模式（参考 `ge/examples/custom_es_api/custom/my_Conv2D.cpp`）：

```
1. 查询 GetRegisteredIrDef("OpType") → 确认 IR 定义存在
2. 参照对应 TorchAir converter → 确认 inputs/attrs/outputs 规范
3. CompliantNodeBuilder(ge_graph).OpType("...")
     .Name(builder.GenerateNodeName("..."))
     .IrDefInputsV2({...})   → 定义输入
     .IrDefOutputsV2({...})  → 定义输出
     .IrDefAttrsV2({...})    → 定义属性
     .Build()
4. AddEdgeAndUpdatePeerDesc(...) → 连边
5. node.SetAttr(...)           → 设置算子属性
6. builder.GetTensorHolderFromNode(std::move(node), output_index) → 返回 EsTensorHolder
```

TorchAir converter 参考：`cann-recipes-infer/ops/ascendc/torch_ops_extension/custom_ops/converter/*.py`（23个文件）

| # | 算子 | GE Op Type | 输入数 | 输出数 | Attr数 | 出现频率 | 工作量 | 优先级 | TorchAir 参考 |
|---|------|-----------|--------|--------|--------|----------|--------|--------|--------------|
| P3-1 | **HcPre** | `HcPre` | 4 | 3 | 4 | 每层2×=86 | 1.5d | **P0** | `npu_hc_pre.py` |
| P3-2 | **HcPost** | `HcPost` | 2-3 | 1-2 | 2 | 每层2×=86 | 1d | **P0** | `npu_hc_post.py` |
| P3-3 | **RMSNorm** | `npu_rms_norm` | 2 | 1 | 2 | 每层3×+1 | 1d | **P0** | torch_npu内置 |
| P3-4 | **InplacePartialRotaryMul** | `InplacePartialRotaryMul` | 3-5 | 1 | 3 | 每层3-5× | 2d | **P0** | `npu_inplace_partial_rotary_mul.py` |
| P3-5 | **SparseAttnSharedkv** | `SparseAttnSharedkv` | 12 | 2 | 9 | 每层1×=43 | **2.5d** | **P0** | `npu_sparse_attn_sharedkv.py` |
| P3-6 | **SparseAttnSharedkvMetadata** | `SparseAttnSharedkvMetadata` | 5-8 | 1 | 5 | 每步 | 1.5d | **P0** | `npu_sparse_attn_sharedkv_metadata.py` |
| P3-7 | Compressor | `Compressor` | 3-4 | 1 | 2-3 | C4A/C128A层 | 1d | **P1** | `npu_compressor.py` |
| P3-8 | QuantLightningIndexer | `QuantLightningIndexer` | 3-4 | 1 | 2-3 | C4A层 | 1d | **P1** | `npu_quant_lightning_indexer.py` |
| P3-9 | QuantLightningIndexerMetadata | `QuantLightningIndexerMetadata` | 3-5 | 1 | 3 | 每步 | 1d | **P1** | `npu_quant_lightning_indexer_metadata.py` |
| P3-10 | MoeGatingTopKHash | `MoeGatingTopKHash` | 2-3 | 2-3 | 3 | 每层1×=43 | 1d | **P1** | `npu_moe_gating_top_k.py` |
| P3-11 | GroupedMatMul | `npu_grouped_matmul` | 3-4 | 1 | 3-4 | 每层2×=86 | 1.5d | **P1** | torch_npu内置 |
| P3-12 | MC2 Dispatch | `npu_moe_distribute_dispatch_v2` | 4-5 | 2-3 | 3 | 每层1×=43 | 1.5d | **P1** | `npu_moe_init_routing_group_quant.py` |
| P3-13 | MC2 Combine | `npu_moe_distribute_combine_v2` | 3-4 | 1-2 | 3 | 每层1×=43 | 1.5d | **P1** | 同上 |
| P3-14 | TransposeBatchMatMul | `npu_transpose_batchmatmul` | 2 | 1 | 2 | 每层1×=43 | 0.5d | **P1** | torch_npu内置 |
| P3-15 | Embedding | `aten`→GE | 2 | 1 | 0 | 1× | 0.5d | **P0** | torch_npu内置 |

**bf16 不需要的算子**（量化模式才需要）：
- QuantMatMul, RmsNormDynamicQuant, DequantSwigluClampQuant
- KvCompressEpilog（float8 only）
- ScatterNdUpdateAsc（无GE converter）

**Phase 总计**: **~18d**（1人） / **~12d**（2人并行） | **依赖**: P0-4 验证 | **关键产出**: 15个bf16 ES wrapper 可编译

---

### P4: Graph 构建

| # | 工作项 | 详情 | 工作量 | 验收标准 |
|---|--------|------|--------|----------|
| P4-1 | 输入 placeholder 创建 | ~866个 Data 输入，保序：input_ids(0) + embed(1) + 43×~20(2-862) + norm(863) + lm_head(864) + hc_head(865-867) + KV/metadata(868+) | 2d | placeholder 数量/顺序/dtype/shape 正确 |
| P4-3 | Embedding → HC expansion | `Embedding(input_ids, embed_weight)` → `Reshape(hidden, {-1, hc_mult, hidden})` | 0.5d | Embedding 输出 shape 正确 |
| P4-4 | 单层 Decoder 构建 | **最复杂项**：HcPre → RMSNorm → MLA(wq_a, wkv, wq_b matmuls + RoPE) → Compressor → Indexer → SparseAttn → Derotary → wo_a TransposeBatchMatMul → wo_b → HcPost + HcPre → RMSNorm → MoE(gate topk, MC2 dispatch, grouped matmul w13+w2+swiglu, MC2 combine, shared expert) → HcPost | **5d** | 单层 Graph dump 可验证算子链路 |
| P4-5 | 43层循环 | `for (layer=0; layer<43)` 调用 P4-4，每层权重 slice 不同 | 1d | 43层 Graph 构建无报错 |
| P4-6 | HC Head → Norm → LM Head | HyperConnectionHead + RMSNorm + MatMul | 0.5d | LM Head 输出 logits shape 正确 |
| P4-7 | KV Cache 输入/输出 | KV Cache 作为 Data 输入+输出（frozen=0），每层 k_cache/v_cache pair | 1.5d | KV Cache placeholder 正确 |
| P4-8 | Metadata 输入 | SparseAttnSharedkvMetadata + QuantLightningIndexerMetadata 作为 Data 输入（frozen=0） | 1d | Metadata placeholder 正确 |
| P4-9 | BuildAndReset | 收集所有输出 → `builder->BuildAndReset({logits, kv_cache_outputs...})` | 0.5d | `ge::Graph` 对象成功创建 |

**Phase 总计**: **~11d** | **依赖**: P3 全部 wrapper | **关键产出**: 完整 DS4 Graph 对象

---

### P5: 双图编译与执行

| # | 工作项 | 详情 | 工作量 | 验收标准 |
|---|--------|------|--------|----------|
| P5-1 | Prefill Graph 编译 | 同一张 graph，`AddGraph(prefill_id, *model_graph_, options)` + `CompileGraph(prefill_id)`，options 含 `ge.inputShape: "-1"`（-1 动态 shape） | 1.5d | CompileGraph SUCCESS，耗时 <10min |
| P5-2 | Decode Graph 编译 | 同一张 graph，`AddGraph(decode_id, *model_graph_, options)` + `CompileGraph(decode_id)`，options 含固定 batch=1（无 dynamicDims） | 1d | CompileGraph SUCCESS，耗时 <10min |
| P5-3 | RunGraph 执行 | `session_->RunGraph(graph_id, inputs, outputs)`，同步执行 | 1d | RunGraph SUCCESS |
| P5-4 | 输入组装（保序） | `collect_all_weights()` (~866个) + KV Cache + metadata → `torch_to_ge()` → 保序 vector，顺序与 placeholder 一致 | 1.5d | 输入顺序与 Graph placeholder 完全一致 |
| P5-5 | 输出转换 | `ge::Tensor → torch::Tensor → ModelOutput`（logits） | 1d | logits dtype/shape 正确 |
| P5-6 | ForwardType 路径选择 | PREFILL/CHUNKED_PREFILL→Prefill Graph, DECODE→Decode Graph, MIXED→CHECK(false), EMPTY→skip | 0.5d | 路径切换正确 |

**Phase 总计**: **~5.5d** | **依赖**: P4 | **关键产出**: 双图可编译执行

---

### P6: 测试验证

| # | 工作项 | 详情 | 工作量 | 验收标准 |
|---|--------|------|--------|----------|
| P6-1 | 单层 Decode 测试 | 1层 + Embedding + Norm + LM Head，batch=1 固定shape，单token推理 | 2d | logits 数值与 eager/Python 对齐 |
| P6-2 | 完整 43层 Decode 测试 | 完整模型 batch=1 Decode 单token | 2d | logits 数值正确 |
| P6-3 | Prefill Graph 测试 | 多token Prefill 推理 | 1.5d | Prefill 输出正确 |
| P6-4 | 双图切换测试 | Prefill→Decode 连续推理 | 0.5d | 切换无报错，结果一致 |

**Phase 总计**: **~6d** | **依赖**: P5 | **关键产出**: 端到端推理正确性

---

## 四、总工作量汇总

| Phase | 1人 | 2人并行 | 依赖 | 关键产出 |
|-------|-----|---------|------|----------|
| P0: 环境验证 | 4.5d | 3d | GE库 | API可用性确认 |
| P1: Runtime | 5.5d | 4d | P0 | Executor可加载 |
| P2: Model权重 | 4d | 3d | P1 | ~866权重保序可收集 |
| P3: ES Wrapper | 18d | 12d | P0-4 | 15个wrapper可编译 |
| P4: Graph构建 | 11d | 11d(不可并行) | P3 | 完整DS4 Graph |
| P5: 编译执行 | 5.5d | 4d | P4 | 双图可运行 |
| P6: 测试 | 6d | 4d | P5 | 端到端正确 |
| **总计** | **~49d** | **~31d** | | |

---

## 五、关键路径与并行策略

```
P0(4.5d) → P1(5.5d) → P2(4d) → P3(18d) → P4(11d) → P5(5.5d) → P6(6d)
                                    ↑
                            关键瓶颈：ES Wrapper（18d）
```

**推荐2人并行分工**：
- **人A**（架构侧）：P0 → P1 → P2 → P4 → P5 → P6
- **人B**（wrapper侧）：P0-4验证 → P3全部15个wrapper
- 人A 在等 P3 期间可先推进 P0/P1/P2
- P4 需等 P3 完成，人A集中做 Graph 构建

---

## 六、分步 Milestone

| Milestone | 目标 | 包含工作 | 1人/2人预估 |
|-----------|------|----------|-------------|
| **M1: 单算子通** | 1个ES wrapper可编译运行 | P0 + P3-1(HcPre) | ~6d / ~4d |
| **M2: 单层Decode通** | 1层+Embedding+Norm+LM Head，batch=1单token正确推理 | P1 + P2 + P3(核心6个P0) + P4(单层) + P5 + P6-1 | ~26d / ~18d |
| **M3: 43层全通** | 完整模型 Prefill+Decode 双图正确推理 | P3(P1 8个MoE) + P4(43层) + P5 + P6 | ~20d / ~13d |
| **总计到 M3** | | | **~49d / ~31d** |

---

## 七、优先级排序

| 优先级 | 工作项 | 原因 |
|--------|--------|------|
| **P0** | P0 环境验证 | 不验证GE API可用性，后续全部白费 |
| **P0** | P3-1~6 核心Attention wrapper | Attention链路：HcPre/Post+RMSNorm+RoPE+SparseAttn+Metadata，缺一不可 |
| **P0** | P4-4 单层构建 | 最复杂单项，单层正确才能43层 |
| **P0** | P5 编译执行 | 无此步无法验证推理 |
| **P1** | P3-7~14 MoE wrapper | MoE链路8个wrapper，缺则FFN不可用 |
| **P1** | P1-P2 Runtime+权重 | 基础设施，可与P3并行 |

---

## 八、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| CompliantNodeBuilder 不支持某些 DS4 op IR 定义 | 中 | 阻塞P3 | P0-4先验证关键算子（SparseAttnSharedkv） |
| CompileGraph 失败或 >10min | 中 | 阻塞P5 | 先验证1层编译；降低算子数量 |
| 权重输入顺序不一致 | 中 | 推理结果错误 | P4-1 + P5-4 单元测试保序 |
| 43层Graph构建内存溢出 | 低 | 编译失败 | 先验证1层→逐步递增 |
| KV Cache 输入输出处理错误 | 中 | Attention计算错误 | 单层测试先验证KV Cache |

---

## 九、打通后的升级路径

| # | 升级项 | 工作量 | 优先级 | 说明 |
|---|--------|--------|--------|------|
| 1 | frozen_parameter=true + frozenInput | 2d | **高** | 权重地址不变→缩短Graph下发时间 |
| 2 | Decode dynamicDims {1,4,8,16,32,64,128} | 3d | **高** | 支持多batch Decode |
| 3 | RunGraphWithStreamAsync | 1d | 中 | 异步执行，与CUDA stream对齐 |
| 4 | 性能基准 + frozen效果验证 | 2d | 中 | Decode Graph vs eager性能对比 |
| 5 | 量化优化（w8a8int8 NZ + ConstPlaceHolder + QuantMatMul） | 27-34d | 低 | 大幅性能提升，独立Phase |

**从"单batch单图"升级到"完整双图+dynamicDims+frozen_parameter"仅需 ~6d。**

---

## 十、代码文件规划

### 新增文件

```
xllm/core/runtime/
  ├── ge_graph_executor_impl.h/.cpp   # Executor（已有骨架，需实现）
  ├── ge_tensor_converter.h/.cpp      # Tensor转换（已有骨架，需实现）
  ├── ge_session_manager.h/.cpp       # Session管理（已有骨架，需实现）

xllm/models/llm/npu_ge/
  ├── deepseek_v4.h/.cpp              # Model（已有骨架，需实现）
  ├── deepseek_v4_decoder_layer.h/.cpp  # Layer权重类（需新建）
  └── CMakeLists.txt                  # 编译配置（需激活）

xllm/core/layers/npu_ge/
  ├── es_wrapper/                     # ES wrapper目录（需新建）
  │   ├── hc_pre.h/.cpp              # HcPre wrapper
  │   ├── hc_post.h/.cpp             # HcPost wrapper
  │   ├── rms_norm.h/.cpp            # RMSNorm wrapper
  │   ├── inplace_partial_rotary_mul.h/.cpp  # RoPE wrapper
  │   ├── sparse_attn_sharedkv.h/.cpp        # SparseAttn wrapper
  │   ├── sparse_attn_sharedkv_metadata.h/.cpp  # Metadata wrapper
  │   ├── compressor.h/.cpp          # Compressor wrapper
  │   ├── quant_lightning_indexer.h/.cpp      # Indexer wrapper
  │   ├── quant_lightning_indexer_metadata.h/.cpp  # Indexer Metadata
  │   ├── moe_gating_topk_hash.h/.cpp        # MoE gate wrapper
  │   ├── grouped_matmul.h/.cpp      # GroupedMatMul wrapper
  │   ├── mc2_dispatch.h/.cpp        # MC2 Dispatch wrapper
  │   ├── mc2_combine.h/.cpp         # MC2 Combine wrapper
  │   ├── transpose_batchmatmul.h/.cpp  # TransposeBMM wrapper
  │   ├── embedding.h/.cpp           # Embedding wrapper
  │   └── CMakeLists.txt
  └── CMakeLists.txt

xllm/CMakeLists.txt                   # 添加 USE_GE option（需修改）
```

### 需修改的现有文件

```
xllm/core/runtime/CMakeLists.txt      # 添加 GE sources 条件编译
xllm/models/llm/CMakeLists.txt        # 添加 npu_ge subdirectory
xllm/core/layers/CMakeLists.txt       # 添加 npu_ge subdirectory
```

---

**文档结束**