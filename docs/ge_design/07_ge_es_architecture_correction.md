# GE ES 架构理解修正说明

> 更新日期：2026-05-21
> 原因：基于 DS4 Python 实现和 GE Demo 的实际用法，修正架构理解

---

## 核心修正点

### 修正 1：Graph 包含完整推理流程（而非仅 43 Layers）

**错误理解**：
- Decode Graph 只包含中间的 43 DecoderLayers
- Graph 输入是 hidden_states（embedding 之后）

**正确理解**：
- Decode Graph 包含完整推理流程（6 步）
  - Step 1: Embedding lookup（tokens → hidden_states）
  - Step 2: HyperConnection expansion
  - Step 3: 43 DecoderLayers
  - Step 4: HyperConnection Head
  - Step 5: Final Norm
  - Step 6: LM Head（hidden_states → logits）
- **Graph 输入是 input_ids（原始 tokens）**

**证据**：
- DS4 Python 实现：`modeling_deepseek.py:2011-2019`
  - `main_decode()` 函数调用 `forward()`
  - `forward()` 包含 Embedding → 43 Layers → HC Head → Norm → LM Head
  - Graph 输入：`input_ids`（原始 tokens）

---

### 修正 2：一个 EsGraphBuilder 构建整个 Model（而非分层构建）

**错误理解**：
- 每个 Layer 有独立的 Operation
- 需要分层创建多个 Builder
- 类似 ATB 的逐层构建方式

**正确理解**：
- **一个 EsGraphBuilder 构建整个 model**
- **直接调用算子 API**：`Embedding()`, `MatMul()`, `RMSNorm()` 等
- **不分层构建**：没有独立的 Layer Operation
- **Layer Builder 只是辅助函数**（代码组织，非架构必需）

**证据**：
- GE Demo：`make_transformer_graph.cpp:50-86`
  - `MakeTransformerGraphByEs()` 直接调用算子
  - 没有 Layer Builder 或 Operation 的概念
  - 所有算子在同一个 builder 上构建

---

## 更新的文档

### 1. 总结文档

**文件**：`00_ge_ds4_code_implementation_summary.md`

**更新内容**：
- ✅ 添加"架构澄清：GE ES 构建方式"章节
- ✅ 明确 Graph 包含完整推理流程（6 步）
- ✅ 明确 Graph 输入是 input_ids
- ✅ 添加与 ATB 的架构对比表
- ✅ 添加 DS4 Python 参考文档位置
- ✅ 澄清 Layer Builder 的定位（辅助函数）

### 2. 代码框架文档

**文件**：`02_ge_ds4_code_framework.md`

**更新内容**：
- ✅ 修正 `build_decode_graph()` 示例代码
- ✅ 从 `hidden_states` 输入改为 `input_ids` 输入
- ✅ 添加 Embedding、HyperConnection expansion 步骤
- ✅ 添加 HyperConnection Head、Final Norm、LM Head 步骤

### 3. 代码初稿

**文件**：`xllm/models/llm/npu_ge/deepseek_v4.cpp`

**更新内容**：
- ✅ 修正 `build_decode_graph()` TODO 注释
- ✅ 明确包含完整推理流程（6 步）
- ✅ 输入改为 `input_ids`
- ✅ 添加注释说明架构特征

**架构简化**：
- ❌ 删除 `DeepseekV4DecoderLayerBuilder` 类（不必要）
- ✅ 直接在 `build_model_graph()` 中构建算子

---

## 关键架构对比

| 维度 | ATB (DS2) | GE ES (DS4) |
|------|-----------|-------------|
| **构建模式** | 逐层构建 Operation | 一次性构建整个图 |
| **Graph 组成** | Operation 组合 | 直接包含所有算子 |
| **Graph 输入** | hidden_states | **input_ids** |
| **Graph 包含** | 只有中间 layers | **完整推理流程** |
| **Layer 抽象** | 独立 Operation | **没有 Layer Operation** |

---

## 参考文档位置

### DS4 Python 实现（torch.compile）

**路径**：
`/home/lianghao/thj/code/cann-recipes-infer/models/deepseek-v4/models/modeling_deepseek.py`

**关键代码**：
- Graph 编译：`get_cached_graph()`（line 1736-1760）
- Decode 主函数：`main_decode()`（line 2011-2019）
- Model forward：`DeepseekV3Model.forward()`（line 1467-1552）

### GE Demo（ES API）

**路径**：
`/home/lianghao/thj/code/ge/examples/es/transformer/cpp/src/make_transformer_graph.cpp`

**关键代码**：
- Graph 构建：`MakeTransformerGraphByEs()`（line 50-86）

---

## 后续工作

### 已完成
- ✅ 更新所有设计文档
- ✅ 更新代码初稿注释
- ✅ 添加架构澄清章节

### 待完成
- ❌ 实现 build_decode_graph()（需要 GE ES API）
- ❌ 配置 GE 库依赖（需要 GE SDK 环境）
- ❌ 测试完整推理流程的正确性

---

**修正说明结束**