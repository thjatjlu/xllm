# DeepSeek V4 GE 方案 - 代码框架（初稿）

> 文档版本：v1.0
> 创建日期：2025-01-19

---

## 一、核心类设计

### 1.1 GeGraphExecutorImpl（Executor 层）

```cpp
// xllm/core/runtime/ge_graph_executor_impl.h

#pragma once

#include "executor_impl.h"
#include "executor_impl_factory.h"
#include <ge/ge_api.h>
#include <acl/acl.h>

namespace xllm::ge {

class GeGraphExecutorImpl : public ExecutorImpl {
 public:
  GeGraphExecutorImpl(CausalLM* model,
                      const ModelArgs& args,
                      const torch::Device& device,
                      const runtime::Options& options);
  
  ~GeGraphExecutorImpl() override;
  
  ForwardInput prepare_inputs(Batch& batch) override;
  
  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;
  
 private:
  CausalLM* model_;
  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  
  ge::Session* session_;
  uint32_t decode_graph_id_;
  
  // Helper methods
  ModelOutput run_decode_graph(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params);
  
  std::vector<ge::Tensor> convert_inputs(...);
  ModelOutput convert_outputs(const std::vector<ge::Tensor>& ge_outputs);
};

REGISTER_EXECUTOR("npu_ge", GeGraphExecutorImpl);

}  // namespace xllm::ge
```

### 1.2 GeTensorConverter（Tensor 转换工具）

```cpp
// xllm/core/runtime/ge_tensor_converter.h

#pragma once

#include <torch/torch.h>
#include <ge/tensor.h>
#include <ge/graph.h>

namespace xllm::ge {

class GeTensorConverter {
 public:
  // torch::Tensor → ge::Tensor（零拷贝）
  static ge::Tensor torch_to_ge(const torch::Tensor& torch_tensor);
  
  // ge::Tensor → torch::Tensor（零拷贝）
  static torch::Tensor ge_to_torch(const ge::Tensor& ge_tensor);
  
 private:
  // 数据类型映射表
  static ge::DataType map_dtype(torch::ScalarType torch_dtype);
  static torch::ScalarType map_ge_dtype(ge::DataType ge_dtype);
  
  // 格式推断
  static ge::Format infer_format(const std::vector<int64_t>& sizes);
};

}  // namespace xllm::ge
```

---

## 二、核心实现逻辑

### 2.1 Executor::run() 实现

```cpp
// xllm/core/runtime/ge_graph_executor_impl.cpp

ModelOutput GeGraphExecutorImpl::run(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
  
  // Step 1: 判断 ForwardType
  if (!params.batch_forward_type.is_decode()) {
    // PREFILL / CHUNKED_PREFILL / MIXED → eager mode
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  
  // Step 2: DECODE → 使用 decode_graph
  return run_decode_graph(tokens, positions, kv_caches, params);
}

ModelOutput GeGraphExecutorImpl::run_decode_graph(...) {
  // Step 1: 准备 GE inputs
  std::vector<ge::Tensor> ge_inputs = convert_inputs(
      tokens, positions, kv_caches, params);
  
  // Step 2: 获取 NPU stream
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  
  // Step 3: 执行 GE graph
  std::vector<ge::Tensor> ge_outputs;
  ge::Status ret = session_->RunGraphWithStreamAsync(
      decode_graph_id_, stream, ge_inputs, ge_outputs);
  
  if (ret != ge::SUCCESS) {
    LOG(ERROR) << "RunGraph failed, error=" << ret;
    throw std::runtime_error("GE graph execution failed");
  }
  
  // Step 4: 同步 stream
  aclrtSynchronizeStream(stream);
  
  // Step 5: 转换输出
  return convert_outputs(ge_outputs);
}
```

### 2.2 Tensor 转换实现

```cpp
// xllm/core/runtime/ge_tensor_converter.cpp

ge::Tensor GeTensorConverter::torch_to_ge(const torch::Tensor& torch_tensor) {
  // 1. 提取属性
  auto sizes = torch_tensor.sizes().vec();
  auto dtype = torch_tensor.scalar_type();
  auto data_ptr = torch_tensor.data_ptr();
  
  // 2. 映射数据类型
  ge::DataType ge_dtype = map_dtype(dtype);
  ge::Format format = infer_format(sizes);
  
  // 3. 创建 TensorDesc
  ge::Shape shape(sizes);
  ge::TensorDesc desc(shape, format, ge_dtype);
  
  // 4. 创建 ge::Tensor（零拷贝，直接引用 NPU memory）
  ge::Tensor ge_tensor;
  ge_tensor.SetTensorDesc(desc);
  ge_tensor.SetData(
      reinterpret_cast<uint8_t*>(data_ptr),
      torch_tensor.numel() * torch_tensor.element_size());
  
  return ge_tensor;
}

ge::DataType GeTensorConverter::map_dtype(torch::ScalarType torch_dtype) {
  switch (torch_dtype) {
    case torch::kFloat32: return ge::DT_FLOAT;
    case torch::kFloat16: return ge::DT_FLOAT16;
    case torch::kBFloat16: return ge::DT_BF16;
    case torch::kInt32: return ge::DT_INT32;
    case torch::kInt64: return ge::DT_INT64;
    default:
      CHECK(false) << "Unsupported dtype: " << torch_dtype;
      return ge::DT_UNDEFINED;
  }
}
```

---

## 三、Model 层实现

### 3.1 Model 定义

```cpp
// xllm/models/llm/npu_ge/deepseek_v4.h

#pragma once

#include "models/llm/llm_model_base.h"
#include <ge/ge_api.h>

namespace xllm::models {

class DeepseekV4GeModelImpl : public torch::nn::Module {
 public:
  DeepseekV4GeModelImpl(const ModelContext& context);
  
  void load_state_dict(const StateDict& state_dict);
  void init_model();
  
  torch::Tensor forward(torch::Tensor& tokens,
                        torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params);
  
 private:
  // Model components
  torch::nn::Embedding embed_tokens_{nullptr};
  std::vector<DeepseekV4GeDecoderLayer> layers_;
  RMSNorm norm_{nullptr};
  torch::nn::Linear lm_head_{nullptr};
  
  // GE graph management
  ge::Session* session_;
  uint32_t decode_graph_id_;
  
  void build_decode_graph();
  void bind_weights_to_graph(ge::Graph& graph);
};

TORCH_MODULE(DeepseekV4GeModel);

class DeepseekV4GeForCausalLMImpl 
    : public LlmForCausalLMImplBase<DeepseekV4GeModel> {
  // 继承基类的 forward 实现
  // 只需实现特定的权重加载方法
};

TORCH_MODULE(DeepseekV4GeForCausalLM);

REGISTER_CAUSAL_MODEL(deepseek_v4_npu_ge, DeepseekV4GeForCausalLM);

}  // namespace xllm::models
```

### 3.2 init_model() 实现

```cpp
// xllm/models/llm/npu_ge/deepseek_v4.cpp

void DeepseekV4GeModelImpl::init_model() {
  // Step 1: 初始化 GE Session
  std::map<ge::AscendString, ge::AscendString> session_options;
  session_ = new ge::Session(session_options);
  
  // Step 2: 构建 Decode Graph
  build_decode_graph();
  
  LOG(INFO) << "DeepSeek V4 GE model initialized (decode graph compiled)";
}

void DeepseekV4GeModelImpl::build_decode_graph() {
  // 完整推理流程（参考 DS4 Python 实现）
  // 一个 Graph 包含：Embedding → 43 Layers → HC Head → Norm → LM Head
  
  // Step 1: 创建 EsGraphBuilder（构建整个 model）
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_decode");
  
  // Step 2: 创建输入（原始 tokens）
  auto input_ids = builder->CreateInput(0, "input_ids", ge::DT_INT32, {-1});
  
  // Step 3: Embedding lookup
  auto hidden = Embedding(input_ids, embed_tokens_weight_);
  
  // Step 4: HyperConnection expansion
  hidden = Reshape(hidden, {-1, hc_mult_, hidden_size_});
  
  // Step 5: 构建所有 43 DecoderLayers（直接在 builder 上调用）
  auto x = hidden;
  for (int layer = 0; layer < n_layers_; layer++) {
    // 直接调用算子（不分层构建）
    x = build_decoder_layer_ops(builder, x, positions, layer);
  }
  
  // Step 6: HyperConnection Head
  x = HyperConnectionHead(x, hc_head_weights_);
  
  // Step 7: Final norm
  auto norm_out = RMSNorm(x, norm_weight_);
  
  // Step 8: LM Head
  auto logits = MatMul(norm_out, lm_head_weight_);
  
  // Step 9: Build graph（一次性构建完整推理流程）
  auto graph = builder->BuildAndReset({logits});
  
  // Step 10: 配置 dynamicDims
  std::map<ge::AscendString, ge::AscendString> graph_options = {
    {"ge.inputShape", "input_ids:-1"},
    {"ge.dynamicDims", "1,4,8,16,32,64,128"}
  };
  
  // Step 11: AddGraph + CompileGraph
  decode_graph_id_ = next_graph_id_++;
  session_->AddGraph(decode_graph_id_, *graph, graph_options);
  
  ge::Status ret = session_->CompileGraph(decode_graph_id_);
  CHECK(ret == ge::SUCCESS) << "CompileGraph failed: " << ret;
  
  LOG(INFO) << "Decode graph compiled successfully (完整推理流程)";
}
```

---

## 四、Graph 构建方式

### 4.1 不需要独立的 Builder 类

**正确理解**：
- ✅ **所有算子在同一 EsGraphBuilder 上构建**
- ✅ **直接在 build_model_graph() 中构建 43 个 DecoderLayer**
- ✅ **权重作为输入 placeholder**（约 565 个）

### 4.2 可选辅助函数（用于代码组织）

```cpp
// 在 deepseek_v4.cpp 中使用 static helper function

static ge::es::EsTensorHolder build_decoder_layer_ops(
    ge::es::EsGraphBuilder& builder,
    ge::es::EsTensorHolder hidden,
    ge::es::EsTensorHolder positions,
    const std::vector<ge::es::EsTensorHolder>& layer_weights) {
  
  // Step 1: HyperConnectionPre
  auto hc_pre_out = build_hyper_connection_pre(builder, hidden, layer_weights[0]);
  
  // Step 2: MLA Attention
  auto attn_out = build_mla_attention(builder, hc_pre_out, positions, layer_weights[1-4]);
  
  // Step 3: HyperConnectionPost
  auto hc_post_out = build_hyper_connection_post(builder, attn_out, hidden, layer_weights[5]);
  
  // Step 4: FusedMoE
  auto moe_out = build_fused_moe(builder, hc_post_out, layer_weights[6-12]);
  
  return moe_out;
}

void DeepseekV4GeModelImpl::build_model_graph() {
  auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_ge");
  
  // 创建输入 placeholder（约 565 个权重输入）
  auto input_ids = builder->CreateInput(0, "input_ids", ge::DT_INT32, {-1});
  std::vector<ge::es::EsTensorHolder> all_weights = collect_all_weights(builder);
  
  // Embedding
  auto hidden = Embedding(input_ids, all_weights[0]);
  
  // 43 DecoderLayers
  for (int layer = 0; layer < 43; layer++) {
    auto layer_weights = get_layer_weights(all_weights, layer);
    hidden = build_decoder_layer_ops(builder, hidden, positions, layer_weights);
  }
  
  // Final Norm + LM Head
  auto norm_out = RMSNorm(hidden, all_weights[564]);
  auto logits = MatMul(norm_out, all_weights[565]);
  
  auto graph = builder->BuildAndReset({logits});
}
```

---

**文档结束**