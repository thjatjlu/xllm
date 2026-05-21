/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "models/llm/npu_ge/deepseek_v4.h"

#include <glog/logging.h>

#include "core/layers/common/embedding.h"
#include "core/layers/common/rms_norm.h"
#include "core/runtime/ge_session_manager.h"

namespace xllm::models {

DeepseekV4GeModelImpl::DeepseekV4GeModelImpl(const ModelContext& context)
    : model_args_(context.get_model_args()),
      device_(context.get_tensor_options().device()) {
  auto options = context.get_tensor_options();
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();

  n_layers_ = model_args.n_layers();
  hidden_size_ = model_args.hidden_size();
  vocab_size_ = model_args.vocab_size();

  embed_tokens_ = register_module(
      "embed_tokens",
      layer::WordEmbedding(vocab_size_, hidden_size_, parallel_args, options));

  norm_ = register_module(
      "norm", layer::RMSNorm(hidden_size_, model_args.rms_norm_eps(), options));

  lm_head_ = register_module(
      "lm_head", torch::nn::Linear(hidden_size_, vocab_size_, options));

  LOG(INFO) << "DeepseekV4GeModel created (n_layers=" << n_layers_
            << ", hidden_size=" << hidden_size_ << ")";
}

void DeepseekV4GeModelImpl::load_state_dict(const StateDict& state_dict) {
  // 权重加载到成员变量（每次执行需要）
  // frozen_parameter=true 要求权重地址不变

  embed_tokens_->load_state_dict(state_dict);
  norm_->load_state_dict(state_dict);
  lm_head_->load_state_dict(state_dict);

  // for (int i = 0; i < n_layers_; ++i) {
  //   layers_[i]->load_state_dict(state_dict);
  // }

  LOG(INFO)
      << "Weights loaded (member variables will be used for each execution)";
}

void DeepseekV4GeModelImpl::init_model() {
  // TODO: Initialize GE Session
  // auto& session_manager = ge::GeSessionManager::get_instance();
  // session_ = session_manager.get_or_create_session();

  // TODO: Build model graph (ES 构建，只一次)
  // build_model_graph();

  // TODO: Compile two graphs (同一个 graph，不同的 dynamicDims 配置)
  // compile_prefill_graph();  // Prefill Graph (-1 dynamic shape)
  // compile_decode_graph();   // Decode Graph (dynamicDims)

  LOG(INFO) << "DeepSeek V4 GE model initialized (双图方案 pending)";
}

void DeepseekV4GeModelImpl::build_model_graph() {
  // TODO: Build model graph using GE ES API
  // 权重作为 Graph 输入（约 565 个，每次执行传入）
  // frozen_parameter=true：权重地址不变

  // Step 1: 收集所有权重（按顺序）
  // auto weights = collect_all_weights();  // 约 565 个权重

  // Step 2: 创建所有输入 placeholder（按顺序）
  // auto builder = std::make_unique<ge::es::EsGraphBuilder>();
  // int input_index = 0;

  // // 输入 0: input_ids
  // builder->CreateInput(input_index++, "input_ids", ge::DT_INT32, {-1});

  // // 输入 1-565: 所有权重（从 weight tensor 提取 shape/dtype）
  // for (auto& w : weights) {
  //   builder->CreateInput(input_index++,
  //                        "weight_" + std::to_string(input_index),
  //                        GeTensorConverter::map_dtype(w.scalar_type()),
  //                        w.sizes().vec());
  // }

  // Step 3: 构建算子（引用权重输入）
  // auto hidden = Embedding(input_ids, weight_input_0);
  // for (int layer = 0; layer < 43; layer++) {
  //   hidden = build_layer(hidden, layer_weights_inputs[layer]);
  // }
  // auto logits = MatMul(hidden, lm_head_weight_input);

  // Step 4: Build graph
  // model_graph_ = builder->BuildAndReset({logits});

  LOG(INFO) << "Model graph build pending"
            << " - Weight inputs: ~565 (方案 A：不合并)"
            << " - Weights passed as inputs at each execution"
            << " - frozen_parameter=true (地址不变)";
}

void DeepseekV4GeModelImpl::compile_prefill_graph() {
  // TODO: Compile Prefill Graph (-1 dynamic shape)
  // 权重作为输入（约 565 个），每次执行传入
  // frozen_parameter=true：权重地址不变

  // Step 1: Configure -1 dynamic shape
  // std::map<ge::AscendString, ge::AscendString> prefill_options = {
  //   {"ge.inputShape", "input_ids:-1"},
  //   {"frozen_parameter", "true"}  // 权重地址不变
  // };

  // Step 2: AddGraph + CompileGraph
  // prefill_graph_id_ = session_manager.next_graph_id();
  // session_->AddGraph(prefill_graph_id_, *model_graph_, prefill_options);
  // ge::Status ret = session_->CompileGraph(prefill_graph_id_);

  LOG(INFO) << "Prefill Graph compilation pending (约 5min)"
            << " - Weight inputs: ~565 (passed at each execution)"
            << " - frozen_parameter=true (地址不变优化)";
}

void DeepseekV4GeModelImpl::compile_decode_graph() {
  // TODO: Compile Decode Graph (dynamicDims 分档)
  // 权重作为输入（约 565 个），每次执行传入
  // frozen_parameter=true：权重地址不变

  // Step 1: Configure dynamicDims
  // std::map<ge::AscendString, ge::AscendString> decode_options = {
  //   {"ge.inputShape", "input_ids:-1"},
  //   {"ge.dynamicDims", "1,4,8,16,32,64,128"},
  //   {"frozen_parameter", "true"}  // 权重地址不变
  // };

  // Step 2: AddGraph + CompileGraph
  // decode_graph_id_ = session_manager.next_graph_id();
  // session_->AddGraph(decode_graph_id_, *model_graph_, decode_options);
  // ge::Status ret = session_->CompileGraph(decode_graph_id_);

  LOG(INFO)
      << "Decode Graph compilation pending (dynamicDims: 1,4,8,16,32,64,128)"
      << " - Weight inputs: ~565 (passed at each execution)"
      << " - frozen_parameter=true (地址不变优化)";
}

void DeepseekV4GeModelImpl::build_decode_graph() {
  // TODO: Build decode graph using GE ES API
  // 完整推理流程（参考 DS4 Python 实现）
  // 一个 Graph 包含：Embedding → 43 Layers → HC Head → Norm → LM Head

  // Step 1: Create graph builder（构建整个 model）
  // auto builder = std::make_unique<ge::es::EsGraphBuilder>("ds4_decode");

  // Step 2: Create input（原始 tokens，而非 hidden_states）
  // auto input_ids = builder->CreateInput(0, "input_ids", ge::DT_INT32, {-1});

  // Step 3: Embedding lookup
  // auto hidden = Embedding(input_ids, embed_tokens_->weight());

  // Step 4: HyperConnection expansion
  // hidden = Reshape(hidden, {-1, hc_mult_, hidden_size_});

  // Step 5: Build all 43 DecoderLayers（直接在 builder 上调用）
  // auto x = hidden;
  // for (int layer = 0; layer < n_layers_; layer++) {
  //   x = build_decoder_layer_ops(*builder, x, positions, layer);
  // }

  // Step 6: HyperConnection Head
  // x = HyperConnectionHead(x, hc_head_weights_);

  // Step 7: Final norm
  // auto norm_out = RMSNorm(x, norm_->weight());

  // Step 8: LM Head
  // auto logits = MatMul(norm_out, lm_head_->weight());

  // Step 9: Build graph（一次性构建完整推理流程）
  // auto graph = builder->BuildAndReset({logits});

  // Step 10: Configure dynamicDims
  // std::map<ge::AscendString, ge::AscendString> options = {
  //   {"ge.inputShape", "input_ids:-1"},
  //   {"ge.dynamicDims", "1,4,8,16,32,64,128"}
  // };

  // Step 11: AddGraph + CompileGraph
  // decode_graph_id_ = session_manager.next_graph_id();
  // session_->AddGraph(decode_graph_id_, *graph, options);
  // ge::Status ret = session_->CompileGraph(decode_graph_id_);
  // CHECK(ret == ge::SUCCESS) << "CompileGraph failed: " << ret;

  LOG(INFO) << "Decode graph build pending (GE implementation required)"
            << " - Graph includes complete inference flow:"
            << " Embedding → 43 Layers → HC Head → Norm → LM Head";
}

void DeepseekV4GeModelImpl::bind_weights_to_graph(ge::Graph& graph) {
  // TODO: Bind weights to GE graph
  // This ensures weights are used directly without copying

  // for (int layer = 0; layer < n_layers_; layer++) {
  //   // Bind attention weights
  //   // Bind MoE weights
  //   // Bind norm weights
  // }

  // Bind final norm and lm_head weights
}

torch::Tensor DeepseekV4GeModelImpl::forward(torch::Tensor& tokens,
                                             torch::Tensor& positions,
                                             std::vector<KVCache>& kv_caches,
                                             const ModelInputParams& params) {
  // Eager mode forward（参考 DS4 MLU forward_native 实现）
  // 注意：Graph 模式由 Executor 选择，Model::forward 只实现 eager logic

  // Step 1: Embedding lookup
  // auto hidden = embed_tokens_(tokens);

  // Step 2: HyperConnection expansion
  // hidden = hidden.unsqueeze(1).repeat({1, hc_mult_, 1});

  // Step 3: Process through all 43 DecoderLayers
  // std::optional<torch::Tensor> residual;
  // for (int i = 0; i < n_layers_; ++i) {
  //   hidden = layers_[i]->forward(hidden, residual, positions,
  //                                 params.attn_metadata, kv_caches[i],
  //                                 params);
  // }

  // Step 4: HyperConnection Head
  // hidden = hc_head_(hidden, hc_head_fn_, hc_head_scale_, hc_head_base_);

  // Step 5: Final norm
  // auto [h, res] = norm_(hidden, std::nullopt);

  // 注意：LM Head 在 DeepseekV4GeForCausalLM 中处理，不在 Model::forward 中

  // return h;  // 返回 hidden_states，不是 logits

  LOG(WARNING) << "DeepseekV4GeModelImpl::forward() not implemented, returning "
                  "empty tensor";
  return torch::Tensor();
}

}  // namespace xllm::models