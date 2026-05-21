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

#include "ge_graph_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"

namespace xllm::ge {

GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // TODO: Initialize GE Session
  // std::map<ge::AscendString, ge::AscendString> session_options;
  // session_ = new ge::Session(session_options);

  // TODO: Build and compile two graphs (双图方案)
  // build_model_graph();       // ES 构建，只一次
  // compile_prefill_graph();   // Prefill Graph (-1 dynamic shape)
  // compile_decode_graph();    // Decode Graph (dynamicDims)

  LOG(INFO) << "GeGraphExecutorImpl initialized (backend: npu_ge, 双图方案)";
}

GeGraphExecutorImpl::~GeGraphExecutorImpl() {
  // TODO: Clean up GE Session and Graphs
  // if (session_) {
  //   session_->RemoveGraph(decode_graph_id_);
  //   delete session_;
  // }
}

ForwardInput GeGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // TODO: Implement prepare_inputs logic
  // Similar to MluGraphExecutorImpl::prepare_inputs
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput GeGraphExecutorImpl::run(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
  // Step 1: Check ForwardType
  if (!params.batch_forward_type.is_decode()) {
    // PREFILL / CHUNKED_PREFILL / MIXED → eager mode
    COUNTER_INC(num_model_execution_total_eager);
    VLOG(50) << "GeGraphExecutorImpl::run() in eager mode (forward_type="
             << params.batch_forward_type.to_string() << ")";

    // TODO: Call model forward in eager mode
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Step 2: DECODE → use decode graph
  VLOG(50) << "GeGraphExecutorImpl::run() in decode graph mode";
  COUNTER_INC(num_model_execution_total_graph);

  return run_decode_graph(tokens, positions, kv_caches, params);
}

ModelOutput GeGraphExecutorImpl::run_decode_graph(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
  // TODO: Implement decode graph execution
  // 权重作为 Graph 输入（约 565 个），每次执行传入
  // 输入顺序：Graph 构建顺序 = 执行传入顺序

  // Step 1: 收集所有权重（按顺序）
  // auto weights = model_->collect_all_weights();  // 约 565 个

  // Step 2: Convert all inputs to GE tensors（按顺序）
  // std::vector<ge::Tensor> ge_inputs;
  // ge_inputs.push_back(GeTensorConverter::torch_to_ge(tokens));  // 输入 0
  // for (auto& w : weights) {
  //   ge_inputs.push_back(GeTensorConverter::torch_to_ge(w));    // 输入 1-565
  // }

  // Step 3: Run GE graph
  // aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  // std::vector<ge::Tensor> ge_outputs;
  // session_->RunGraph(decode_graph_id_, ge_inputs, ge_outputs);

  // Step 4: Convert outputs
  // return convert_outputs(ge_outputs);

  LOG(WARNING) << "Decode graph not implemented, fallback to eager mode";
  return model_->forward(tokens, positions, kv_caches, params);
}

std::vector<ge::Tensor> GeGraphExecutorImpl::convert_inputs(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
  // TODO: Implement torch::Tensor → ge::Tensor conversion
  // Use zero-copy approach similar to design document

  std::vector<ge::Tensor> ge_inputs;

  // ge_inputs.push_back(GeTensorConverter::torch_to_ge(tokens));
  // ge_inputs.push_back(GeTensorConverter::torch_to_ge(positions));

  // // Add KV Cache tensors
  // for (auto& kv_cache : kv_caches) {
  //   ge_inputs.push_back(GeTensorConverter::torch_to_ge(kv_cache.k_cache));
  //   ge_inputs.push_back(GeTensorConverter::torch_to_ge(kv_cache.v_cache));
  // }

  // // Add block_tables and other metadata
  // ge_inputs.push_back(GeTensorConverter::torch_to_ge(params.block_tables));

  return ge_inputs;
}

ModelOutput GeGraphExecutorImpl::convert_outputs(
    const std::vector<ge::Tensor>& ge_outputs) {
  // TODO: Implement ge::Tensor → torch::Tensor conversion
  // Use zero-copy approach

  // auto hidden_states = GeTensorConverter::ge_to_torch(ge_outputs[0]);
  // return ModelOutput(hidden_states);

  return ModelOutput();
}

}  // namespace xllm::ge