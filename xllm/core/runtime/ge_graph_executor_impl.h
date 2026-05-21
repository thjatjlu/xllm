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

#pragma once

#include <torch/torch.h>

#include "executor_impl.h"
#include "executor_impl_factory.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "options.h"

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

  // TODO: Add GE Session and Graph ID management
  // ge::Session* session_;
  // uint32_t prefill_graph_id_;  // Prefill Graph (-1 dynamic shape)
  // uint32_t decode_graph_id_;   // Decode Graph (dynamicDims)

  // TODO: Implement prefill graph execution (-1 dynamic shape)
  ModelOutput run_prefill_graph(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params);

  // TODO: Implement decode graph execution (dynamicDims)
  ModelOutput run_decode_graph(const torch::Tensor& tokens,
                               const torch::Tensor& positions,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& params);

  // TODO: Implement tensor conversion helpers
  std::vector<ge::Tensor> convert_inputs(const torch::Tensor& tokens,
                                         const torch::Tensor& positions,
                                         std::vector<KVCache>& kv_caches,
                                         const ModelInputParams& params);

  ModelOutput convert_outputs(const std::vector<ge::Tensor>& ge_outputs);
};

REGISTER_EXECUTOR("npu_ge", GeGraphExecutorImpl);

}  // namespace xllm::ge