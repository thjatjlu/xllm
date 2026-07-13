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

#if defined(USE_TORCH_DELEGATE)

#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"

namespace xllm {

class EpModel final : public CausalLM {
 public:
  explicit EpModel(const torch::TensorOptions& options);
  ~EpModel() override;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override;

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override;

  void load_model(std::unique_ptr<ModelLoader> loader) override;

  torch::Device device() const override;
  const torch::TensorOptions& options() const override;

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {}
  void update_expert_weight(int32_t layer_id) override {}

  bool is_initialized() const { return initialized_; }

 private:
  // PIMPL to hide CANN/GE/gert types from this header.
  struct Impl;

  void parse_graph_io();

  std::unique_ptr<Impl> impl_;
  torch::TensorOptions options_;
  uint64_t device_id_ = 0;
  bool initialized_ = false;

  // Ordered graph input/output names parsed from the GE Graph.
  // Populated by parse_graph_io() during load_model().
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> output_shapes_;
  std::vector<int> output_dtypes_;  // ge::DataType as int to avoid header leak
};

}  // namespace xllm

#endif  // USE_TORCH_DELEGATE
