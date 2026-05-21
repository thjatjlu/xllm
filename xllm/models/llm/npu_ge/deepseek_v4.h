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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/model/model_output.h"
#include "core/framework/state_dict/state_dict.h"
#include "llm_model_base.h"

namespace xllm::models {

class DeepseekV4GeModelImpl : public torch::nn::Module {
 public:
  explicit DeepseekV4GeModelImpl(const ModelContext& context);

  void load_state_dict(const StateDict& state_dict);

  void init_model();

  torch::Tensor forward(torch::Tensor& tokens,
                        torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& params);
  // 返回 hidden_states（不含 LM Head）
  // LM Head 在 DeepseekV4GeForCausalLM 中处理

  // ❌ Graph 模式不支持 Rolling Load（报错处理）
  void free_weights() {
    LOG(ERROR) << "free_weights() not supported in GE Graph mode";
    CHECK(false) << "Rolling Load is not supported in GE Graph mode. "
                 << "Weights are passed as inputs (约 565 个). "
                 << "frozen_parameter=true requires address stability. "
                 << "Member variables cannot be released.";
  }

  void reload_weights() {
    LOG(ERROR) << "reload_weights() not supported in GE Graph mode";
    CHECK(false) << "Rolling Load is not supported in GE Graph mode. "
                 << "Weights are passed as inputs (约 565 个). "
                 << "frozen_parameter=true requires address stability. "
                 << "Member variables cannot be released.";
  }

  // 收集所有权重（按顺序，用于 Graph 构建/执行）
  std::vector<torch::Tensor> collect_all_weights();

 private:
  ModelArgs model_args_;
  torch::Device device_;

  // 成员变量：存储权重（每次执行需要，不可释放）
  // 原因：权重作为 Graph 输入，每次执行传入
  // frozen_parameter=true 要求权重地址不变
  torch::nn::Embedding embed_tokens_{nullptr};
  std::vector<DeepseekV4GeDecoderLayer> layers_;
  RMSNorm norm_{nullptr};
  torch::nn::Linear lm_head_{nullptr};

  int32_t n_layers_;
  int64_t hidden_size_;
  int64_t vocab_size_;
  int64_t hc_mult_;

  // HyperConnection Head weights（每次执行需要）
  torch::Tensor hc_head_weight_;
  torch::Tensor hc_head_scale_;
  torch::Tensor hc_head_base_;

  // TODO: Add GE Session and Graph ID management
  // ge::Session* session_;
  // uint32_t prefill_graph_id_;  // Prefill Graph (-1 dynamic shape)
  // uint32_t decode_graph_id_;   // Decode Graph (dynamicDims)

  void build_model_graph();
  void compile_prefill_graph();
  void compile_decode_graph();
};

TORCH_MODULE(DeepseekV4GeModel);

class DeepseekV4GeForCausalLMImpl
    : public LlmForCausalLMImplBase<DeepseekV4GeModel> {
 public:
  explicit DeepseekV4GeForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DeepseekV4GeModel>(context) {}
};

TORCH_MODULE(DeepseekV4GeForCausalLM);

REGISTER_CAUSAL_MODEL(deepseek_v4_npu_ge, DeepseekV4GeForCausalLM);

}  // namespace xllm::models