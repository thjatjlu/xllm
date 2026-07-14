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

#include "ge_graph_batch_input_builder.h"

namespace xllm {

GeGraphBatchInputBuilder::GeGraphBatchInputBuilder(
    const std::vector<SequencesGroup*>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    const uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    MPMCThreadPool* thread_pool)
    : sequence_groups_(sequence_groups),
      mm_data_vec_(mm_data_vec),
      batch_id_(batch_id),
      args_(args) {}

ForwardInput GeGraphBatchInputBuilder::build_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  int32_t num_sequences = 0;
  for (size_t i = 0; i < sequence_groups_.size(); ++i) {
    num_sequences += static_cast<int32_t>(sequence_groups_[i]->sequences().size());
  }
  if (num_sequences == 0) {
    return ForwardInput{};
  }

  ForwardInput forward_input;
  auto& input_params = forward_input.input_params;

  forward_input.token_ids = torch::tensor({0}, torch::kInt);
  forward_input.positions = torch::tensor({0}, torch::kInt);
  forward_input.token_ids_host = forward_input.token_ids;
  forward_input.positions_host = forward_input.positions;

  input_params.meta.num_sequences = num_sequences;
  input_params.meta.batch_id = batch_id_;
  input_params.meta.kv_max_seq_len = 1;
  input_params.meta.q_max_seq_len = 1;

  std::vector<MMData> mm_datas;
  for (size_t i = 0; i < sequence_groups_.size(); ++i) {
    for (const auto& seq : sequence_groups_[i]->sequences()) {
      if (seq->mm_data().valid()) {
        mm_datas.push_back(seq->mm_data());
      }
    }
  }
  if (!mm_datas.empty()) {
    input_params.multimodal.mm_data = MMBatchData(mm_datas);
  }

  std::vector<const RequestSamplingParam*> sampling_params;
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  int32_t idx = 0;
  for (size_t i = 0; i < sequence_groups_.size(); ++i) {
    for (const auto& seq : sequence_groups_[i]->sequences()) {
      sampling_params.push_back(seq->sampling_param());
      selected_token_idxes.push_back(idx);
      sample_idxes.push_back(idx);
      ++idx;
    }
  }
  forward_input.sampling_params.init(sampling_params,
                                     selected_token_idxes,
                                     sample_idxes,
                                     std::vector<std::vector<int64_t>>{},
                                     std::vector<std::vector<int32_t>>{},
                                     std::vector<int32_t>{});

  return forward_input;
}

}  // namespace xllm
