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

namespace xllm::ge {

class GeTensorConverter {
 public:
  static ge::Tensor torch_to_ge(const torch::Tensor& torch_tensor);

  static torch::Tensor ge_to_torch(const ge::Tensor& ge_tensor);

 private:
  static ge::DataType map_dtype(torch::ScalarType torch_dtype);
  static torch::ScalarType map_ge_dtype(ge::DataType ge_dtype);

  static ge::Format infer_format(const std::vector<int64_t>& sizes);
};

}  // namespace xllm::ge