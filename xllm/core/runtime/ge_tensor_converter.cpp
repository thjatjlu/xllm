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

#include "ge_tensor_converter.h"

#include <glog/logging.h>

namespace xllm::ge {

ge::Tensor GeTensorConverter::torch_to_ge(const torch::Tensor& torch_tensor) {
  // TODO: Implement zero-copy torch::Tensor → ge::Tensor conversion

  // Step 1: Extract tensor properties
  // auto sizes = torch_tensor.sizes().vec();
  // auto dtype = torch_tensor.scalar_type();
  // auto data_ptr = torch_tensor.data_ptr();

  // Step 2: Map dtype
  // ge::DataType ge_dtype = map_dtype(dtype);
  // ge::Format format = infer_format(sizes);

  // Step 3: Create TensorDesc
  // ge::Shape shape(sizes);
  // ge::TensorDesc desc(shape, format, ge_dtype);

  // Step 4: Create ge::Tensor (zero-copy)
  // ge::Tensor ge_tensor;
  // ge_tensor.SetTensorDesc(desc);
  // ge_tensor.SetData(
  //     reinterpret_cast<uint8_t*>(data_ptr),
  //     torch_tensor.numel() * torch_tensor.element_size());

  // return ge_tensor;

  return ge::Tensor();
}

torch::Tensor GeTensorConverter::ge_to_torch(const ge::Tensor& ge_tensor) {
  // TODO: Implement zero-copy ge::Tensor → torch::Tensor conversion

  // Step 1: Get TensorDesc
  // auto desc = ge_tensor.GetTensorDesc();
  // auto shape = desc.GetShape();
  // auto ge_dtype = desc.GetDataType();

  // Step 2: Convert to torch properties
  // std::vector<int64_t> sizes(shape.GetDims().begin(), shape.GetDims().end());
  // torch::ScalarType torch_dtype = map_ge_dtype(ge_dtype);

  // Step 3: Get data pointer
  // auto data_ptr = ge_tensor.GetData();
  // auto data_size = ge_tensor.GetSize();

  // Step 4: Create torch tensor from NPU memory
  // return torch::from_blob(data_ptr, sizes, torch_dtype);

  return torch::Tensor();
}

ge::DataType GeTensorConverter::map_dtype(torch::ScalarType torch_dtype) {
  // TODO: Implement dtype mapping

  // switch (torch_dtype) {
  //   case torch::kFloat32: return ge::DT_FLOAT;
  //   case torch::kFloat16: return ge::DT_FLOAT16;
  //   case torch::kBFloat16: return ge::DT_BF16;
  //   case torch::kInt32: return ge::DT_INT32;
  //   case torch::kInt64: return ge::DT_INT64;
  //   default:
  //     LOG(FATAL) << "Unsupported dtype: " << torch_dtype;
  //     return ge::DT_UNDEFINED;
  // }

  return ge::DT_UNDEFINED;
}

torch::ScalarType GeTensorConverter::map_ge_dtype(ge::DataType ge_dtype) {
  // TODO: Implement reverse dtype mapping

  // switch (ge_dtype) {
  //   case ge::DT_FLOAT: return torch::kFloat32;
  //   case ge::DT_FLOAT16: return torch::kFloat16;
  //   case ge::DT_BF16: return torch::kBFloat16;
  //   case ge::DT_INT32: return torch::kInt32;
  //   case ge::DT_INT64: return torch::kInt64;
  //   default:
  //     LOG(FATAL) << "Unsupported ge dtype: " << ge_dtype;
  //     return torch::kFloat32;
  // }

  return torch::kFloat32;
}

ge::Format GeTensorConverter::infer_format(const std::vector<int64_t>& sizes) {
  // TODO: Implement format inference based on tensor shape

  // Default to FORMAT_ND for flexible shapes
  // return ge::FORMAT_ND;

  return ge::FORMAT_ND;
}

}  // namespace xllm::ge