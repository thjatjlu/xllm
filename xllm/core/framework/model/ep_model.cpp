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

#include "core/framework/model/ep_model.h"

#if defined(USE_TORCH_DELEGATE)

#include <acl/acl_rt.h>
#include <ge/ge_api_v2.h>
#include <graph/graph.h>
#include <glog/logging.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <string>

#include "absl/strings/str_join.h"
#include "torch_delegate/epair_model_loader.h"
#include "torch_delegate/td_types.h"

namespace xllm {

namespace {

ge::DataType torch_dtype_to_ge(torch::ScalarType scalar_type) {
  switch (scalar_type) {
    case torch::kFloat32:
      return ge::DT_FLOAT;
    case torch::kFloat16:
      return ge::DT_FLOAT16;
    case torch::kBFloat16:
      return ge::DT_BF16;
    case torch::kInt32:
      return ge::DT_INT32;
    case torch::kInt64:
      return ge::DT_INT64;
    case torch::kBool:
      return ge::DT_BOOL;
    case torch::kUInt8:
      return ge::DT_UINT8;
    case torch::kInt8:
      return ge::DT_INT8;
    case torch::kInt16:
      return ge::DT_INT16;
    case torch::kFloat64:
      return ge::DT_DOUBLE;
    default:
      LOG(FATAL) << "Unsupported torch dtype for GE conversion: "
                 << static_cast<int32_t>(scalar_type);
      return ge::DT_FLOAT;
  }
}

torch::ScalarType ge_dtype_to_torch(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_FLOAT:
      return torch::kFloat32;
    case ge::DT_FLOAT16:
      return torch::kFloat16;
    case ge::DT_BF16:
      return torch::kBFloat16;
    case ge::DT_INT32:
      return torch::kInt32;
    case ge::DT_INT64:
      return torch::kInt64;
    case ge::DT_BOOL:
      return torch::kBool;
    case ge::DT_UINT8:
      return torch::kUInt8;
    case ge::DT_INT8:
      return torch::kInt8;
    case ge::DT_INT16:
      return torch::kInt16;
    case ge::DT_DOUBLE:
      return torch::kFloat64;
    default:
      LOG(FATAL) << "Unsupported GE dtype for torch conversion: "
                 << static_cast<uint32_t>(dtype);
      return torch::kFloat32;
  }
}

size_t ge_dtype_size(ge::DataType dtype) {
  switch (dtype) {
    case ge::DT_FLOAT:
    case ge::DT_INT32:
    case ge::DT_UINT32:
      return 4;
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
    case ge::DT_INT16:
    case ge::DT_UINT16:
      return 2;
    case ge::DT_INT64:
    case ge::DT_UINT64:
    case ge::DT_DOUBLE:
      return 8;
    case ge::DT_BOOL:
    case ge::DT_INT8:
    case ge::DT_UINT8:
      return 1;
    default:
      return 4;
  }
}

gert::Tensor torch_to_ge(const torch::Tensor& tensor) {
  CHECK(tensor.is_contiguous()) << "Input tensor must be contiguous";
  CHECK(tensor.device().is_privateuseone())
      << "Input tensor must be on NPU device";

  gert::Tensor ge_tensor;

  gert::StorageShape shape;
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    shape.MutableOriginShape().AppendDim(tensor.size(i));
    shape.MutableStorageShape().AppendDim(tensor.size(i));
  }
  ge_tensor.GetShape() = shape;
  ge_tensor.MutableFormat() =
      gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
  ge_tensor.SetDataType(torch_dtype_to_ge(tensor.scalar_type()));

  const size_t bytes = static_cast<size_t>(tensor.nbytes());
  void* dev_ptr = nullptr;
  auto acl_ret = aclrtMalloc(&dev_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtMalloc failed for input tensor, size=" << bytes;

  acl_ret = aclrtMemcpy(dev_ptr, bytes, tensor.data_ptr(), bytes,
                        ACL_MEMCPY_DEVICE_TO_DEVICE);
  CHECK(acl_ret == ACL_SUCCESS)
      << "aclrtMemcpy D2D failed for input tensor, size=" << bytes;

  ge_tensor.SetData(gert::TensorData(dev_ptr, nullptr, bytes,
                                     gert::kOnDeviceHbm));
  return ge_tensor;
}

torch::Tensor ge_to_torch(const gert::Tensor& ge_tensor,
                          const torch::Device& device) {
  const auto& origin_shape = ge_tensor.GetShape().GetOriginShape();
  std::vector<int64_t> dims;
  dims.reserve(origin_shape.GetDimNum());
  for (size_t i = 0; i < origin_shape.GetDimNum(); ++i) {
    dims.push_back(origin_shape.GetDim(i));
  }

  auto options = torch::TensorOptions()
                     .device(device)
                     .dtype(ge_dtype_to_torch(ge_tensor.GetDataType()));
  torch::Tensor out = torch::empty(dims, options);

  const size_t bytes = ge_tensor.GetSize();
  if (bytes > 0) {
    auto acl_ret = aclrtMemcpy(out.data_ptr(), bytes, ge_tensor.GetAddr(),
                              bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    CHECK(acl_ret == ACL_SUCCESS)
        << "aclrtMemcpy D2D failed for output tensor, size=" << bytes;
  }
  return out;
}

bool is_token_ids(const std::string& name) {
  return name == "input_ids" || name == "tokens" || name == "token_ids";
}

bool is_positions(const std::string& name) {
  return name == "position_ids" || name == "positions";
}

struct KVCacheMatch {
  bool is_key;
  int32_t layer;
};

std::optional<KVCacheMatch> parse_kv_cache_name(const std::string& name) {
  static const std::regex kPatterns[] = {
      std::regex(R"(past_key_values\[(\d+)\]\.(key|value))"),
      std::regex(R"(([kv])_cache\[(\d+)\])"),
      std::regex(R"((key|value)_cache\[(\d+)\])"),
  };

  std::smatch match;

  if (std::regex_match(name, match, kPatterns[0])) {
    int32_t layer = std::stoi(match[1].str());
    bool is_key = (match[2].str() == "key");
    return KVCacheMatch{is_key, layer};
  }

  if (std::regex_match(name, match, kPatterns[1])) {
    bool is_key = (match[1].str() == "k");
    int32_t layer = std::stoi(match[2].str());
    return KVCacheMatch{is_key, layer};
  }

  if (std::regex_match(name, match, kPatterns[2])) {
    bool is_key = (match[1].str() == "key");
    int32_t layer = std::stoi(match[2].str());
    return KVCacheMatch{is_key, layer};
  }

  return std::nullopt;
}

void initialize_ge_once() {
  static std::once_flag ge_init_flag;
  std::call_once(ge_init_flag, []() {
    std::map<ge::AscendString, ge::AscendString> options = {
        {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
    };
    auto status = ge::GEInitializeV2(options);
    CHECK(status == ge::SUCCESS) << "GEInitializeV2 failed, status="
                                 << static_cast<uint32_t>(status);
    LOG(INFO) << "GE initialized successfully (process-level)";
  });
}

}  // anonymous namespace

struct EpModel::Impl {
  std::unique_ptr<td::EpairModelLoader> loader;
};

EpModel::EpModel(const torch::TensorOptions& options)
    : options_(options), impl_(std::make_unique<Impl>()) {
  device_id_ = static_cast<uint64_t>(options.device().index());
}

EpModel::~EpModel() = default;

void EpModel::load_model(std::unique_ptr<ModelLoader> loader) {
  initialize_ge_once();

  std::string epair_path = loader->model_weights_path() + "/model.epair";
  LOG(INFO) << "EpModel loading epair: " << epair_path;

  impl_->loader =
      std::make_unique<td::EpairModelLoader>(epair_path.c_str());

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(static_cast<int32_t>(device_id_)).stream();

  std::map<ge::AscendString, ge::AscendString> compile_options = {
      {ge::AscendString("ge.session_device_id"),
       ge::AscendString(std::to_string(device_id_).c_str())},
  };
  auto status = impl_->loader->CompileAndLoad(compile_options, stream);
  CHECK(status == td::SUCCESS)
      << "EpairModelLoader CompileAndLoad failed, status=" << status
      << ", path=" << epair_path;

  parse_graph_io();

  LOG(INFO) << "EpModel loaded: " << input_names_.size() << " inputs, "
            << output_names_.size() << " outputs"
            << ", path=" << epair_path;
  VLOG(1) << "EpModel input_names: [";
  for (size_t i = 0; i < input_names_.size(); ++i) {
    VLOG(1) << "  [" << i << "] " << input_names_[i];
  }
  VLOG(1) << "]";

  initialized_ = true;
}

void EpModel::parse_graph_io() {
  ge::Graph graph;
  auto status = impl_->loader->GetGEGraph(&graph);
  CHECK(status == td::SUCCESS) << "GetGEGraph failed, status=" << status;

  std::map<int64_t, std::string> indexed_inputs;
  const auto nodes = graph.GetDirectNode();
  for (const auto& node : nodes) {
    ge::AscendString type;
    node.GetType(type);

    if (std::string(type.GetString()) == "Data") {
      ge::AscendString name;
      node.GetName(name);
      int64_t idx = -1;
      node.GetAttr(ge::AscendString("index"), idx);
      indexed_inputs[idx] = name.GetString();
    }

    if (std::string(type.GetString()) == "NetOutput") {
      for (size_t i = 0; i < node.GetInputsSize(); ++i) {
        auto [src_node, src_port] =
            node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
        ge::AscendString name;
        src_node->GetName(name);
        output_names_.push_back(name.GetString());

        ge::TensorDesc tensor_desc;
        auto ret = src_node->GetOutputDesc(src_port, tensor_desc);
        if (ret == ge::GRAPH_SUCCESS) {
          output_shapes_.push_back(tensor_desc.GetShape().GetDims());
          output_dtypes_.push_back(
              static_cast<int>(tensor_desc.GetDataType()));
        } else {
          LOG(WARNING) << "Failed to get output desc for index " << i
                       << ", using defaults";
          output_shapes_.push_back({});
          output_dtypes_.push_back(static_cast<int>(ge::DT_FLOAT));
        }
      }
    }
  }

  input_names_.reserve(indexed_inputs.size());
  for (auto& [idx, name] : indexed_inputs) {
    input_names_.push_back(std::move(name));
  }

  VLOG(1) << "EpModel output info:";
  for (size_t i = 0; i < output_names_.size(); ++i) {
    VLOG(1) << "  [" << i << "] name=" << output_names_[i]
            << ", dtype=" << output_dtypes_[i]
            << ", shape=["
            << absl::StrJoin(output_shapes_[i], ",") << "]";
  }
}

ModelOutput EpModel::forward(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_caches,
                             const ModelInputParams& params) {
  if (!initialized_) {
    LOG(ERROR) << "EpModel not initialized";
    return ModelOutput();
  }

  std::vector<gert::Tensor> graph_inputs;
  graph_inputs.reserve(input_names_.size());
  std::vector<void*> input_dev_ptrs;
  input_dev_ptrs.reserve(input_names_.size());

  for (const auto& name : input_names_) {
    if (is_token_ids(name)) {
      auto ge = torch_to_ge(tokens);
      input_dev_ptrs.push_back(const_cast<void*>(ge.GetAddr()));
      graph_inputs.push_back(std::move(ge));
    } else if (is_positions(name)) {
      auto ge = torch_to_ge(positions);
      input_dev_ptrs.push_back(const_cast<void*>(ge.GetAddr()));
      graph_inputs.push_back(std::move(ge));
    } else if (auto kv_match = parse_kv_cache_name(name)) {
      const int32_t layer = kv_match->layer;
      CHECK(layer >= 0 && layer < static_cast<int32_t>(kv_caches.size()))
          << "KV cache layer index out of range: " << layer
          << ", total layers=" << kv_caches.size();
      const torch::Tensor& cache_tensor = kv_match->is_key
                                              ? kv_caches[layer].get_k_cache()
                                              : kv_caches[layer].get_v_cache();
      auto ge = torch_to_ge(cache_tensor);
      input_dev_ptrs.push_back(const_cast<void*>(ge.GetAddr()));
      graph_inputs.push_back(std::move(ge));
    } else {
      auto tensor_opt =
          params.multimodal.mm_data.get<torch::Tensor>(name);
      if (!tensor_opt.has_value()) {
        LOG(ERROR) << "Missing input tensor in MMData: " << name;
        for (auto* ptr : input_dev_ptrs) {
          if (ptr) {
            aclrtFree(ptr);
          }
        }
        return ModelOutput();
      }
      auto ge = torch_to_ge(tensor_opt.value());
      input_dev_ptrs.push_back(const_cast<void*>(ge.GetAddr()));
      graph_inputs.push_back(std::move(ge));
    }
  }

  aclrtStream stream =
      c10_npu::getCurrentNPUStream(static_cast<int32_t>(device_id_)).stream();

  // Output buffer preparation:
  // - Static shape: pre-allocate device memory for better performance.
  // - Dynamic shape: leave gert::Tensor data unset; GE will allocate
  //   internally during execution.
  std::vector<gert::Tensor> device_outputs;
  device_outputs.resize(output_names_.size());
  std::vector<void*> output_dev_ptrs;
  output_dev_ptrs.reserve(device_outputs.size());

  for (size_t i = 0; i < device_outputs.size() && i < output_names_.size();
       ++i) {
    auto ge_dtype = static_cast<ge::DataType>(output_dtypes_[i]);
    const auto& dims = output_shapes_[i];

    bool is_static = true;
    int64_t num_elements = 1;
    for (const auto& dim : dims) {
      if (dim < 0) {
        is_static = false;
        break;
      }
      num_elements *= dim;
    }

    if (is_static) {
      gert::StorageShape shape;
      for (const auto& dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
      }
      device_outputs[i].GetShape() = shape;
      device_outputs[i].MutableFormat() =
          gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
      device_outputs[i].SetDataType(ge_dtype);

      size_t bytes =
          static_cast<size_t>(num_elements) * ge_dtype_size(ge_dtype);
      void* dev_ptr = nullptr;
      if (bytes > 0) {
        auto acl_ret =
            aclrtMalloc(&dev_ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK(acl_ret == ACL_SUCCESS)
            << "aclrtMalloc failed for output " << i << ", size=" << bytes;
      }
      output_dev_ptrs.push_back(dev_ptr);
      device_outputs[i].SetData(gert::TensorData(dev_ptr, nullptr, bytes,
                                                 gert::kOnDeviceHbm));
      VLOG(1) << "Output [" << i << "] \"" << output_names_[i]
              << "\" static pre-alloc: bytes=" << bytes;
    } else {
      output_dev_ptrs.push_back(nullptr);
      VLOG(1) << "Output [" << i << "] \"" << output_names_[i]
              << "\" dynamic shape, deferred to GE runtime";
    }
  }

  auto run_status = impl_->loader->RunModelWithStreamAsync(
      stream, graph_inputs, device_outputs);
  if (run_status != td::SUCCESS) {
    LOG(ERROR) << "RunModelWithStreamAsync failed, status=" << run_status;
    for (auto* ptr : input_dev_ptrs) {
      if (ptr) {
        aclrtFree(ptr);
      }
    }
    for (auto* ptr : output_dev_ptrs) {
      if (ptr) {
        aclrtFree(ptr);
      }
    }
    return ModelOutput();
  }

  aclrtSynchronizeStream(stream);

  ModelOutput result;

  for (size_t i = 0; i < device_outputs.size() && i < output_names_.size(); ++i) {
    // For dynamic outputs, GE allocated device memory internally.
    // Collect the address so we can free it after D2H copy.
    if (output_dev_ptrs[i] == nullptr) {
      output_dev_ptrs[i] = const_cast<void*>(device_outputs[i].GetAddr());
    }

    const void* src_addr = device_outputs[i].GetAddr();
    if (src_addr == nullptr) {
      LOG(WARNING) << "Output [" << i << "] \"" << output_names_[i]
                   << "\" has null device address after execution";
      continue;
    }

    auto ge_dtype = static_cast<ge::DataType>(output_dtypes_[i]);
    const auto& dims = output_shapes_[i];

    size_t bytes = device_outputs[i].GetSize();
    if (bytes == 0) {
      int64_t num_elements = 1;
      for (const auto& dim : dims) {
        if (dim > 0) {
          num_elements *= dim;
        }
      }
      bytes = static_cast<size_t>(num_elements) * ge_dtype_size(ge_dtype);
    }

    LOG(INFO) << "Output [" << i << "] \"" << output_names_[i]
              << "\" src_addr=" << src_addr
              << ", ge_bytes=" << device_outputs[i].GetSize()
              << ", computed_bytes=" << bytes
              << ", dims=[" << absl::StrJoin(dims, ",") << "]"
              << ", ge_dtype=" << static_cast<int>(ge_dtype);

    auto torch_dtype = ge_dtype_to_torch(ge_dtype);

    // Create output tensor on CPU and do D2H copy (synchronous).
    // Avoids stream ordering issues between torch NPU allocator and
    // aclrtMemcpy D2D async submission.
    torch::Tensor torch_output =
        torch::empty(dims, torch::TensorOptions()
                               .device(torch::kCPU)
                               .dtype(torch_dtype));

    LOG(INFO) << "torch_output(CPU): data_ptr=" << torch_output.data_ptr()
              << ", nbytes=" << torch_output.nbytes();

    if (bytes > 0 && torch_output.data_ptr() != nullptr) {
      auto acl_ret = aclrtMemcpy(torch_output.data_ptr(), bytes, src_addr,
                                 bytes, ACL_MEMCPY_DEVICE_TO_HOST);
      LOG(INFO) << "aclrtMemcpy D2H result=" << acl_ret
                << ", dst=" << torch_output.data_ptr()
                << ", src=" << src_addr
                << ", bytes=" << bytes;
      CHECK(acl_ret == ACL_SUCCESS)
          << "aclrtMemcpy D2H failed for output " << i
          << ", bytes=" << bytes;
    } else {
      LOG(WARNING) << "Output [" << i << "] \"" << output_names_[i]
                   << "\" empty output or null torch data_ptr, bytes=" << bytes;
    }

    VLOG(1) << "Output [" << i << "] \"" << output_names_[i]
            << "\" D2H copied: bytes=" << bytes
            << ", dims=[" << absl::StrJoin(dims, ",") << "]";

    result.graph_outputs[output_names_[i]] = std::move(torch_output);
  }

  // Synchronize before freeing device memory: aclrtMemcpy D2D may be
  // asynchronously submitted to the stream.
  aclrtSynchronizeStream(stream);

  if (!result.graph_outputs.empty()) {
    auto it = result.graph_outputs.begin();
    result.hidden_states = it->second;
  }

  for (auto* ptr : input_dev_ptrs) {
    if (ptr) {
      aclrtFree(ptr);
    }
  }
  for (auto* ptr : output_dev_ptrs) {
    if (ptr) {
      aclrtFree(ptr);
    }
  }

  return result;
}

torch::Tensor EpModel::logits(const torch::Tensor& hidden_states,
                              const torch::Tensor& selected_idxes) {
  LOG(WARNING) << "EpModel::logits() called but GE graph handles "
                  "sampling internally; returning empty tensor";
  return torch::Tensor();
}

torch::Device EpModel::device() const {
  return options_.device();
}

const torch::TensorOptions& EpModel::options() const {
  return options_;
}

}  // namespace xllm

#endif  // USE_TORCH_DELEGATE
