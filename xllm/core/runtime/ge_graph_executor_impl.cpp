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

#if defined(USE_TORCH_DELEGATE)

#include <glog/logging.h>

#include <typeinfo>

#include "common/metrics.h"
#include "core/framework/model/ep_model.h"

namespace xllm {

GeGraphExecutorImpl::GeGraphExecutorImpl(CausalLM* model,
                                         const ModelArgs& args,
                                         const torch::Device& device,
                                         const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  if (model_ == nullptr) {
    LOG(ERROR) << "GeGraphExecutorImpl requires non-null model";
    return;
  }

  auto* ep_model = dynamic_cast<EpModel*>(model_);
  if (ep_model == nullptr) {
    LOG(ERROR) << "GeGraphExecutorImpl requires EpModel, got "
               << typeid(*model_).name();
    return;
  }

  if (!ep_model->is_initialized()) {
    LOG(ERROR) << "EpModel is not initialized";
    return;
  }

  device_id_ = static_cast<uint64_t>(device.index());
  initialized_ = true;
}

ForwardInput GeGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // GE graph mode: prepare_inputs is handled by GeGraphWorkerPipeline,
  // not by the executor. This method should not be called.
  LOG(FATAL) << "GeGraphExecutorImpl::prepare_inputs should not be called; "
             << "input preparation is handled by GeGraphWorkerPipeline";
  return ForwardInput{};
}

ModelOutput GeGraphExecutorImpl::run(const torch::Tensor& tokens,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& params) {
  if (!initialized_) {
    LOG(ERROR) << "GeGraphExecutorImpl not initialized";
    return ModelOutput();
  }
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm

#endif  // USE_TORCH_DELEGATE
