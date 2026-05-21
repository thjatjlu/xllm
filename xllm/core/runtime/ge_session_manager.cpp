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

#include "ge_session_manager.h"

#include <glog/logging.h>

namespace xllm::ge {

GeSessionManager& GeSessionManager::get_instance() {
  static GeSessionManager instance;
  return instance;
}

ge::Session* GeSessionManager::get_or_create_session() {
  // TODO: Implement GE Session creation

  // if (session_ == nullptr) {
  //   std::map<ge::AscendString, ge::AscendString> options;
  //   session_ = new ge::Session(options);
  //   LOG(INFO) << "GE Session created";
  // }

  // return session_;

  return nullptr;
}

void GeSessionManager::destroy_session() {
  // TODO: Implement GE Session cleanup

  // if (session_ != nullptr) {
  //   delete session_;
  //   session_ = nullptr;
  //   LOG(INFO) << "GE Session destroyed";
  // }
}

uint32_t GeSessionManager::next_graph_id() { return next_graph_id_++; }

}  // namespace xllm::ge