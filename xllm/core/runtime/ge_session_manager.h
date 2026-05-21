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

#include <cstdint>
#include <memory>

namespace xllm::ge {

class GeSessionManager {
 public:
  static GeSessionManager& get_instance();

  ge::Session* get_or_create_session();

  void destroy_session();

  uint32_t next_graph_id();

 private:
  GeSessionManager() = default;
  ~GeSessionManager() = default;

  ge::Session* session_ = nullptr;
  uint32_t next_graph_id_ = 0;
};

}  // namespace xllm::ge