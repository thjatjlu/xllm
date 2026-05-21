# GE Model CMakeLists.txt
# DeepSeek V4 NPU GE Model Implementation

# Note: This CMakeLists.txt is currently for documentation purposes only.
# Actual compilation will be enabled after GE dependencies are configured.

# TODO: Add GE library dependencies when ready for compilation
# find_package(GE REQUIRED)
# find_package(ACL REQUIRED)

# TODO: Define GE model sources
# set(GE_MODEL_SOURCES
#   deepseek_v4.cpp
# )

# TODO: Create library target
# add_library(xllm_ge_models STATIC ${GE_MODEL_SOURCES})

# TODO: Add include directories
# target_include_directories(xllm_ge_models PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_SOURCE_DIR}/xllm/core/runtime
#   $ENV{ASCEND_HOME_PATH}/include/ge
# )

# TODO: Link dependencies
# target_link_libraries(xllm_ge_models
#   xllm_ge_runtime
#   xllm_ge_layers
#   ge_runner
#   ge_compiler
#   ascendcl
# )

# TODO: Add to parent CMakeLists.txt
# add_subdirectory(npu_ge)