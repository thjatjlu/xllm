# GE Layer CMakeLists.txt
# DeepSeek V4 NPU GE Layer Implementation

# Note: This CMakeLists.txt is currently for documentation purposes only.
# Actual compilation will be enabled after GE dependencies are configured.

# TODO: Add GE library dependencies when ready for compilation
# find_package(GE REQUIRED)
# find_package(ACL REQUIRED)

# TODO: Define GE layer sources
# set(GE_LAYER_SOURCES
#   deepseek_v4_decoder_layer_builder.cpp
# )

# TODO: Create library target
# add_library(xllm_ge_layers STATIC ${GE_LAYER_SOURCES})

# TODO: Add include directories
# target_include_directories(xllm_ge_layers PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   $ENV{ASCEND_HOME_PATH}/include/ge
# )

# TODO: Link dependencies
# target_link_libraries(xllm_ge_layers
#   ge_runner
#   ge_compiler
#   ascendcl
# )

# TODO: Add to parent CMakeLists.txt
# add_subdirectory(npu_ge)