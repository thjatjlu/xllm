# torch-delegate 接入 xLLM 调试指南

## 概述

本文档描述如何使用最简单的 `mm + bias + transpose` 模型（来自 torch-delegate examples），打通 xLLM 的 GE Graph 全链路（请求 → EpModel → GE 图执行 → 响应）。

## 前置条件

```bash
# CANN 环境
source /usr/local/Ascend/cann/set_env.sh

# torch-delegate 环境（编译安装后）
source /path/to/torch-delegate-install/set_env.sh

# 验证
echo $ASCEND_HOME_PATH     # CANN 路径
echo $TORCH_DELEGATE_ROOT   # torch-delegate 安装路径
ls $TORCH_DELEGATE_ROOT/lib/libtorch_delegate_backend.so  # 确认 .so 存在
```

---

## Step 1：导出 epair 模型

```bash
cd /path/to/torch-delegate/examples/export_and_load_example

# 导出（生成 epair_model.epair + input_0.bin）
python3 export_ep_and_epair.py
```

模型结构：
```python
class MyModel(torch.nn.Module):
    def forward(self, x):          # 输入: x, [3, 4], float32
        y = torch.mm(x, self.weight) + self.bias
        return torch.transpose(y, 0, 1)  # 输出: [3, 3], float32
```

验证：
```bash
ls epair_model.epair input_0.bin
```

---

## Step 2：组装 xLLM 模型目录

模型目录放在 xllm 代码仓的上一层：

```bash
cd /path/to/xllm       # 进入 xllm 代码目录
cd ..                   # 回到上一层
mkdir -p models/simple-ge

# 复制 epair 文件（必须命名为 model.epair）
cp /path/to/torch-delegate/examples/export_and_load_example/epair_model.epair \
   models/simple-ge/model.epair
```

创建最小 `config.json`：
```bash
cat > models/simple-ge/config.json << 'EOF'
{
  "hidden_size": 4,
  "n_layers": 1,
  "n_heads": 1,
  "head_dim": 4,
  "dtype": "float32",
  "vocab_size": 1,
  "max_position_embeddings": 1,
  "eos_token_id": 0,
  "bos_token_id": 0
}
EOF
```

验证目录结构：
```bash
ls models/simple-ge/
# 应输出: config.json  model.epair
```

---

## Step 3：编译 xLLM

xLLM 使用 `python setup.py build` 编译，底层仍然是调 CMake。通过 `CMAKE_ARGS` 环境变量传递额外 CMake 选项。

```bash
cd /path/to/xllm

export TORCH_DELEGATE_ROOT=/path/to/torch-delegate-install
source /usr/local/Ascend/cann/set_env.sh

# 通过 CMAKE_ARGS 传递 USE_TORCH_DELEGATE=ON
export CMAKE_ARGS="-DUSE_TORCH_DELEGATE=ON"

# 编译（与平时一样）
python setup.py build --device npu

# 或者 build + install
python setup.py build --device npu
pip install -e .
```

验证编译：
```bash
# 确认 USE_TORCH_DELEGATE 宏生效（查找编译产物中的 EpModel 符号）
find build/ -name "xllm" -type f 2>/dev/null | head -1 | xargs nm 2>/dev/null | grep EpModel | head -3
# 或者检查 .o 文件
find build/ -name "ep_model*" | head -5
```

> **不设置 `CMAKE_ARGS` 时完全无影响**：`USE_TORCH_DELEGATE` 默认 `OFF`，所有 GE Graph 代码通过 `#ifdef` 跳过，编译产物和之前完全一致。

---

## Step 4：启动 xLLM

```bash
cd /path/to/xllm

export TORCH_DELEGATE_ROOT=/path/to/torch-delegate-install
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$TORCH_DELEGATE_ROOT/lib:$LD_LIBRARY_PATH

# 开启详细日志
# 注意：参数名是 --model（不是 --model_path），--devices 是设备号
GLOG_v=1 ./build/xllm/core/server/xllm serve \
  --model=../models/simple-ge \
  --backend=ge \
  --host=0.0.0.0 \
  --port=8080 \
  --devices=npu:0
```

### 预期日志

```
GE initialized successfully (process-level)
EpModel loading epair: ../models/simple-ge/model.epair
EpModel loaded: 1 inputs, 1 outputs, path=../models/simple-ge/model.epair
EpModel input_names: [0] x
```

### 日志确认清单

| 日志行 | 含义 | 不出现的排查 |
|--------|------|-------------|
| `GE initialized successfully` | GE V2 初始化成功 | 检查 CANN 环境 |
| `EpModel loading epair` | 开始加载 epair | 检查 `--backend=ge` 是否生效 |
| `EpModel loaded: N inputs, M outputs` | epair 解析成功 | 检查 epair 文件是否完整 |
| `EpModel input_names: [0] x` | 图输入名解析正确 | 后续请求的 name 需与此一致 |

---

## Step 5：发送测试请求

### 5.1 准备请求体

```bash
cat > /tmp/req.json << 'EOF'
{
  "model": "simple-ge",
  "prompt": "",
  "token_ids": [0],
  "input_tensors": [
    {
      "name": "x",
      "shape": [3, 4],
      "data_type": "FLOAT",
      "contents": {
        "fp32_contents": [
          1.0, 0.0, 0.0, 0.0,
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0
        ]
      }
    }
  ],
  "max_tokens": 1
}
EOF
```

> **说明**：
> - `token_ids: [0]` 是占位，xLLM 需要至少有 token_ids 或 input_tensors
> - `prompt` 是 proto 必填字段，GE 模式传空字符串即可
> - `input_tensors` 的 `name` 必须与日志中 `input_names_` 一致（本例为 `"x"`）
> - `data_type` 必须用 `FLOAT`（proto 枚举名），字段名是 `data_type`（不是 `dtype`）

### 5.2 发送请求

```bash
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/req.json | python3 -m json.tool
```

### 5.3 预期响应

```json
{
  "id": "...",
  "object": "text_completion",
  "created": ...,
  "model": "simple-ge",
  "choices": [
    {
      "index": 0,
      "text": "",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 0,
    "total_tokens": 1
  },
  "output_tensors": [
    {
      "name": "<图输出名>",
      "shape": [3, 3],
      "datatype": "FLOAT",
      "contents": {
        "fp32_contents": [0.1, 0.2, ...]
      }
    }
  ]
}
```

> `output_tensors` 中的 `name` 由 epair 图决定，不是硬编码的。

---

## Step 6：精度对比验证

### 6.1 获取参考输出

```bash
cd /path/to/torch-delegate/examples/export_and_load_example

# 运行 example 的 C++ 独立程序
bash run.sh

# 生成的 delegate_output_0.bin 是参考输出
xxd delegate_output_0.bin | head
```

### 6.2 提取 xLLM 响应中的输出

```bash
# 发送请求并提取 fp32_contents
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d @/tmp/req.json \
  | python3 -c "
import sys, json
resp = json.load(sys.stdin)
tensors = resp.get('output_tensors', [])
for t in tensors:
    print(f'name: {t[\"name\"]}')
    print(f'shape: {t[\"shape\"]}')
    data = t['contents']['fp32_contents']
    print(f'data: {data}')
"
```

### 6.3 对比

```bash
python3 -c "
import numpy as np
ref = np.fromfile('/path/to/torch-delegate/examples/export_and_load_example/delegate_output_0.bin', dtype=np.float32)
print(f'参考输出: {ref.tolist()}')
print(f'请手动对比上面 xLLM 响应中的 fp32_contents')
"
```

---

## 常见问题排查

### 启动阶段

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `unknown command line flag 'model_path'` | 参数名不对 | 用 `--model`（不是 `--model_path`） |
| `GEInitializeV2 failed` | CANN 环境未初始化 | `source set_env.sh`，检查 `ASCEND_HOME_PATH` |
| `Failed to find model weights files` | HFModelLoader 没检测到 epair | 确认 `model.epair` 在模型目录下 |
| `CompileAndLoad failed` | epair 文件损坏或 CANN 版本不兼容 | 检查 torch-delegate 和 CANN 版本一致 |
| `EpModel loaded: 0 inputs` | GetGEGraph 解析失败 | 检查 epair 文件是否正确导出 |
| `undefined reference to GEInitializeV2` | 链接缺少 libge_runner_v2.so | 确认 CMakeLists.txt 中 `find_library(GE_RUNNER_V2_SO ...)` 生效 |
| ABI 链接错误 | torch-delegate 和 xLLM 的 libtorch 版本不一致 | 确保 `Torch_DIR` 指向同一 PyTorch |

### 请求阶段

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `Missing input tensor in MMData: x` | input_tensors 的 name 和图输入名不匹配 | 查看日志 `input_names_`，确保 name 一致 |
| `aclrtMalloc failed` | NPU 显存不足 | `npu-smi info` 检查，减小 batch |
| `RunModelWithStreamAsync failed` | 图执行错误（shape/dtype 不匹配） | 开启 `GLOG_v=1` 查看详细日志 |
| 500 错误，无具体信息 | brpc 内部异常 | `curl -v` 查看详细 HTTP 响应 |

### 响应阶段

| 现象 | 可能原因 | 解决 |
|------|----------|------|
| `output_tensors` 为空 | graph_outputs 序列化未命中 | 检查日志 `output_names_` 是否正确 |
| 输出数据全零 | 图执行成功但输出未正确读取 | 检查 `ge_to_torch` 是否正确转换 |
| `choices[0].text` 为空 | 正常行为，GE 图不输出文本 | 结果在 `output_tensors` 中 |

---

## 调试技巧

### 开启详细日志

```bash
# GLOG_v 是 glog 的 verbose 级别开关：
#   不设  = 默认（LOG(INFO) 输出，VLOG 不输出）
#   GLOG_v=1 = 额外输出 VLOG(1)（如 input_names 列表）
#   GLOG_v=2 = 额外输出 VLOG(2)
GLOG_v=1 ./build/xllm/core/server/xllm serve --model=../models/simple-ge --backend=ge
```

### 关键日志 grep

```bash
# 实时查看 GE 相关日志
./build/xllm/core/server/xllm serve ... 2>&1 | grep -E "EpModel|GE|graph|delegate"

# 查看图输入输出解析结果
GLOG_v=1 ./build/xllm/core/server/xllm serve ... 2>&1 | grep "input_names\|output_names"
```

### 检查 NPU 状态

```bash
# 查看 NPU 显存使用
watch -n 1 npu-smi info

# 查看进程占用
npu-smi info | grep xllm
```

### 检查动态库依赖

```bash
# 确认 xllm 链接了 torch_delegate
ldd ./build/xllm/core/server/xllm | grep torch_delegate

# 确认无 libtorch_cpu 冲突
ldd ./build/xllm/core/server/xllm | grep libtorch_cpu
ldd $TORCH_DELEGATE_ROOT/lib/libtorch_delegate_backend.so | grep libtorch_cpu
# 两者应指向同一个文件
```

---

## 验证清单

```
□ torch-delegate 编译成功，libtorch_delegate_backend.so 存在
□ xLLM 编译成功，无链接错误
□ 模型目录有 config.json + model.epair
□ CANN 环境已 source set_env.sh
□ TORCH_DELEGATE_ROOT 环境变量已设置
□ LD_LIBRARY_PATH 包含 torch-delegate lib 路径
□ xLLM 能启动，日志显示 "GE initialized successfully"
□ 日志显示 "EpModel loaded: N inputs, M outputs"
□ input_names_ 与请求中 input_tensors 的 name 一致
□ 发送请求后能收到 200 响应（不 crash）
□ 响应中 output_tensors 包含正确的 name + shape + 数据
□ 输出数据与 example 的 delegate_output_0.bin 精度一致
```

---

## 回退到 ATB 路径

如果 GE Graph 路径有问题，可快速回退：

```bash
# 只需改 backend 参数
./build/xllm/core/server/xllm serve \
  --model=../models/onerec \
  --backend=rec \
  --port=8080
```

无需重新编译，`USE_TORCH_DELEGATE` 编译选项不影响 ATB 路径。
