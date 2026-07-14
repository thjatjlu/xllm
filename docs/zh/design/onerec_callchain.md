# OneRec 端到端调用链详解

本文档描述 **OneRec（kOneRec）** 模型在 xLLM 框架中，从 `main()` 到最终 `OneRecModelImpl::forward()` 的**完整端到端调用链路**。  
覆盖：启动初始化、模型加载、HTTP 请求接入、请求构建、调度打包、引擎执行、Worker 前向、模型 forward 等完整阶段。

---

## 架构总览

xLLM 是一个 C++ 推理引擎，整体采用**分层编排 + 策略模式**架构：

```
┌────────────────────────────────────────────────────────────────────────┐
│  [API 层] APIService (brpc HTTP server)                                │
│  接收 /v1/completions、/v1/chat/completions 等请求                     │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │ 转发给具体 ServiceImpl
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  [Master 层] RecMaster                                                 │
│  职责: 接收请求，驱动调度循环                                            │
│  成员:                                                                 │
│    ├── pipeline_ : RecMasterPipeline (请求构建策略)                    │
│    ├── scheduler_ : FixedStepsScheduler (调度器，循环驱动 step)         │
│    ├── threadpool_ (请求构建线程池)                                     │
│    └── engine_ : RecEngine                                             │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  [Engine 层] RecEngine                                                 │
│  职责: 管理 device、Worker、KV Cache；执行 step 驱动推理                │
│  成员:                                                                 │
│    ├── pipeline_ : RecEnginePipeline (step 策略)                       │
│    ├── workers_ (Worker 实例)                                          │
│    ├── process_groups_ (多卡通信组)                                    │
│    ├── kv_cache_manager_ (block 分配器)                                │
│    └── tokenizer_, args_                                               │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  [Worker 层] Worker → RecWorkerImpl                                    │
│  职责: 在 device 上具体执行模型前向计算                                  │
│  成员:                                                                 │
│    ├── work_pipelines_ : vector<RecWorkPipeline> (device 执行策略)     │
│    │    └── RecPipelineRuntime { stream, model, executor, context }    │
│    ├── kv_caches_                                                      │
│    ├── model_ (CausalLM*)                                              │
│    └── model_executor_                                                 │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  [Executor 层] Executor → BaseExecutorImpl                             │
│  职责: 将 token/position/kv_cache/params 喂给模型，执行 forward         │
│  run() → model_->forward(...)                                          │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│  [Model 层] OneRec 模型栈                                              │
│  RecCausalLMImpl<OneRecForConditionalGeneration>                       │
│    → OneRecForConditionalGenerationImpl (RecForCausalLMImplBase 子类) │
│      → OneRecModelImpl (含 encoder_/decoder_ 为 OneRecStack)          │
│        → NpuOneRecBlockLayerImpl (每层 attention + FFN/MoE)           │
└────────────────────────────────────────────────────────────────────────┘
```

**Pipeline 策略说明**：OneRec 在 Master/Engine/Worker 各层都有对应的 Pipeline 子类，根据 `RecPipelineType` / `RecModelKind` 在工厂中创建具体实现。同一次运行中，三层会统一选择一致的 Pipeline 类型（PrefillOnly 或 XAttention）。

---

## 第一阶段：main → 启动与初始化

### 1.1 程序入口 `main()`
**文件：`xllm/xllm.cpp:481`**

```cpp
int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);     // 解析 gflags
    google::InitGoogleLogging("xllm");                     // 初始化日志
    initialize_configs();                                   // 注册 17 类 Config 单例

    // 校验 --model flag
    if (ModelConfig::get_instance().model().empty()) {
        HelpFormatter::print_error("--model flag is required");
        return 1;
    }

    return run();
}
```

### 1.2 配置初始化 `initialize_configs()`
**文件：`xllm/xllm.cpp:74`**

按固定顺序初始化 17 个 Config 单例，每个 Config 都通过 gflags 注册对应的命令行参数：
- `RecConfig`：rec 专用（`rec_worker_max_concurrency`、`enable_constrained_decoding`、`max_decode_rounds` 等）
- `ModelConfig`：`--model`、`--backend`、`--devices`
- `KVCacheConfig`：`--block_size`、`--max_cache_size`、`--max_memory_utilization`
- `SchedulerConfig`、`ParallelConfig`、`BeamSearchConfig` 等

### 1.3 主运行函数 `run()`
**文件：`xllm/xllm.cpp:329`**

核心步骤：
```cpp
int run() {
    // (1) 校验模型路径
    if (!std::filesystem::exists(model_config.model())) { LOG(FATAL) ... }

    // (2) 解析 model_id / backend（若未指定，从路径推断）
    //    backend 推断: model_path/.. 目录结构（如 "llm"/"vlm"/"rec"）

    // (3) 解析 model_type（从模型 config.json 的 model_type 字段读取，例如 "onerec"）
    std::string model_type = util::get_model_type(model_path, model_config.backend());
    //   对于 onerec: model_type == "onerec"

    // (4) 配置校验
    validate_config(model_type);   // 检查 backend 合法、device 等

    // (5) create_master
    Options options = create_options(host + ":" + port, is_local);
    master = create_master(model_config.backend(), options);
    //   若 backend=="rec" → new RecMaster(options)   [见 1.5]

    master->run();   // 启动调度循环 [见第四阶段]

    // (6) 启动 HTTP server
    std::vector<std::string> model_names = {model_config.model_id()};
    auto api_service = std::make_unique<APIService>(master.get(), ...);
    //   内部会创建 RecCompletionServiceImpl / ChatServiceImpl [见第二阶段]
    auto xllm_server = ServerRegistry::get_instance().register_server("HttpServer");
    xllm_server->start(std::move(api_service));
}
```

### 1.4 `create_master` 工厂
**文件：`xllm/core/distributed_runtime/master.cpp:483`**

```cpp
std::unique_ptr<Master> create_master(const std::string& backend,
                                       const Options& options) {
    if (backend == "llm")   return std::make_unique<LLMMaster>(options);
    if (backend == "vlm")   return std::make_unique<VLMMaster>(options);
    if (backend == "dit")   return std::make_unique<DiTMaster>(options);
    if (backend == "rec")   return std::make_unique<RecMaster>(options);  // ★
    LOG(FATAL) << "Failed to create master, backend is" << backend;
}
```

### 1.5 `Master::Master` 基类构造函数（engine_ 的创建）
**文件：`xllm/core/distributed_runtime/master.cpp:148`**

`Master::Master(options, type)` 根据 EngineType 构造对应的 `engine_`：

```cpp
Master::Master(const Options& options, EngineType type) : options_(options), engine_type_(type) {
    // ... 校验/打印 ...

    if (type == EngineType::REC) {
        // 创建 RecEngine
        runtime::Options eng_options;
        eng_options.model_path(options_.model_path())
                 .devices(devices)
                 .block_size(options_.block_size())
                 .max_cache_size(options_.max_cache_size())
                 .max_memory_utilization(options_.max_memory_utilization())
                 ...
                 .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());

        engine_ = std::make_unique<RecEngine>(eng_options);   // ★ 关键赋值
    }
}
```

`RecEngine` 是 `Engine` 的子类（见 `rec_engine.h:39`），持有：
- `pipeline_`：`RecEnginePipeline` 具体实现
- `workers_`：`std::vector<std::unique_ptr<Worker>>`
- `kv_cache_manager_`
- `tokenizer_`、`args_` 等

### 1.6 RecEngine::init()
**文件：`xllm/core/distributed_runtime/rec_engine.cpp:68`**

```cpp
bool RecEngine::init() {
    if (!init_model()) return false;                       // ① 创建模型+worker
    auto kv_cache_cap = estimate_kv_cache_capacity();     // ② 估算 KV cache 容量
    if (!allocate_kv_cache(kv_cache_cap)) return false;   // ③ 分配 KV cache
    return true;
}
```

#### 1.6.1 RecEngine::init_model()（核心：模型实例创建）
**文件：`xllm/core/distributed_runtime/rec_engine.cpp:84`**

```cpp
bool RecEngine::init_model() {
    // 1. 从模型目录加载 config
    auto model_loader = ModelLoader::create(model_path);
    tokenizer_  = model_loader->tokenizer();     // Tokenizer
    args_       = model_loader->model_args();    // ModelArgs（含 model_type="onerec"）
    quant_args_ = model_loader->quant_args();
    tokenizer_args_ = model_loader->tokenizer_args();

    // 2. 根据 model_type 选择 Pipeline 类型
    rec_model_kind_ = get_rec_model_kind(args_.model_type());  // kOneRec
    auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);
    pipeline_ = create_pipeline(pipeline_type, *this);

    // 3. 设置 Worker
    pipeline_->setup_workers();            // OneRec 本地 worker 无需 setup
    pipeline_->process_group_test();       // 测试进程组

    // 4. 计算 KV cache config
    n_local_kv_heads_ = std::max(1, n_kv_heads / world_size);
    head_dim_ = args_.head_dim();
    dtype_ = util::parse_dtype(args_.dtype(), devices[0]);

    // 5. ★ 关键: 在 Worker 上创建并加载模型
    return pipeline_->init_model_workers(model_path);
}
```

#### 1.6.2 OneRecLocalEnginePipeline::init_model_workers()

**文件：`xllm/core/distributed_runtime/rec_engine.cpp:571`**

这是 OneRec 专用的 Worker 初始化入口。对每个 device：
```cpp
bool OneRecLocalEnginePipeline::init_model_workers(const std::string& model_path) {
    // 1. 创建 ProcessGroup（多卡通信组，单卡时 rank=0, world_size=1）
    if (world_size == 1) {
        engine_.process_groups_.emplace_back(
            std::make_unique<ProcessGroup>(0, world_size, devices[0]));
    } else {
        // NPU 用 create_npu_process_groups，CUDA 用 create_local_process_groups
        engine_.process_groups_ = parallel_state::create_npu_process_groups(devices);
    }

    // 2. 为每个 rank 创建 Worker（实际持有 RecWorkerImpl）
    for (int32_t rank = 0; rank < world_size; ++rank) {
        ProcessGroup* pg = engine_.process_groups_[rank].get();
        ParallelArgs parallel_args(rank, world_size, pg);
        engine_.workers_.emplace_back(std::make_unique<Worker>(
            parallel_args, devices[rank], engine_.options_, WorkerType::REC));
    }

    // 3. 异步在所有 worker 上初始化模型
    for (auto& worker : engine_.workers_) {
        futures.emplace_back(worker->init_model_async(
            model_path,
            ExecutionConfig::get_instance().random_seed(),
            MasterStatus::WAKEUP));
    }
    auto results = folly::collectAll(futures).get();
    return true;
}
```

`Worker::init_model_async` 委托 `RecWorkerImpl::init_model(...)`（见 1.6.3）。

### 1.7 RecWorkerImpl::init_model — 模型实例化链路

这里分为两层：

#### 第一层：`RecWorkerImpl::init_model(weights_path, seed, master_status)`
**文件：`xllm/core/runtime/rec_worker_impl.cpp:2971`**

```cpp
bool RecWorkerImpl::init_model(const std::string& model_weights_path,
                                int32_t random_seed, MasterStatus master_status) {
    // 调用基类 WorkerImpl::init_model(...)
    if (!WorkerImpl::init_model(model_weights_path, random_seed, master_status)) {
        return false;
    }
    // EPLB 相关（可选）
    return true;
}
```

#### 中间层：`WorkerImpl::init_model(...)` 
**文件：`xllm/core/runtime/worker_impl.cpp:1276`**

```cpp
bool WorkerImpl::init_model(const std::string& model_weights_path, ...) {
    // (1) 解析 tokenizer / args / dtype
    auto model_loader = ModelLoader::create(model_weights_path);
    auto args = model_loader->model_args();
    torch::ScalarType dtype = util::parse_dtype(args.dtype(), device_);

    // (2) 构造 ModelContext（包含 parallel_args + args + quant_args + tensor_options）
    context_ = ModelContext(parallel_args_, args, quant_args, tensor_options);
    context_.set_model_id(options_.model_id());

    // (3) ★ 虚函数调用：this->init_model(context_)
    //     对于 RecWorkerImpl，实际调用 RecWorkerImpl::init_model(ModelContext& context)
    bool status = this->init_model(context_);

    // (4) 加载权重
    if (master_status == MasterStatus::WAKEUP) {
        this->load_model(std::move(model_loader));   // 触发 model_->load_model()
    }

    status_ = Status::LOADED;
    return true;
}
```

#### 第二层：`RecWorkerImpl::init_model(ModelContext& context)`（核心：模型+Pipeline+Executor 创建）
**文件：`xllm/core/runtime/rec_worker_impl.cpp:2990`**

```cpp
bool RecWorkerImpl::init_model(ModelContext& context) {
    const auto& model_type = context.get_model_args().model_type();
    rec_model_kind_ = get_rec_model_kind(model_type);   // kOneRec
    auto pipeline_type = get_rec_pipeline_type(rec_model_kind_);

    // 按并发度创建 pipeline（每个 pipeline 一个 model 实例）
    work_pipelines_.reserve(options_.rec_worker_max_concurrency());
    for (size_t i = 0; i < options_.rec_worker_max_concurrency(); ++i) {
        RecPipelineRuntime runtime(*this);
        runtime.stream = device_.get_stream_from_pool();
        runtime.context = std::make_unique<ModelContext>(...);

        if (rec_model_kind_ == RecModelKind::kOneRec) {
            // ★ 关键：通过 ModelRegistry 创建 OneRec 模型实例
            runtime.model = create_rec_model(*runtime.context.get());
        } else {
            runtime.model = create_llm_model(*runtime.context.get());
        }

        // 创建 Executor（包装 model + device + options）
        runtime.executor = std::make_unique<Executor>(
            runtime.model.get(),
            runtime.context->get_model_args(),
            runtime.worker.device(),
            runtime.worker.options_);

        // 创建 WorkPipeline
        work_pipelines_.emplace_back(create_pipeline(pipeline_type, runtime));
        index_queue_.enqueue(i);
    }

    model_.reset(work_pipelines_[0]->runtime().model.get());
    model_executor_.reset(work_pipelines_[0]->runtime().executor.get());
    return true;
}
```

### 1.8 `create_rec_model()` → ModelRegistry 工厂
**文件：`xllm/models/model_registry.cpp:355`**

```cpp
std::unique_ptr<CausalLM> create_rec_model(const ModelContext& context) {
    std::string resolved_name;
    // 解析注册名（例如 "onerec" → "onerec"）
    resolve_model_registration_name(context.get_model_args().model_type(),
                                    &resolved_name, nullptr);

    // 从 ModelRegistry 找注册的 factory
    auto factory = ModelRegistry::get_rec_model_factory(resolved_name);
    if (factory) return factory(context);
    LOG(ERROR) << "Unsupported rec model type: "
               << context.get_model_args().model_type();
    return nullptr;
}
```

**Factory 如何注册？** 在 `xllm/models/rec/npu/onerec.h:309`：
```cpp
REGISTER_REC_MODEL(onerec, OneRecForConditionalGeneration);
```

宏展开后（`xllm/models/model_registry.h:165`）：
```cpp
const bool onerec_rec_registered = []() {
    ModelRegistry::register_rec_model_factory("onerec",
        [](const ModelContext& context) -> std::unique_ptr<RecCausalLM> {
            auto model = OneRecForConditionalGeneration(context);  // 构造 torch module
            model->eval();
            return std::make_unique<RecCausalLMImpl<OneRecForConditionalGeneration>>(
                std::move(model), context.get_tensor_options());
        });
    return true;
}();
```

实际返回的对象类型：
```
RecCausalLMImpl<OneRecForConditionalGeneration>
  └── 持有: OneRecForConditionalGeneration （torch module）
              └── 持有: OneRecModelImpl （torch module，含 encoder_/decoder_）
```

此时 torch Module 已创建但权重未加载。

### 1.9 权重加载
**文件：`xllm/core/runtime/rec_worker_impl.cpp:3055`**

```cpp
void RecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
    // 第一个 pipeline 用原始 loader
    work_pipelines_[0]->runtime().model->load_model(std::move(loader));

    // 后续 pipeline（并发实例）各自创建 ModelLoader 加载
    for (size_t i = 1; i < work_pipelines_.size(); ++i) {
        auto model_loader = ModelLoader::create(model_weights_path);
        work_pipelines_[i]->runtime().model->load_model(std::move(model_loader));
    }
}
```

对 OneRec 模型（`RecCausalLMImpl::load_model` → `OneRecForConditionalGenerationImpl::load_model`）：

**文件：`xllm/models/rec/npu/onerec.h:262`**

```cpp
void OneRecForConditionalGenerationImpl::load_model(unique_ptr<ModelLoader> loader, ...) {
    for (const auto& state_dict : loader->get_state_dicts()) {
        // 支持 "module.module3.t5_model." 兼容前缀
        StateDict prefixed = state_dict->get_dict_with_prefix("module.module3.t5_model.");
        StateDict model_state_dict = prefixed.size() > 0
            ? prefixed
            : state_dict->get_dict_with_prefix("model.");
        if (model_state_dict.size() == 0) model_state_dict = *state_dict;

        model_->load_state_dict(model_state_dict);   // → OneRecModelImpl::load_state_dict

        // lm_head 权重（与 shared embedding tie 时共享）
        if (tie_word_embeddings_) {
            auto shared_dict = model_state_dict.get_dict_with_prefix("shared.");
            lm_head_->load_state_dict(shared_dict);
        } else {
            lm_head_->load_state_dict(state_dict->get_dict_with_prefix("lm_head."));
        }
    }
    model_->verify_loaded_weights();    // 校验是否所有权重都成功加载
    model_->merge_loaded_weights();     // 合并可融合权重（如 QKV）
}
```

**OneRecModelImpl::load_state_dict** 分发给 encoder_/decoder_/shared_（NPU 版）：
**`xllm/models/rec/npu/onerec.h:128`**

```cpp
void load_state_dict(const StateDict& state_dict) {
    shared_->load_state_dict(state_dict.get_dict_with_prefix("shared."));
    encoder_->load_state_dict(state_dict.get_dict_with_prefix("encoder."));
    decoder_->load_state_dict(state_dict.get_dict_with_prefix("decoder."));
}
```

`OneRecStack::load_state_dict` 按前缀分配给每个 block（`onerec_npu_impl.h:354`）。
`NpuOneRecBlockLayerImpl::load_state_dict` 加载到具体的 ATB op graph（`npu_onerec_block_layer_impl.cpp:1019`）。

### 1.10 KV Cache 分配
**文件：`xllm/core/distributed_runtime/rec_engine.cpp:206`**

```cpp
bool RecEngine::allocate_kv_cache(const KVCacheCapacity& kv_cache_cap) {
    // 构造 KVCacheShape（block 维度：n_layers × block_size × n_local_kv_heads × head_dim × 2(K+V)）
    const KVCacheShape kv_cache_shape(kv_cache_cap, args_, world_size);

    // 创建 BlockManagerPool：管理 block 的分配/释放
    kv_cache_manager_ = std::make_unique<BlockManagerPool>(options, dp_size_);

    // 让 pipeline 在每个 worker 上分配实际 KV cache 张量
    return pipeline_->allocate_kv_cache(kv_cache_shape);
}
```

### 1.11 RecMaster 构造：收尾

**文件：`xllm/core/distributed_runtime/rec_master.cpp:518`**

```cpp
RecMaster::RecMaster(const Options& options) : Master(options, EngineType::REC) {
    CHECK(engine_->init());  // ★ 完成上面 1.6-1.10 的全部模型初始化

    model_args_ = engine_->model_args();
    rec_type_ = get_rec_type(model_args_);   // kOneRec

    // 创建 scheduler
    ContinuousScheduler::Options scheduler_options;
    scheduler_options.max_tokens_per_batch(options_.max_tokens_per_batch())
                     .max_seqs_per_batch(options_.max_seqs_per_batch())
                     ...
                     .rec_worker_max_concurrency(options_.rec_worker_max_concurrency());
    scheduler_ = create_fixed_steps_scheduler(engine_.get(), scheduler_options);

    threadpool_ = std::make_unique<ThreadPool>(
        options_.num_request_handling_threads(), false, "RecMaster.request");

    // 创建 Master 层 Pipeline 策略
    auto rec_model_kind = get_rec_model_kind(model_args_.model_type());
    auto pipeline_type = get_rec_pipeline_type(rec_model_kind);
    pipeline_ = create_pipeline(pipeline_type, *this);
    // OneRec → OneRecPrefillOnlyMasterPipeline 或 OneRecXAttentionMasterPipeline
}
```

### 1.12 启动调度循环
**文件：`xllm/core/distributed_runtime/rec_master.cpp:585`**

```cpp
void RecMaster::run() {
    running_.store(true);
    loop_thread_ = std::thread([this]() {
        const auto timeout = absl::Milliseconds(5);
        while (!stopped_.load()) {
            scheduler_->step(timeout);   // ★ 驱动调度+推理
        }
        running_.store(false);
    });
}
```

此时服务端已就绪，等待 HTTP 请求。

---

## 第二阶段：HTTP 请求接入

### 2.1 `APIService::Completions(...)`
**文件：`xllm/api_service/api_service.cpp:144`**

```cpp
void APIService::Completions(::google::protobuf::RpcController* controller,
                              const CompletionRequest* request,
                              CompletionResponse* response, Closure* done) {
    // 若是 REC 模式
    if (rec_completion_service_impl_) {
        auto arena = GetArenaWithCheck<CompletionCall>(response);
        std::shared_ptr<Call> call = std::make_shared<CompletionCall>(
            ctrl, done_guard.release(), request, response, ...);
        rec_completion_service_impl_->process_async(call);   // ★
    }
}
```

### 2.2 `RecCompletionServiceImpl::process_async_impl()`
**文件：`xllm/api_service/rec_completion_service_impl.cpp:267`**

```cpp
void RecCompletionServiceImpl::process_async_impl(std::shared_ptr<CompletionCall> call) {
    const auto& rpc_request = call->request();
    const auto& model = rpc_request.model();

    // 1. 检查 model 是否支持
    if (!models_.contains(model)) {
        call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
        return;
    }

    // 2. 限流检查
    if (master_->get_rate_limiter()->is_limited()) {
        call->finish_with_error(StatusCode::RESOURCE_EXHAUSTED, ...);
        return;
    }

    // 3. 解析 RequestParams
    RequestParams request_params(rpc_request, call->get_x_request_id(), ...);

    // 4. 解析 prompt_tokens / input_tensors
    std::optional<std::vector<int>> prompt_tokens = std::nullopt;
    if (rpc_request.has_routing()) {
        prompt_tokens = std::vector<int>();
        for (int i = 0; i < rpc_request.token_ids_size(); i++)
            prompt_tokens->emplace_back(rpc_request.token_ids(i));
    }
    auto input_tensors = /* 从 rpc 解析 */ ...;

    // 5. 调用 RecMaster::handle_request
    master_->handle_request(
        rpc_request.prompt(),
        std::move(prompt_tokens),
        std::move(input_tensors),
        std::move(request_params),
        /*callback=*/[call, model, master, stream, ...]
            (const RequestOutput& req_output) mutable -> bool {
                // 这里写响应给客户端
                ...
            });
}
```

### 2.3 `RecMaster::handle_request(...)`
**文件：`xllm/core/distributed_runtime/rec_master.cpp:613`**

```cpp
void RecMaster::handle_request(std::string prompt,
                                std::optional<std::vector<int>> prompt_tokens,
                                std::optional<std::vector<proto::InferInputTensor>> input_tensors,
                                RequestParams sp, OutputCallback callback) {
    // 转发到 pipeline 的 generate_request + schedule_request
    schedule_request(std::move(sp), std::move(callback),
        [this, prompt = std::move(prompt),
         prompt_tokens = std::move(prompt_tokens),
         input_tensors = std::move(input_tensors)](
            const RequestParams& params, OutputCallback cb) mutable {
            return pipeline_->generate_request(
                std::move(prompt), std::move(prompt_tokens),
                std::move(input_tensors), params, std::move(cb));
        });
}
```

---

## 第三阶段：请求构建（Master 层 Pipeline）

### 3.1 `RecMaster::schedule_request()`
**文件：`xllm/core/distributed_runtime/rec_master.cpp:713`**

```cpp
void RecMaster::schedule_request(RequestParams sp, OutputCallback callback,
                                  RequestBuilder build_request) {
    scheduler_->incr_pending_requests(1);

    // 包装 callback，加入 log 计量
    auto cb = [callback = std::move(callback),
               scheduler = scheduler_.get()](const RequestOutput& output) {
        output.log_request_status();
        return callback(output);
    };

    // 异步构建请求
    threadpool_->schedule([this, sp = std::move(sp),
                           callback = std::move(cb),
                           build_request = std::move(build_request)]() mutable {
        SCOPE_GUARD([this] { scheduler_->decr_pending_requests(); });

        Timer timer;
        if (!sp.verify_params(callback)) return;  // 参数校验

        // ★ 调用 pipeline->generate_request，构造 Request 对象
        auto request = build_request(sp, std::move(callback));
        if (!request) return;

        // 提交给调度器
        if (!scheduler_->add_request(request)) {
            CALLBACK_WITH_ERROR(StatusCode::RESOURCE_EXHAUSTED,
                                "No available resources to schedule request");
        }
    });
}
```

### 3.2 OneRec Pipeline `generate_request()`

**`OneRecPrefillOnlyMasterPipeline::generate_request`** 或 **`OneRecXAttentionMasterPipeline::generate_request`**

**文件：`xllm/core/distributed_runtime/rec_master.cpp:466`、`:481`**

两者都委托给基类 `generate_onerec_request_common`：

```cpp
std::shared_ptr<Request> OneRecPrefillOnlyMasterPipeline::generate_request(
    std::string prompt,
    std::optional<std::vector<int>> prompt_tokens,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp, OutputCallback callback) {
    return generate_onerec_request_common(
        std::move(prompt), std::move(prompt_tokens),
        std::move(input_tensors), sp, std::move(callback),
        /*build_stop_checker=*/false);    // PrefillOnly: 无 stop 检查
    // 注意：OneRecXAttentionMasterPipeline 用 true
}
```

### 3.3 `generate_onerec_request_common()`
**文件：`xllm/core/distributed_runtime/rec_master.cpp:338`**

```cpp
std::shared_ptr<Request> generate_onerec_request_common(
    std::string prompt, ...,
    std::optional<std::vector<proto::InferInputTensor>> input_tensors,
    const RequestParams& sp, OutputCallback callback,
    bool build_stop_checker) {
    std::vector<int32_t> local_prompt_tokens;
    MMData processed_mm_data;

    // 处理 OneRec 输入：sparse_embedding、decoder_context_embedding
    if (!process_onerec_inputs(prompt_tokens, input_tensors,
                                master_.model_args_,
                                &local_prompt_tokens,
                                &processed_mm_data, callback)) {
        return nullptr;
    }

    // 构造 Request
    return master_.build_request_common(
        std::move(prompt), std::move(local_prompt_tokens),
        std::move(processed_mm_data), sp, std::move(callback),
        build_stop_checker);
}
```

`process_onerec_inputs`（`rec_master.cpp:74`）：
- 若传了 `input_tensors`，要求包含 `sparse_embedding` 张量，可选 `decoder_context_embedding`
- 校验 shape / dtype / numel，转换为 `torch::Tensor`（`bf16`）放入 `MMData`

### 3.4 `RecMaster::build_request_common()`
**文件：`xllm/core/distributed_runtime/rec_master.cpp:747`**

```cpp
std::shared_ptr<Request> RecMaster::build_request_common(
    std::string prompt, std::vector<int32_t> prompt_tokens,
    MMData mm_data, const RequestParams& sp,
    OutputCallback callback, bool build_stop_checker) {
    // (1) 长度检查
    int32_t max_context_len = model_args_.max_position_embeddings();
    if (prompt_tokens.size() >= max_context_len) { ... }

    uint32_t max_tokens = sp.max_tokens;
    size_t capacity = prompt_tokens.size() + max_tokens + ...;

    // (2) 构造采样参数
    RequestSamplingParam sampling_param;
    sampling_param.frequency_penalty = sp.frequency_penalty;
    sampling_param.temperature       = sp.temperature;
    sampling_param.top_p             = sp.top_p;
    sampling_param.top_k             = sp.top_k;
    sampling_param.beam_width        = sp.beam_width;
    sampling_param.num_return_sequences = sp.num_return_sequences;
    ...

    // (3) 构造 stopping_checker
    StoppingChecker stopping_checker;
    if (build_stop_checker) {
        stopping_checker = StoppingChecker(max_tokens, max_context_len,
            model_args_.eos_token_id(), sp.ignore_eos, stop_tokens, ...);
    }

    // (4) 构造 RequestState + Request 对象
    RequestState req_state(std::move(prompt), std::move(prompt_tokens),
                           std::move(mm_data), std::move(sampling_param),
                           std::move(stopping_checker), ...);

    auto request = std::make_shared<Request>(std::move(req_state),
                                              std::move(callback));
    // Request 内含 1 个或多个 Sequence（best_of / n）
    request->expand_sequences(/*is_prefill=*/false);
    return request;
}
```

`Request` 对象封装了：
- `sequences_`：每个请求至少一个 `Sequence`
- `prompt_tokens_`、`mm_data_`、`sampling_param_`
- `callback_`（用于将结果传回 API 层）

### 3.5 入队

`scheduler_->add_request(request)` 将请求写入 `request_queue_`（`folly::MPMCQueue`）。
- **文件：`xllm/core/scheduler/fixed_steps_scheduler.cpp:55`**

```cpp
bool FixedStepsScheduler::add_request(std::shared_ptr<Request>& request) {
    return request_queue_.write(request);   // 无锁入队
}
```

---

## 第四阶段：调度（Scheduler 循环）

### 4.1 `FixedStepsScheduler::step(timeout)`
**文件：`xllm/core/scheduler/fixed_steps_scheduler.cpp:343`**

被 `RecMaster::run()` 的 loop_thread 每 5ms 调用一次。

```cpp
void FixedStepsScheduler::step(const absl::Duration& timeout) {
    if (!options_.enable_schedule_overlap()) {
        // ① 构造一个 batch（可能阻塞等待 timeout）
        ScheduleResult result = schedule_request(timeout);

        // ② 检查 batch 是否为空
        bool all_empty = std::all_of(
            result.batches.begin(), result.batches.end(),
            [](const Batch& one) { return one.empty(); });
        if (all_empty) return;   // 没活干，本轮 step 返回

        // ③ 构造 function lambda（执行推理 + 后处理）
        auto function = [this,
                         batches = std::move(result.batches),
                         requests = std::move(result.requests),
                         sequences = std::move(result.sequences)]() mutable {
            // 3.1 调用 engine 执行一次推理（一次前向）
            engine_->step(batches);

            // 3.2 后处理：更新请求状态 + 释放已完成请求
            std::vector<std::shared_ptr<Request>> finished_requests;
            for (auto& request : requests) {
                if (request) {
                    request->update_connection_status();
                    if (request->finished() || request->cancelled()) {
                        kv_cache_manager_->deallocate(request.get());
                        finished_requests.emplace_back(request);
                    }
                }
            }
            // 3.3 通过 ResponseProcessor 把结果回写 HTTP response
            if (!finished_requests.empty()) {
                response_processor_->process_completed_requests(finished_requests);
            }

            if (options_.rec_worker_max_concurrency() > 1) {
                step_semaphore_.release();
            }
        };

        // ④ 执行 function
        if (options_.rec_worker_max_concurrency() > 1) {
            step_semaphore_.acquire();
            step_threadpool_->schedule(function);   // 异步提交，立即返回
        } else {
            function();                              // 同步执行，等推理完成
        }
    }
}
```

### 4.2 `schedule_request(timeout)`（等待 + 打包 batch）
**文件：`xllm/core/scheduler/fixed_steps_scheduler.cpp:311`**

```cpp
ScheduleResult FixedStepsScheduler::schedule_request(const absl::Duration& timeout) {
    const auto deadline = absl::Now() + timeout;
    ScheduleResult result;
    while (true) {
        result.batches = prepare_batch();
        bool all_empty = std::all_of(result.batches.begin(), result.batches.end(),
                                      [](const Batch& b) { return b.empty(); });
        if (!all_empty) {
            // 把 running_requests_ / running_sequences_ 搬到 result
            result.requests = std::move(running_requests_);
            result.sequences = std::move(running_sequences_);
            return result;
        }
        const auto now = absl::Now();
        if (now > deadline) break;
        absl::SleepFor(absl::Milliseconds(1));   // 等 1ms 重试
    }
    return result;
}
```

### 4.3 `prepare_batch()`
**文件：`xllm/core/scheduler/fixed_steps_scheduler.cpp:192`**

核心打包流程：

```cpp
std::vector<Batch> FixedStepsScheduler::prepare_batch() {
    // (1) 把 request_queue_ 中的新请求搬到 waiting_priority_queue_
    std::shared_ptr<Request> request;
    while (request_queue_.read(request)) {
        if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
            waiting_priority_queue_->push(request);  // 新请求 → 等待队列
        } else {
            running_requests_.emplace_back(request);  // 已被 prefill 的请求（disagg-pd）
        }
    }

    // (2) 把 finished/cancelled 的请求清理掉
    for (auto it = running_requests_.rbegin(); ...) {
        request->update_connection_status();
        if (request->finished() || request->cancelled()) {
            if (scheduler_pipeline_->requires_kv_cache()) {
                kv_cache_manager_->deallocate(request.get());
            }
            finished_requests.emplace_back(request);
            *it = nullptr;
        }
    }
    running_requests_.clear();
    running_sequences_.clear();
    running_sequences_budgets_.clear();
    // （处理 finished_requests）

    // (3) 懒初始化 SchedulerPipeline（按 rec_type/mode 创建）
    if (!scheduler_pipeline_ && !waiting_priority_queue_->empty()) {
        auto rec_type = waiting_priority_queue_->top()->state().rec_type;
        scheduler_pipeline_ = create_scheduler_pipeline(rec_type, is_multi_round);
        // OneRec + xattention → OneRecXAttentionSchedulerPipeline
        // OneRec 普通      → OneRecSchedulerPipeline
    }

    // (4) 调度 prefill 请求：从 waiting 中选出，分配 KV cache，加入 running
    handle_prefill_requests(remaining_token_budget, remaining_seq_budget,
                             finished_requests);

    // (5) 构造 Batch 对象
    return scheduler_pipeline_->create_batches(*this, batch_factory);
    //   OneRec 普通 → create_rec_batches
    //   XAttention → create_rec_batches（同样）
}
```

`handle_prefill_requests`（`fixed_steps_scheduler.cpp:68`）：
- 从 `waiting_priority_queue_` 中按优先级 pop 请求
- 分配 KV cache block（`scheduler_pipeline_->allocate_kv_cache(kv_cache_manager_, seq)`）
- 将 sequence 加入 `running_sequences_`
- 直到 token budget 或 seq budget 用尽

---

## 第五阶段：引擎执行（RecEngine Pipeline）

### 5.1 `RecEngine::step()` 
**文件：`xllm/core/distributed_runtime/rec_engine.cpp:233`**

```cpp
ForwardOutput RecEngine::step(std::vector<Batch>& batches) {
    return pipeline_->step(batches);   // 虚函数分发
}
```

`pipeline_` 在 `RecEngine::init_model()` 中已创建（见 1.6.1）。

### 5.2 OneRec PrefillOnly 路径
**`OneRecPrefillOnlyEnginePipeline::step()`**
**文件：`xllm/core/distributed_runtime/rec_engine.cpp:697`**

OneRec 默认模式：一次 step 内完成 prefill + 多轮 decode。

```cpp
ForwardOutput OneRecPrefillOnlyEnginePipeline::step(std::vector<Batch>& batches) {
    if (engine_.workers_.empty()) return {};

    // (1) 在 rank-0 worker 上构造 ForwardInput
    auto forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
    if (!forward_inputs.token_ids.defined()) return {};

    // (2) prefill：跑一次完整 encoder + decoder
    const auto& prefill_output = get_model_output(forward_inputs);
    batches[0].process_sample_output(prefill_output.sample_output, false);

    // (3) 多轮 decode：kRecDecodeSteps 次
    ForwardOutput decode_output;
    for (size_t i = 0; i < kRecDecodeSteps; ++i) {
        forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
        decode_output = get_model_output(forward_inputs);
        batches[0].process_sample_output(decode_output.sample_output, false,
                                         /*force_requested=*/i+1 == kRecDecodeSteps);
    }

    batches[0].finish();    // 触发 request->finished() 标志
    return decode_output;
}
```

`get_model_output`（`rec_engine.cpp:758`）：
```cpp
ForwardOutput get_model_output(const ForwardInput& model_inputs) {
    std::vector<folly::SemiFuture<std::optional<ForwardOutput>>> futures;
    for (auto& worker : engine_.workers_) {
        futures.emplace_back(worker->step_async(model_inputs));   // ★ 异步提交
    }
    auto results = folly::collectAll(futures).get();              // ★ 等待所有 worker
    // 校验结果，合并输出
    return merge_outputs(results);
}
```

### 5.3 OneRec XAttention 路径
**`OneRecXAttentionEnginePipeline::step()`**
**文件：`xllm/core/distributed_runtime/rec_engine.cpp:841`**

与 PrefillOnly 类似但只跑一次 step（多轮 beam search 在 worker 内完成）：

```cpp
ForwardOutput OneRecXAttentionEnginePipeline::step(std::vector<Batch>& batches) {
    auto forward_inputs = engine_.workers_[0]->prepare_inputs(batches[0]);
    if (!forward_inputs.token_ids.defined()) return {};

    const auto& output = get_model_output(forward_inputs);
    // 处理 beam_sequence_group 或普通 sample_output
    if (output.beam_sequence_group.defined()) {
        batches[0].process_beam_sequence_group(output);
    } else {
        batches[0].process_sample_output(output.sample_output, false);
    }
    batches[0].finish();
    return output;
}
```

---

## 第六阶段：Worker 执行（device 上前向）

> **注意**：输入数据在此阶段从 Host(CPU) 传输到 Device(NPU)。完整的 H2D 传输链路详见 [附录 A：输入数据流转与 H2D 传输](#附录-a输入数据流转与-h2d-传输)。

### 6.1 `Worker::step_async()` → `RecWorkerImpl::step_async()`
**文件：`xllm/core/runtime/worker.cpp:136`**

```cpp
folly::SemiFuture<std::optional<ForwardOutput>>
Worker::step_async(const ForwardInput& inputs) {
    return impl_->step_async(inputs);   // impl_ 是 RecWorkerImpl
}
```

**文件：`xllm/core/runtime/rec_worker_impl.cpp:3151`**

```cpp
folly::SemiFuture<std::optional<ForwardOutput>>
RecWorkerImpl::step_async(const ForwardInput& input) {
    folly::Promise<std::optional<ForwardOutput>> promise;
    auto future = promise.getSemiFuture();

    size_t index;
    index_queue_.wait_dequeue(index);   // 从并发池中取一个 pipeline slot

    // 异步在 step_threadpool_ 上执行
    step_threadpool_->schedule_with_tid(
        [this, &input, index, promise = std::move(promise)]() mutable {
            auto stream_guard = work_pipelines_[index]->runtime().stream->set_stream_guard();

            ForwardInput input_on_device;
            // 1. 把 input 拷到 device、准备 attention mask 等
            work_pipelines_[index]->prepare_work_before_execute(input, input_on_device);

            // 2. 调用 pipeline 的 step 做实际模型推理
            const auto output = work_pipelines_[index]->step(input_on_device);
            promise.setValue(output);

            index_queue_.enqueue(index);  // 归还 slot
        }, index);
    return future;
}
```

### 6.2 `OneRecWorkPipeline::step()`（实际前向）
**文件：`xllm/core/runtime/rec_worker_impl.cpp:715`**

这是 OneRec 模型真正跑 forward 的地方。核心逻辑：

```cpp
std::optional<ForwardOutput> OneRecWorkPipeline::step(const ForwardInput& input) {
    Timer timer;
    runtime_.worker.device_.set_device();

    ForwardInput& mutable_input = const_cast<ForwardInput&>(input);
    auto* onerec_params = mutable_input.input_params.onerec_params();

    // 定义 run_onerec_forward：设置标志 → executor->forward
    auto run_onerec_forward = [&](const torch::Tensor& token_ids,
                                   const torch::Tensor& positions,
                                   bool is_encoder_forward,
                                   bool forward_has_encoder_output,
                                   bool is_hybrid_mode) {
        mutable_onerec_params.is_encoder_forward = is_encoder_forward;
        mutable_onerec_params.has_encoder_output = forward_has_encoder_output;
        mutable_onerec_params.is_hybrid_mode = is_hybrid_mode;
        return runtime_.executor->forward(token_ids, positions,
                                           runtime_.worker.kv_caches_,
                                           mutable_input.input_params);
    };

    torch::Tensor hidden_states;
    auto rec_stage = onerec_params->rec_stage;

    // ============ PREFILL ============
    if (rec_stage == PREFILL) {
        if (onerec_params->is_first_prefill) {
            // (1a) 跑 encoder（使用 encoder_token_ids 或 encoder_sparse_embedding）
            torch::Tensor encoder_tokens = has_sparse_embedding
                ? onerec_params->encoder_sparse_embedding
                : onerec_params->encoder_token_ids;
            auto encoder_output = run_onerec_forward(
                encoder_tokens, onerec_params->encoder_positions,
                /*is_encoder_forward=*/true, false, false);
            onerec_params->encoder_output = encoder_output.hidden_states;

            // (1b) 跑 decoder
            auto model_output = run_onerec_forward(
                mutable_input.token_ids, mutable_input.positions,
                /*is_encoder_forward=*/false,
                /*forward_has_encoder_output=*/true, false);
            hidden_states = model_output.hidden_states;
        } else {
            // (1c) 后续 prefill（直接用 cached encoder output）
            auto model_output = run_onerec_forward(
                mutable_input.token_ids, mutable_input.positions,
                false, /*has_encoder_output=*/true, false);
            hidden_states = model_output.hidden_states;
        }
    }
    // ============ DECODE ============
    else {
        auto model_output = run_onerec_forward(
            mutable_input.token_ids, mutable_input.positions,
            false, onerec_params->has_encoder_output, false);
        hidden_states = model_output.hidden_states;
    }

    // (2) 算 logits
    torch::Tensor logits = runtime_.model->logits(
        hidden_states, mutable_input.sampling_params.selected_token_idxes);

    // (3) 采样（可选 constrained decoding）
    SampleOutput sample_output;
    if (constrained_decoding_ && ...) {
        // 约束解码：用 filter_mask/trie
    } else {
        sample_output = rec_sampler_->sample(logits, mutable_input.sampling_params, ...);
    }

    return ForwardOutput{hidden_states, sample_output, logits};
}
```

`OneRecXAttentionWorkPipeline::step()`（`rec_worker_impl.cpp:1345`）类似但包含多轮 beam search 解码循环：每轮跑一次 executor forward + beam search kernel + cache select。

---

## 附录 A：输入数据流转与 H2D 传输

OneRec 模型的输入数据（token_ids、encoder_sparse_embedding、decoder_context_embedding 等）在整个处理链路中**始终是 Host(CPU) tensor**，直到 Worker 层执行前向之前才统一做一次 H2D（Host-to-Device）传输。

### A.1 数据流转全链路

```
HTTP 请求 (protobuf fp32_contents, 纯 host 内存)
  │
  │  rec_completion_service_impl.cpp:313  拷贝 proto
  │  rec_master.cpp:180-182  convert_rec_tensor_to_torch() + .to(kBFloat16)
  ▼
CPU BFloat16 torch::Tensor (从 proto clone 出来, 独立内存)
  │  存入 MMData → RequestState → Sequence.mm_data_
  ▼
Scheduler 打包 Batch (CPU 上)
  │  onerec_batch_input_builder.cpp:697-705  token_ids → CPU pinned Int32
  │  onerec_batch_input_builder.cpp:944-948  encoder_sparse_embedding → CPU BFloat16 (torch::cat)
  │  onerec_batch_input_builder.cpp:950-1007 decoder_context_embedding → CPU BFloat16
  ▼
ForwardInput (全部 CPU tensor, pinned memory 加速后续 H2D)
  │
  │  ★ rec_worker_impl.cpp:424  ← H2D 发生在这里
  │  processed_inputs = inputs.to(runtime_.worker.device(), runtime_.worker.dtype())
  │
  │  内部级联调用:
  │    forward_params.h:452      token_ids.to(device)           → NPU
  │    forward_params.h:453      positions.to(device)           → NPU
  │    model_input_params.h:899  onerec->to(device)
  │      model_input_params.h:86   encoder_sparse_embedding.to(device)  → NPU
  │      model_input_params.h:89   decoder_context_embedding.to(device) → NPU
  │      model_input_params.h:101  encoder_token_ids.to(device)         → NPU
  ▼
OneRecModelImpl::forward() 收到的全部是 device tensor
```

### A.2 各阶段数据位置

| 阶段 | 数据位置 | 说明 |
|------|----------|------|
| API 层（proto 解析） | Host | protobuf `fp32_contents`，纯内存 |
| RecMaster（请求构建） | Host | `convert_rec_tensor_to_torch` 创建 CPU BFloat16 tensor |
| Scheduler（Batch 打包） | Host | pinned memory CPU tensor，为 H2D 优化 |
| **Worker `prepare_work_before_execute()`** | **Host → Device** | **`inputs.to(device)` 是唯一 H2D 点** |
| OneRecModelImpl::forward() | Device | 全部 NPU tensor |

### A.3 Proto → CPU Tensor（RecMaster 层）

**文件：`xllm/core/util/utils.cpp:179`**

```cpp
torch::Tensor convert_rec_tensor_to_torch(const proto::InferInputTensor& input_tensor) {
    const auto& data = contents.fp32_contents();
    return torch::from_blob(
               const_cast<float*>(data.data()),
               shape,
               torch::dtype(torch::kFloat32).requires_grad(false))
        .clone();  // clone 确保独立内存，proto 释放后不受影响
}
```

**文件：`xllm/core/distributed_runtime/rec_master.cpp:180-182`**

```cpp
mm_dict[tensor_name] =
    util::convert_rec_tensor_to_torch(tensor).to(torch::kBFloat16);
// 结果: CPU BFloat16 tensor，存入 MMData(MMType::EMBEDDING, mm_dict)
```

### A.4 Batch 打包（Scheduler 层，CPU pinned memory）

**文件：`xllm/core/framework/batch/onerec_batch_input_builder.cpp`**

token_ids 使用 pinned memory 创建，为后续 H2D 做 DMA 优化：
```cpp
// line 697-705
forward_input.token_ids =
    torch::empty({static_cast<int64_t>(flatten_tokens_vec.size())},
                 torch::TensorOptions()
                     .dtype(torch::kInt)
                     .device(torch::kCPU)
                     .pinned_memory(true));
std::memcpy(forward_input.token_ids.data_ptr<int>(),
            flatten_tokens_vec.data(),
            flatten_tokens_vec.size() * sizeof(int));
```

encoder_sparse_embedding 通过 `torch::cat` 把多个 Sequence 的 CPU tensor 拼接：
```cpp
// line 944-948
onerec_params.encoder_sparse_embedding =
    torch::cat(perf_cache.cache_data.encoder_sparse_embeddings, /*dim=*/0);
// 结果: CPU BFloat16 tensor
```

### A.5 H2D 传输点（Worker 层，唯一入口）

**文件：`xllm/core/runtime/rec_worker_impl.cpp:403-457`**

```cpp
void RecWorkerImpl::RecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
    // ★ 唯一 H2D 入口
    processed_inputs =
        inputs.to(runtime_.worker.device(), runtime_.worker.dtype());
    // ...
}
```

**文件：`xllm/core/runtime/forward_params.h:418-465`** — `ForwardInput::to()` 实现：

```cpp
ForwardInput to(const torch::Device& device, torch::ScalarType dtype) const {
    ForwardInput inputs;
    set_host_views(inputs);
    inputs.token_ids = safe_to(source_token_ids, device, true);    // H2D
    inputs.positions = detail::normalize_positions_for_device(
        safe_to(source_positions, device, true));                   // H2D
    inputs.input_params = input_params.to(device);                  // H2D 级联
    inputs.sampling_params = sampling_params.to(device, dtype);     // H2D
    inputs.device_tensors_ready = true;
    return inputs;
}
```

**文件：`xllm/core/framework/model/model_input_params.h:79-108`** — `OneRecModelInputParams::to()` 逐 tensor 搬到 device：

```cpp
OneRecModelInputParams to(const c10::Device& device) const {
    OneRecModelInputParams result = *this;
    result.encoder_seq_lens_tensor    = encoder_seq_lens_tensor.to(device);
    result.encoder_sparse_embedding   = encoder_sparse_embedding.to(device);    // ★ H2D
    result.decoder_context_embedding  = decoder_context_embedding.to(device);   // ★ H2D
    result.cross_attn_kv_cu_seq_lens  = cross_attn_kv_cu_seq_lens.to(device);
    result.cross_attn_new_cache_slots = cross_attn_new_cache_slots.to(device);
    result.cross_attn_block_tables    = cross_attn_block_tables.to(device);
    result.encoder_token_ids          = encoder_token_ids.to(device);           // ★ H2D
    result.encoder_positions          = encoder_positions.to(device);           // ★ H2D
    return result;
}
```

### A.6 设计意图

这种 **"CPU 侧打包、device 侧一次性 H2D"** 的设计有几个好处：

1. **解耦调度与 device 流**：Scheduler 在 CPU 上打包 batch 不阻塞 device stream，CPU 侧可以并行做请求调度、KV cache 分配等逻辑
2. **Pinned memory 优化**：CPU tensor 用 `pinned_memory(true)` 创建，H2D 传输走 DMA，带宽更高
3. **最小化 H2D 次数**：所有 tensor 在 `prepare_work_before_execute()` 中一次性搬到 device，避免多次零散传输
4. **Host view 保留**：`ForwardInput` 保留 `token_ids_host` / `positions_host` 字段，供后续 CPU 侧逻辑（如采样后处理）直接读取，无需 D2H

### A.7 到达模型时各 tensor 状态

| Tensor | Device | Dtype | Shape |
|--------|--------|-------|-------|
| `token_ids` | NPU | Int32 | `[total_tokens]` (flattened) |
| `positions` | NPU | Int32 | `[1]` (固定 `{0}`) |
| `onerec_params.encoder_sparse_embedding` | NPU | BFloat16 | `[total_encoder_len, hidden_size]` |
| `onerec_params.decoder_context_embedding` | NPU | BFloat16 | `[bs, group_width, ctx_len+seq_len, hidden]` |
| `onerec_params.encoder_token_ids` | NPU | Int32 | `[total_encoder_tokens]` |
| `onerec_params.encoder_positions` | NPU | Int32 | `[1]` |
| `sampling_params.selected_token_idxes` | NPU | Int64 | `[num_sequences]` |

---

## 第七阶段：模型前向（Executor → OneRec）

### 7.1 `Executor::forward()` 
**文件：`xllm/core/runtime/executor.cpp:39`**

```cpp
ModelOutput Executor::forward(const torch::Tensor& tokens,
                               const torch::Tensor& positions,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& params) {
    return impl_->run(tokens, positions, kv_caches, params);
}
```

### 7.2 `BaseExecutorImpl::run()`
**文件：`xllm/core/runtime/base_executor_impl.cpp:35`**

```cpp
ModelOutput BaseExecutorImpl::run(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelInputParams& params) {
    return model_->forward(tokens, positions, kv_caches, params);
    // model_ 是 CausalLM*，实际是 RecCausalLMImpl<...>
}
```

### 7.3 类层次分发

```
BaseExecutorImpl::run
  └── model_->forward()
        │
        ▼ RecCausalLMImpl<OneRecForConditionalGeneration>::forward()
        │ [rec_causal_lm.h:33]
        │   return model_->forward(tokens, positions, kv_caches, parameters);
        ▼
        OneRecForConditionalGenerationImpl::forward()
        │ 继承自 RecForCausalLMImplBase [rec_model_base.h:49]
        │   return model_->forward(tokens, positions, kv_caches, input_params);
        ▼
 ★ OneRecModelImpl::forward(tokens, positions, kv_caches, input_params)
        │ [xllm/models/rec/npu/onerec.h:48]
```

### 7.4 `OneRecModelImpl::forward()`（NPU 版）
**文件：`xllm/models/rec/npu/onerec.h:48`**

```cpp
ModelOutput OneRecModelImpl::forward(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& input_params) {
    if (!tokens.defined()) return ModelOutput();

    const auto* onerec_params = input_params.onerec_params();

    // ====== Encoder 分支 ======
    if (onerec_params->is_encoder_forward) {
        std::vector<KVCache> encoder_kv_caches;   // encoder 不用 KV cache
        auto encoder_output = encoder_(tokens, positions, encoder_kv_caches, input_params);
        // encoder_ 类型是 OneRecStack（见 7.4.1）

        // pad 到固定长度，便于后续 decode 阶段复用
        torch::Tensor cached = pad_encoder_output(encoder_output, input_params);
        {
            std::lock_guard lock(encoder_output_mutex_);
            encoder_output_ = cached;
        }
        return ModelOutput(cached);
    }

    // ====== Decoder 分支 ======
    torch::Tensor cached_encoder_output;
    if (onerec_params->has_encoder_output) {
        std::lock_guard lock(encoder_output_mutex_);
        cached_encoder_output = encoder_output_;
    }

    const torch::Tensor& decoder_context = onerec_params->decoder_context_embedding;
    if (!decoder_context.defined() && !cached_encoder_output.defined()) {
        LOG(ERROR) << "OneRec decoder requires encoder_output or decoder_context";
        return ModelOutput();
    }

    auto decoder_output = decoder_(
        tokens, positions, kv_caches, input_params, cached_encoder_output);
    // decoder_ 也是 OneRecStack（is_decode=true）

    return ModelOutput(decoder_output);
}
```

### 7.4.1 `OneRecStack::forward()`
**文件：`xllm/models/rec/npu/onerec_npu_impl.h:190`**

核心循环：嵌入 → 位置编码 → 多层 block → LayerNorm。

```cpp
torch::Tensor OneRecStackImpl::forward(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& input_params,
    const torch::Tensor& encoder_output) {    // 仅 decoder 时有效
    auto* onerec_params = input_params.onerec_params();
    CHECK(onerec_params != nullptr) << "OneRec requires onerec_params().";

    // ============ (1) 准备 hidden_states h ============
    torch::Tensor h;
    if (onerec_params->is_hybrid_mode && !is_decoder_) {
        h = tokens;   // hybrid: 输入直接是 hidden states
    } else if (onerec_params->decoder_context_embedding.defined()) {
        if (tokens.numel() == 0) {
            h = onerec_params->decoder_context_embedding.reshape({-1, hidden_size_});
        } else {
            h = embed_tokens_(tokens);
            // 将 decoder_context_embedding 中对应位置 fill 到 h 中
            auto context_emb = onerec_params->decoder_context_embedding.clone();
            // reshape、copy_ 等操作
            h = context_emb.view({-1, hidden_size_});
        }
    } else {
        h = embed_tokens_(tokens);   // 标准 token embedding
    }

    // ============ (2) 准备位置编码 / mask ============
    bool is_prefill = (onerec_params->rec_stage == PREFILL);
    auto [query_length, key_length] = compute_sequence_lengths(
        input_params.meta.q_max_seq_len, is_prefill, input_params);

    torch::Tensor effective_attn_mask;
    if (use_absolute_position_embedding_) {
        effective_attn_mask = create_moe_attention_mask(query_length, h, is_decoder_, bs);
    } else {
        effective_attn_mask = compute_position_bias_mask(
            query_length, key_length, h, is_decode_stage, input_params);
        // 内部调用 compute_onerec_position_bias（T5-style 相对位置）
    }
    auto preprocessed_attn_mask = preprocess_attention_mask(effective_attn_mask, h);

    // ============ (3) MoE 准备（若有）============
    torch::Tensor expert_array;
    if (use_moe_) {
        expert_array = torch::arange(0, input_length * num_experts_per_tok_, ...);
    }

    // ============ (4) 多层循环：每个 block 一次 forward ============
    for (size_t i = 0; i < layers_.size(); ++i) {
        // layer synchronizer（用于流水线并行，可选）
        aclrtEvent* event = nullptr;
        std::atomic<bool>* event_flag = nullptr;
        if (input_params.parallel.layer_synchronizer) {
            event = input_params.parallel.layer_synchronizer->get_event(i);
            event_flag = input_params.parallel.layer_synchronizer->get_event_flag(i);
        }

        KVCache dummy_kv_cache;
        KVCache& kv_cache_ref = is_decoder_ ? kv_caches[i] : dummy_kv_cache;

        // ★ 每层 forward
        layers_[i]->forward(
            h,
            preprocessed_attn_mask,
            kv_cache_ref,
            input_params_local,
            npu_encoder_output.defined() ? &npu_encoder_output : nullptr,
            static_cast<int>(i),
            event, event_flag,
            expert_array);
        // layers_[i] 是 NpuOneRecBlockLayer
    }

    // ============ (5) 最终 LayerNorm ============
    std::optional<torch::Tensor> residual = std::nullopt;
    h = std::get<0>(norm_->forward(h, residual));   // RMSNorm
    return h;
}
```

### 7.4.2 `NpuOneRecBlockLayerImpl::forward()`（NPU ATB op graph 执行）
**文件：`xllm/core/layers/npu/npu_onerec_block_layer_impl.cpp:1305`**

每层对应一个 ATB op graph（encoder/decoder 分别有 prefill/decode node）：

```cpp
torch::Tensor NpuOneRecBlockLayerImpl::forward(
    torch::Tensor& x, torch::Tensor& attn_mask, KVCache& kv_cache,
    ModelInputParams& input_params, torch::Tensor* encoder_output,
    int32_t node_id, aclrtEvent* event, std::atomic<bool>* event_flag,
    const torch::Tensor& expert_array) {
    auto* onerec_params = input_params.onerec_params();
    bool is_prefill = (onerec_params->rec_stage == PREFILL);
    bool is_first_prefill = onerec_params->is_first_prefill;

    atb::Status st;
    if (is_prefill) {
        if (is_decoder_) {
            // Decoder prefill node（含 cross-attention + KV cache 写入）
            //   variantPack 设置：x, attn_mask, kv_cache, encoder_output
            if (use_legacy_onerec_prefill_only_contract()) {
                auto& target_node = is_first_prefill
                    ? prefill_node_
                    : decoder_prefill_only_decode_node_;
                build_decoder_node_variant_pack(
                    target_node, x, attn_mask, kv_cache, input_params,
                    /*is_prefill=*/true, is_first_prefill,
                    is_first_prefill ? encoder_output : nullptr, node_id);
                st = execute_node(target_node, node_id, event, event_flag);
            } else {
                // 通用 prefill node
                build_decoder_node_variant_pack(
                    prefill_node_, x, attn_mask, kv_cache, input_params,
                    true, true, encoder_output, node_id);
                st = execute_node(prefill_node_, node_id, event, event_flag);
            }
        } else {
            // Encoder prefill node（不含 KV cache，不做 cross）
            build_encoder_node_variant_pack(
                prefill_node_, x, attn_mask, input_params, true, node_id);
            st = execute_node(prefill_node_, node_id, event, event_flag);
        }
    } else {
        // Decode 阶段（仅 decoder）
        build_decoder_node_variant_pack(
            decode_node_, x, attn_mask, kv_cache, input_params,
            /*is_prefill=*/false, false,
            encoder_output, node_id);
        st = execute_node(decode_node_, node_id + 1000, event, event_flag);
    }
    return at_placeholder_;   // 输出通过 variantPack 的 outTensors 传出
}
```

每层内部（ATB op graph）封装了：
- LayerNorm / RMSNorm
- Self-attention（QKV + attention + output projection）
- Cross-attention（decoder 层，is_first_prefill 或 decode 阶段）
- FFN 或 MoE（shared + routed experts）
- 残差连接

### 7.5 `RecForCausalLMImplBase::logits()`
**文件：`xllm/models/rec/rec_model_base.h:56`**

```cpp
torch::Tensor logits(const torch::Tensor& hidden_states,
                      const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (tie_word_embeddings_) {
        const float denom = std::sqrt(static_cast<float>(args.hidden_size()));
        h = hidden_states * (1.0f / denom);
    }
    if (selected_idxes.defined()) {
        h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);   // 线性输出 → vocabulary logits
}
```

---

## 第八阶段：输出回收

模型前向返回 `ForwardOutput{hidden_states, sample_output, logits}` 后：

```
OneRecWorkPipeline::step() 返回 ForwardOutput
   ↑
RecWorkerImpl::step_async() promise.setValue(output)
   ↑
get_model_output() collectAll 合并所有 worker 输出
   ↑
OneRecPrefillOnlyEnginePipeline::step() / OneRecXAttentionEnginePipeline::step()
  → batches[0].process_sample_output(...)   // 写入 sequence
  → batches[0].finish()                      // 标记完成
   ↑
FixedStepsScheduler::step() lambda 后处理：
  → request->update_connection_status()
  → kv_cache_manager_->deallocate()          // 释放 block
  → response_processor_->process_completed_requests()
       → callback(RequestOutput)              // 用户提供的输出回调
            → 序列化 → HTTP 响应写回客户端
```

---

## 第九阶段：模型注册机制

### 9.1 Macro 展开
**文件：`xllm/models/model_registry.h:165`**

```cpp
#define REGISTER_REC_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_rec_registered = []() {                          \
    ModelRegistry::register_rec_model_factory(                          \
        #ModelType, [](const ModelContext& context) {                   \
          ModelClass model(context);                                    \
          model->eval();                                                \
          return std::make_unique<xllm::RecCausalLMImpl<ModelClass>>(   \
              std::move(model), context.get_tensor_options());          \
        });                                                             \
    return true;                                                        \
  }()
```

### 9.2 OneRec 注册项
**文件：`xllm/models/rec/npu/onerec.h`**

```cpp
REGISTER_REC_MODEL(onerec, OneRecForConditionalGeneration);         // line 309
// 等价于：
// const bool onerec_rec_registered = []() {
//     register_rec_model_factory("onerec",
//         [](ctx) { return RecCausalLMImpl<OneRecForConditionalGeneration>(...); });
//     return true;
// }();

REGISTER_MODEL_ARGS(onerec, [&] {                                    // line 311
    LOAD_ARG_OR(model_type, "model_type", "onerec");
    LOAD_ARG_OR(dtype, "torch_dtype", "bfloat16");
    LOAD_ARG_OR(hidden_size, "d_model", 128);
    LOAD_ARG_OR(intermediate_size, "d_ff", 256);
    LOAD_ARG_OR(n_layers, "num_decoder_layers", 4);
    LOAD_ARG_OR(n_encoder_layers, "num_layers", 12);
    LOAD_ARG_OR(n_heads, "num_heads", 4);
    LOAD_ARG_OR(head_dim, "d_kv", 32);
    LOAD_ARG_OR(vocab_size, "vocab_size", 8200);
    LOAD_ARG_OR(rms_norm_eps, "layer_norm_epsilon", 1e-6);
    LOAD_ARG_OR(max_position_embeddings, "max_length", 500);
    LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", true);
    LOAD_ARG_OR(use_moe, "use_moe", false);
    ...
});

REGISTER_TOKENIZER_ARGS(onerec, [&] {                                // line 357
    SET_ARG(tokenizer_type, "rec");
    LOAD_ARG_OR(vocab_file, "vocab_file", "");
});
```

### 9.3 静态注册触发
**文件：`xllm/models/models.h:52`**

```cpp
#if defined(USE_NPU)
#include "rec/npu/onerec.h"   // include 触发 REGISTER_* 静态初始化
...
```

---

## 类继承关系

```
CausalLM (core/framework/model/causal_lm.h)
  │
  └─ RecCausalLM (core/framework/model/rec_causal_lm.h)
       │
       └─ RecCausalLMImpl<T>  (模板类, rec_causal_lm.h:28)
            │  持有 model_: T
            │  forward/logits/pooler 都委托给 model_
            │
            T = OneRecForConditionalGeneration (torch module)
                │
                └─ OneRecForConditionalGenerationImpl  (models/rec/npu/onerec.h:256)
                     继承 RecForCausalLMImplBase<OneRecModel>
                     持有 model_: OneRecModel + lm_head_: LmHead
                     forward → model_->forward
                     logits  → lm_head_(select(h))
                     load_model → model_->load_state_dict + lm_head_->load_state_dict
                     │
                     └─ OneRecModelImpl  (models/rec/npu/onerec.h:35)
                          torch::nn::Module
                          持有:
                            shared_: WordEmbedding
                            encoder_: OneRecStack (is_decode=false)
                            decoder_: OneRecStack (is_decode=true)
                            encoder_output_: Tensor (缓存)
                          forward():
                            encoder_forward → encoder_()
                            decoder_forward → decoder_(cached_encoder_output)

OneRecStack / OneRecStackImpl  (models/rec/npu/onerec_npu_impl.h:152)
  │  torch::nn::Module
  │  持有:
  │    embed_tokens_: WordEmbedding
  │    position_bias_embedding_: WordEmbedding
  │    layers_: vector<NpuOneRecBlockLayer>
  │    norm_: RMSNorm
  └─ forward():  embed → position bias → loop [layer.forward] → norm

NpuOneRecBlockLayer / NpuOneRecBlockLayerImpl  (core/layers/npu/npu_onerec_block_layer_impl.h:41)
  │  继承 BaseLayer
  │  持有 ATB op graph: prefill_node_, decode_node_, (encoder variants)
  └─ forward(): 根据 is_prefill/is_decoder 选择 node，调用 execute_node()
       内部 ATB op 封装了：Norm + QKV + SelfAttn + (CrossAttn) + FFN/MoE + 残差
```

---

## 关键文件速查表

| 文件 | 角色 |
|------|------|
| `xllm/xllm.cpp` | main / run / create_options |
| `xllm/core/distributed_runtime/master.h/cpp` | Master 基类 + create_master 工厂 |
| `xllm/core/distributed_runtime/rec_master.h/cpp` | RecMaster + Master 层 Pipeline + 调度循环 |
| `xllm/core/distributed_runtime/rec_engine.h/cpp` | RecEngine + Engine 层 Pipeline |
| `xllm/core/distributed_runtime/engine.h` | Engine 接口 |
| `xllm/core/scheduler/fixed_steps_scheduler.h/cpp` | FixedStepsScheduler::step/prepare_batch/handle_prefill |
| `xllm/core/runtime/worker.h/cpp` | Worker 外观层，委托 impl_ |
| `xllm/core/runtime/worker_impl.h/cpp` | WorkerImpl 基类（权重加载等） |
| `xllm/core/runtime/rec_worker_impl.h/cpp` | RecWorkerImpl + Worker 层 Pipeline（OneRec/XAttention/...） |
| `xllm/core/runtime/llm_worker_impl.h/cpp` | LLMWorkerImpl（RecWorkerImpl 的父类） |
| `xllm/core/runtime/executor.h/cpp` | Executor 包装 |
| `xllm/core/runtime/base_executor_impl.h/cpp` | BaseExecutorImpl（run → model_->forward） |
| `xllm/core/framework/model/causal_lm.h` | CausalLM 接口 |
| `xllm/core/framework/model/rec_causal_lm.h` | RecCausalLM + RecCausalLMImpl<T> |
| `xllm/models/rec/rec_model_base.h` | RecForCausalLMImplBase<ModelType> |
| `xllm/models/rec/npu/onerec.h` | OneRecModelImpl + OneRecForConditionalGeneration + 注册宏 |
| `xllm/models/rec/npu/onerec_npu_impl.h` | OneRecStack + NPU position bias 等辅助 |
| `xllm/models/rec/onerec.h` | OneRec CPU/GPU fallback 实现 |
| `xllm/core/layers/npu/npu_onerec_block_layer_impl.h/cpp` | NpuOneRecBlockLayerImpl：每层 ATB op graph |
| `xllm/core/layers/onerec_block_layer.h` | OneRecBlockLayer 抽象基类 |
| `xllm/models/model_registry.h/cpp` | ModelRegistry + REGISTER_* 宏 + create_rec_model |
| `xllm/models/models.h` | include 聚合，触发静态注册 |
| `xllm/api_service/api_service.h/cpp` | APIService（HTTP 路由） |
| `xllm/api_service/rec_completion_service_impl.h/cpp` | /v1/completions 处理 |
| `xllm/api_service/service_impl_factory.cpp` | 服务实现工厂（根据 serving mode） |
| `xllm/server/xllm_server_registry.h/cpp` | HTTP server 注册 |
| `xllm/core/framework/config/rec_config.h` | Rec 专用配置 |
| `xllm/core/runtime/forward_params.h` | ForwardInput 定义 + `to(device)` H2D 入口 |
| `xllm/core/framework/model/model_input_params.h` | ModelInputParams / OneRecModelInputParams 定义 + `to(device)` |
| `xllm/core/framework/batch/onerec_batch_input_builder.cpp` | OneRec Batch 打包：构造 ForwardInput（CPU pinned memory） |
| `xllm/core/util/utils.cpp` | `convert_rec_tensor_to_torch`：proto → CPU tensor |
| `xllm/proto/completion.proto` | CompletionRequest / CompletionResponse 定义 |
| `xllm/proto/rec.proto` | InferInputTensor / InferOutputTensor 定义 |
| `xllm/core/framework/request/request_params.h/cpp` | RequestParams 定义与构造 |
| `xllm/core/framework/request/request.h` | Request 类 |
| `xllm/core/framework/request/request_state.h` | RequestState 定义 |
| `xllm/core/framework/request/sequence.h` | Sequence + OneRecState 定义 |
| `xllm/core/framework/request/request_output.h` | RequestOutput / SequenceOutput 定义 |
| `xllm/core/framework/batch/batch.h/cpp` | Batch 定义 |
| `xllm/core/framework/sampling/sampling_params.h` | SamplingParameters / SampleOutput 定义 |
| `xllm/core/framework/model/model_output.h` | ModelOutput 定义 |

---

## 附录 B：请求到响应的数据结构变化

本节追踪一个 OneRec 请求从 HTTP 入口到 HTTP 出口，数据在各层之间以什么结构存在、如何转换。

### B.1 总览流程图

```
HTTP JSON Body
  │  json2pb::JsonToProtoMessage
  ▼
proto::CompletionRequest
  │  + input_tensors: [InferInputTensor]
  │  RequestParams 构造函数
  ▼
RequestParams
  │  RecMaster::handle_request → pipeline->generate_request
  ▼
Request
  ├── RequestState (sampling/stopping/mm_data)
  └── SequencesGroup → Sequence[] (tokens_, mm_data_, OneRecState)
  │  Scheduler: batch.add(sequence)
  ▼
Batch
  ├── sequences_: vector<Sequence*>
  └── sequence_groups_: vector<SequencesGroup*>
  │  Batch::prepare_rec_forward_input → OneRecBatchInputBuilder
  ▼
ForwardInput
  ├── token_ids: Tensor
  ├── input_params: ModelInputParams
  │     └── rec_params: OneRecModelInputParams
  └── sampling_params: SamplingParameters
  │  OneRecWorkPipeline::step()
  │  1. encoder forward  2. decoder forward
  │  3. logits = model->logits(hidden, selected_idxes)
  │  4. sample_output = rec_sampler_->forward(logits, ...)
  ▼
ForwardOutput
  ├── logits: Tensor
  └── sample_output: SampleOutput (next_tokens, logprobs, ...)
  │  Batch::process_sample_output
  ▼
Sequence::append_token(token)
  │  tokens_ 更新, stopping_checker 评估
  │  Request::generate_output → SequencesGroup::generate_outputs
  ▼
RequestOutput
  ├── request_id, status, usage, finished
  └── outputs: vector<SequenceOutput>
        ├── token_ids, item_ids, item_ids_list
        └── token_ids_logprobs
  │  OutputCallback → send_result_to_client_brpc_rec
  ▼
proto::CompletionResponse
  ├── choices: [Choice]
  ├── usage: Usage
  └── output_tensors: [InferOutputTensor]
        ├── "rec_result": item IDs 或 token IDs
        └── "sku_logprobs" (可选)
  │  json2pb::ProtoMessageToJson
  ▼
HTTP JSON Response
```

### B.2 各层数据结构详解

#### ① proto::CompletionRequest（HTTP 入口）
**定义：`xllm/proto/completion.proto`**

| 字段 | 类型 | 说明 |
|------|------|------|
| `model` | `string` | 模型名 |
| `prompt` | `string` | 文本 prompt |
| `max_tokens` | `optional uint32` | 最大生成长度 |
| `temperature` / `top_p` / `top_k` | `optional float/int64` | 采样参数 |
| `n` / `best_of` | `optional uint32` | 返回数 / 候选数 |
| `stream` | `optional bool` | 流式输出 |
| `beam_width` | `optional int32` | beam search 宽度 |
| `num_return_sequences` | `optional int32` | 返回序列数 |
| `token_ids` | `repeated int32` | 预分词输入 |
| **`input_tensors`** | **`repeated InferInputTensor`** | **OneRec 专用：embedding 输入** |

`InferInputTensor`（`rec.proto`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | `string` | `"encoder_sparse_embedding"` 或 `"decoder_context_embedding"` |
| `data_type` | `DataType` | `FLOAT` 等 |
| `shape` | `repeated int64` | 2D shape `[len, hidden]` |
| `contents` | `InferTensorContents` | `fp32_contents` 等 |

#### ② RequestParams（proto 解析后）
**定义：`xllm/core/framework/request/request_params.h`**

从 proto 拷贝各字段到扁平 C++ 结构，应用默认值：

| 字段 | 类型 | 默认值 |
|------|------|--------|
| `request_id` | `string` | 自动生成 |
| `streaming` | `bool` | `false` |
| `max_tokens` | `uint32_t` | `5120` |
| `n` / `best_of` | `uint32_t` / `optional<uint32_t>` | `1` / `nullopt` |
| `temperature` / `top_p` / `top_k` | `float` / `float` / `int64_t` | `0.0` / `1.0` / `-1` |
| `beam_width` | `int32_t` | `0` |
| `num_return_sequences` | `int32_t` | `0` |

> **注意**：`input_tensors` 不存入 RequestParams，而是通过 `RecMaster::handle_request` 的第三个参数单独传递。

#### ③ Request / RequestState / Sequence（请求对象）
**定义：`xllm/core/framework/request/request.h`、`request_state.h`、`sequence.h`**

`RequestParams` 的字段被拆分到三个子结构中：

```
Request
  ├── state_: RequestState
  │     ├── sampling_param: RequestSamplingParam  ← 采样参数
  │     ├── scheduler_param: SchedulerParam       ← SLO/优先级
  │     ├── stopping_checker: StoppingChecker     ← 停止条件
  │     ├── prompt_tokens: vector<int32_t>
  │     ├── mm_data: MMData                       ← encoder_sparse_embedding 等
  │     └── rec_type: RecType = kOneRec
  │
  └── sequences_group_: SequencesGroup
        └── sequences_: vector<Sequence>
              ├── tokens_: vector<int32_t>          ← prompt + 已生成 token
              ├── mm_data_: MMData                  ← 从 RequestState 拷贝
              ├── kv_state_: KVCacheState           ← KV cache block 管理
              ├── finish_reason_: FinishReason
              └── onerec_state_: OneRecState
                    ├── num_encoder_tokens: size_t
                    ├── num_decoder_embeddings: size_t
                    └── encoder_tokens: vector<int32_t>
```

**关键转换**：`input_tensors` 中的 `encoder_sparse_embedding` / `decoder_context_embedding` 被解析为 CPU BFloat16 `torch::Tensor`，存入 `MMData`（`std::unordered_map<std::string, torch::Tensor>`）。

#### ④ Batch（调度打包）
**定义：`xllm/core/framework/batch/batch.h`**

Scheduler 从多个 Request 中收集 Sequence 指针打包：

| 字段 | 类型 | 说明 |
|------|------|------|
| `sequences_` | `vector<Sequence*>` | 所有待推理的 sequence |
| `sequence_groups_` | `vector<SequencesGroup*>` | 按请求分组 |
| `batch_forward_type_` | `BatchForwardType` | `PREFILL` / `DECODE` / `MIX` |
| `allowed_max_tokens_` | `vector<uint32_t>` | 每 sequence 的 max token 限制 |

#### ⑤ ForwardInput（模型输入）
**定义：`xllm/core/runtime/forward_params.h`**

由 `OneRecBatchInputBuilder::build_rec_forward_input()` 构造：

| 字段 | 类型 | 说明 |
|------|------|------|
| `token_ids` | `torch::Tensor` | 展平的 decoder token IDs `[total_tokens]` |
| `positions` | `torch::Tensor` | 固定 `{0}` |
| `token_ids_host` | `torch::Tensor` | CPU view（H2D 后保留） |
| `input_params` | `ModelInputParams` | 包含 attention/embedding/rec 参数 |
| `sampling_params` | `SamplingParameters` | 采样配置 |

`ModelInputParams` 中的 `rec_params` 是 `std::variant`，OneRec 时为 `OneRecModelInputParams`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `rec_stage` | `RecStage` | `PREFILL` 或 `DECODE` |
| `is_first_prefill` | `bool` | 是否首次 prefill |
| `bs` / `group_width` | `int32_t` | batch 维度 |
| `seq_len` | `int32_t` | decoder token 长度 |
| `encoder_max_seq_len` | `int32_t` | encoder 最大长度 |
| `encoder_seq_lens` | `vector<int32_t>` | 每 sequence 的 encoder 长度 |
| `encoder_sparse_embedding` | `torch::Tensor` | 拼接后的 sparse embedding |
| `decoder_context_embedding` | `torch::Tensor` | `[bs, gw, ctx+seq, hidden]` |
| `encoder_token_ids` | `torch::Tensor` | encoder token IDs（无 sparse emb 时） |
| `generated_tokens` | `vector<vector<int32_t>>` | 已生成 token（约束解码用） |

#### ⑥ ModelOutput（模型输出）
**定义：`xllm/core/framework/model/model_output.h`**

`executor->forward()` 的返回值：

| 字段 | 类型 | 说明 |
|------|------|------|
| `hidden_states` | `torch::Tensor` | `[num_tokens, hidden_size]` |
| `residual` | `torch::Tensor` | 残差流（可选） |

随后 `OneRecWorkPipeline::step()` 调用 `model->logits(hidden_states, selected_token_idxes)` 提取 logits。

#### ⑦ ForwardOutput（推理输出）
**定义：`xllm/core/runtime/forward_params.h`**

| 字段 | 类型 | 说明 |
|------|------|------|
| `logits` | `torch::Tensor` | 原始 logits |
| `sample_output` | `SampleOutput` | 采样结果 |
| `do_sample` | `torch::Tensor` | 每 sequence 是否采样 |
| `beam_sequence_group` | `torch::Tensor` | beam search 结果（XAttention） |

`SampleOutput`（`sampling_params.h`）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `next_tokens` | `torch::Tensor` | `[num_seq]` 采样的 token IDs |
| `probs` | `torch::Tensor` | `[num_seq]` token 概率 |
| `logprobs` | `torch::Tensor` | `[num_seq]` log 概率 |
| `top_tokens` / `top_logprobs` | `torch::Tensor` | top-k tokens/logprobs |

#### ⑧ Batch::process_sample_output（结果回写）

`batch.process_sample_output(sample_output)` 将采样结果写回各 Sequence：

```
For each output_target in batch:
  1. 从 sample_output.next_tokens[i] 取 token ID
  2. 构造 Token{id, logprob, top_tokens, top_logprobs}
  3. sequence->append_token(token)
       → tokens_.push_back(token_id)
       → token_to_count_map_[token_id]++
       → num_tokens_++
  4. stopping_checker->check(token_id)
       → 判断是否命中 stop_token / EOS / max_length
```

#### ⑨ RequestOutput / SequenceOutput（最终输出）
**定义：`xllm/core/framework/request/request_output.h`**

`Request::generate_output(tokenizer)` 构造最终输出：

```
RequestOutput
  ├── request_id: string
  ├── status: optional<Status>
  ├── usage: Usage {num_prompt_tokens, num_generated_tokens, num_total_tokens}
  ├── finished: bool
  └── outputs: vector<SequenceOutput>
        ├── index: size_t
        ├── text: string                    ← 生成的文本
        ├── token_ids: vector<int32_t>      ← 生成的 token IDs
        ├── item_ids: optional<int64_t>     ← OneRec 解码出的推荐 item ID
        ├── item_ids_list: vector<int64_t>  ← 多 item 推荐
        ├── item_info: optional<RecItemInfo> ← 扩展 item 信息
        ├── finish_reason: optional<string> ← "stop" / "length"
        └── token_ids_logprobs: vector<optional<float>>
```

**OneRec 特有**：`Sequence::generate_onerec_output()` 将生成的 token IDs 解码为推荐 item ID（`item_ids`、`item_ids_list`），这是 OneRec 作为推荐模型的核心输出。

#### ⑩ proto::CompletionResponse（HTTP 出口）
**定义：`xllm/proto/completion.proto`**

`send_result_to_client_brpc_rec()` 将 `RequestOutput` 序列化为 proto：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `string` | request_id |
| `object` | `string` | `"text_completion"` |
| `created` | `uint32` | Unix 时间戳 |
| `model` | `string` | 模型名 |
| `choices` | `repeated Choice` | 每 SequenceOutput 一个 Choice |
| `usage` | `Usage` | token 用量统计 |
| **`output_tensors`** | **`repeated InferOutputTensor`** | **OneRec 专用输出** |

`output_tensors` 包含：
- **`"rec_result"`**：推荐结果
  - 若 `enable_convert_tokens_to_item=true`：`int64_contents` = 解码后的 item IDs
  - 否则：`int_contents` = 原始 token IDs
- **`"sku_logprobs"`**（可选）：每 token 的 logprobs

最终通过 `json2pb::ProtoMessageToJson` 转为 JSON 写入 HTTP 响应体。

### B.3 MMData 存取链路详解

MMData 是 OneRec 请求中 embedding 数据的载体，本质是 `std::unordered_map<std::string, torch::Tensor>` 的包装类。

**定义：`xllm/core/framework/multimodal/mm_data.h`**

```cpp
class MMData {
    uint32_t type_;                          // MMType 位掩码（EMBEDDING 等）
    MMItems items_;                          // std::variant<MMItemVec, MMDict>
    // MMDict = std::unordered_map<std::string, MMValue>
    // MMValue = std::variant<torch::Tensor, std::vector<torch::Tensor>, ...>
};
```

#### 三条链路的整体关系

**存入、取出、输出**三条链路都是 xLLM **服务端**处理同一个客户端请求时的操作，发生在请求生命周期的不同阶段：

```
客户端发 HTTP 请求
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  xLLM 服务端                                             │
│                                                          │
│  ① 存入链路（请求接入阶段）                                │
│     HTTP → proto 解析 → CPU Tensor → MMData → Sequence   │
│     目的：把客户端传来的 embedding 数据存起来               │
│     时机：请求刚到达，还没开始推理                          │
│                                                          │
│  ② 取出链路（调度+推理准备阶段）                           │
│     Sequence.mm_data_ → 拼接 → H2D → 送入模型            │
│     目的：把存好的 embedding 取出来喂给模型                 │
│     时机：Scheduler 选中该请求，准备执行推理                │
│                                                          │
│  ③ 输出链路（结果返回阶段）                                │
│     模型输出 → 采样 → token 回写 → 封装 proto → HTTP 返回  │
│     目的：把推理结果返回给客户端                            │
│     时机：模型推理完成后                                    │
└─────────────────────────────────────────────────────────┘
  │
  ▼
客户端收到 HTTP 响应
```

| 链路 | 触发时机 | 谁触发 | 在哪执行 |
|------|----------|--------|----------|
| 存入 | 请求到达 | API 线程 | CPU（Master 层） |
| 取出 | Scheduler 选中请求 | loop_thread_ → Engine → Worker | CPU → NPU |
| 输出 | 模型推理完成 | Engine → Scheduler → ResponseProcessor | NPU → CPU → 网络 |

三条链路之间通过 `Sequence.mm_data_`（存入→取出）和 `Sequence.tokens_`（推理→输出）衔接。数据在服务端"存一次、取一次、算一次、回一次"。

xLLM 是服务端，只负责序列化和返回响应，不解析自己的输出。响应的解析由客户端（curl、Python openai 库、自定义 HTTP 客户端等）完成。

#### 存入链路（完整调用链）

```
RecCompletionServiceImpl::process_async_impl(call)
  │  从 proto 请求中拷贝 input_tensors
  │  rec_completion_service_impl.cpp:308-316
  │
  ├─ input_tensors = std::vector<proto::InferInputTensor>
  │    for (i = 0; i < rpc_request.input_tensors_size(); ++i)
  │        tensors.push_back(rpc_request.input_tensors(i));
  │
  └─ master_->handle_request(prompt, prompt_tokens, input_tensors, ...)
       │  rec_master.cpp:613
       │
       └─ schedule_request(sp, callback, build_request_lambda)
            │  rec_master.cpp:625
            │  将 input_tensors 捕获到 lambda 闭包中
            │
            └─ pipeline_->generate_request(prompt, prompt_tokens, input_tensors, ...)
                 │  rec_master.cpp:632
                 │
                 └─ OneRecPrefillOnlyMasterPipeline::generate_request(...)
                      │  rec_master.cpp:466
                      │
                      └─ generate_onerec_request_common(prompt, prompt_tokens, input_tensors, ...)
                           │  rec_master.cpp:339
                           │
                           ├─ process_onerec_inputs(prompt_tokens, input_tensors, model_args, ...)
                           │    │  rec_master.cpp:350
                           │    │
                           │    │  遍历 input_tensors 中的每个 InferInputTensor:
                           │    │  rec_master.cpp:106-196
                           │    │
                           │    ├─ 校验 name (只允许 "sparse_embedding" / "decoder_context_embedding")
                           │    ├─ 校验 shape (必须 2D [len, hidden])
                           │    ├─ 校验 dtype (必须 FLOAT/fp32)
                           │    ├─ 校验 numel (len * hidden == fp32_contents.size())
                           │    │
                           │    └─ util::convert_rec_tensor_to_torch(tensor).to(torch::kBFloat16)
                           │         │  utils.cpp:179
                           │         │
                           │         ├─ torch::from_blob(fp32_contents.data(), shape, kFloat32)
                           │         │    包裹 protobuf 内存为 CPU Tensor（不拷贝）
                           │         │
                           │         └─ .clone()
                           │              拷贝到独立内存，protobuf 释放后不受影响
                           │              结果: CPU fp32 Tensor → .to(kBFloat16) → CPU bf16 Tensor
                           │
                           │    存入 mm_dict:
                           │    rec_master.cpp:181-182
                           │    mm_dict["sparse_embedding"] = CPU bf16 Tensor
                           │    mm_dict["decoder_context_embedding"] = CPU bf16 Tensor (可选)
                           │
                           │    封装为 MMData:
                           │    rec_master.cpp:216
                           │    *processed_mm_data = MMData(MMType::EMBEDDING, mm_dict)
                           │
                           └─ build_request_common(prompt, prompt_tokens, processed_mm_data, ...)
                                │  rec_master.cpp:361
                                │
                                ├─ 构造 RequestState:
                                │    rec_master.cpp:842-857
                                │    RequestState req_state(prompt, prompt_tokens,
                                │        std::move(mm_data),  ← MMData 移入 RequestState
                                │        sampling_param, stopping_checker, ...);
                                │    req_state.rec_type = RecType::kOneRec;
                                │
                                ├─ 构造 Request:
                                │    rec_master.cpp:860
                                │    auto request = std::make_shared<Request>(
                                │        request_id, x_request_id, x_request_time,
                                │        std::move(req_state), ...);
                                │
                                │    Request 构造函数:
                                │    request.cpp:35-48
                                │    Request::Request(..., const RequestState& state, ...)
                                │        : state_(std::move(state)) {
                                │        create_sequences_group();
                                │    }
                                │
                                ├─ create_sequences_group():
                                │    request.cpp:50-71
                                │    sequences_group_ = std::make_unique<SequencesGroup>(
                                │        state_.prompt,
                                │        state_.prompt_tokens,
                                │        state_.input_embedding,
                                │        state_.mm_data,          ← 传入 MMData 引用
                                │        sequence_params);
                                │
                                ├─ SequencesGroup 构造:
                                │    sequences_group.cpp:31-42
                                │    SequencesGroup::SequencesGroup(..., const MMData& mm_data, ...)
                                │        : mm_data_(mm_data), ... {    ← 拷贝 MMData 到成员
                                │        add();                         ← 创建第一个 Sequence
                                │    }
                                │
                                ├─ SequencesGroup::add():
                                │    sequences_group.cpp:44-56
                                │    sequences_.emplace_back(std::make_unique<Sequence>(
                                │        index, prompt_tokens_, input_embedding_,
                                │        mm_data_,                   ← 传入 MMData 引用
                                │        decoder, sequence_params_));
                                │
                                └─ Sequence 构造:
                                     sequence.cpp:226-241
                                     Sequence::Sequence(..., const MMData& mm_data, ...)
                                         : mm_data_(mm_data), ... {    ← 拷贝 MMData 到成员
                                         if (is_onerec_model()) {
                                             init_onerec_sequence(prompt_token_ids, ...);
                                             return;
                                         }
                                     }

                                     init_onerec_sequence():
                                     sequence.cpp:117-156
                                     ├─ 从 mm_data_ 读取 encoder_sparse_embedding 确定 encoder 长度
                                     │    auto emb = mm_data_.get<torch::Tensor>("sparse_embedding");
                                     │    onerec_state.num_encoder_tokens = emb.value().size(0);
                                     │
                                     ├─ 从 mm_data_ 读取 decoder_context_embedding 确定 decoder 长度
                                     │    auto ctx = mm_data_.get<torch::Tensor>("decoder_context_embedding");
                                     │    onerec_state.num_decoder_embeddings = ctx.value().size(0);
                                     │
                                     └─ 初始化 tokens_ 容量、填充 BOS token 等
```

#### 取出链路（完整调用链）

取出链路从调度循环开始，经过 batch 构建、engine 分发、worker 准备输入，最终从 Sequence.mm_data_ 中取出 embedding 数据并送入模型。

```
RecMaster::run() 的 loop_thread_
  │  rec_master.cpp:592-597
  │  while (!stopped_) { scheduler_->step(timeout); }
  │
  └─ FixedStepsScheduler::step(timeout)
       │  fixed_steps_scheduler.cpp:343
       │
       ├─① schedule_request(timeout) → prepare_batch()
       │    │  fixed_steps_scheduler.cpp:346 → 311 → 192
       │    │
       │    ├─ 从 request_queue_ 读取新请求，放入 waiting_priority_queue_
       │    │    fixed_steps_scheduler.cpp:198-213
       │    │    while (request_queue_.read(request)) {
       │    │        request->expand_sequences(false);
       │    │        waiting_priority_queue_->push(request);
       │    │    }
       │    │    // request 内含 SequencesGroup → Sequence（携带 mm_data_）
       │    │
       │    ├─ handle_prefill_requests()：从等待队列挑选请求进入运行队列
       │    │    fixed_steps_scheduler.cpp:68-174
       │    │    while (!waiting_priority_queue_->empty() && budget > 0) {
       │    │        request = waiting_priority_queue_->top();
       │    │        for (auto& seq : request->sequences()) {
       │    │            // 检查 token/seq budget
       │    │            // OneRec 不需要 KV cache（requires_kv_cache=false）
       │    │            prefill_sequences.emplace_back(seq.get());
       │    │        }
       │    │        running_requests_.emplace_back(request);
       │    │        running_sequences_.insert(..., prefill_sequences);
       │    │    }
       │    │    // 此时 running_requests_ 中的 Request 持有 Sequence（含 mm_data_）
       │    │
       │    └─ scheduler_pipeline_->create_batches(*this, batch_factory)
       │         │  fixed_steps_scheduler.cpp:275
       │         │
       │         └─ OneRecSchedulerPipeline::create_batches()
       │              │  fixed_steps_scheduler.cpp:425-433
       │              │
       │              └─ batch_factory->create_rec_batches(
       │                     running_requests_, running_sequences_, ...)
       │                   │  batch_factory.cpp:96-154
       │                   │
       │                   │  遍历 running_requests_，将每个 request 的
       │                   │  sequence_group 加入 Batch:
       │                   │  batch_factory.cpp:123-127
       │                   │  for (const auto& request : running_requests) {
       │                   │      auto seq_group = request->sequence_group();
       │                   │      batches[dp_rank].add(seq_group);
       │                   │      // Batch.sequence_groups_ 现在持有 SequencesGroup*
       │                   │      //   → SequencesGroup 内含 Sequence*
       │                   │      //     → Sequence 内含 mm_data_（CPU bf16 Tensor）
       │                   │  }
       │                   │
       │                   └─ 返回 std::vector<Batch> batches
       │                        // batches[0].sequence_groups_ 指向各 Request 的
       │                        // SequencesGroup，间接持有所有 Sequence 的 mm_data_
       │
       ├─② engine_->step(batches)
       │    │  fixed_steps_scheduler.cpp:361
       │    │
       │    └─ RecEngine::step(batches)
       │         │  rec_engine.cpp:233
       │         │
       │         └─ pipeline_->step(batches)
       │              │  虚函数分发
       │              │
       │              └─ OneRecPrefillOnlyEnginePipeline::step(batches)
       │                   │  rec_engine.cpp:697
       │                   │
       │                   └─③ engine_.workers_[0]->prepare_inputs(batches[0])
       │                        │  rec_engine.cpp:706
       │                        │  // batches[0] 就是上面构建的 Batch，
       │                        │  // 内含 sequence_groups_ → Sequence（含 mm_data_）
       │                        │
       │                        └─ Worker::prepare_inputs(batch)
       │                             │  worker.cpp:121
       │                             │
       │                             └─ RecWorkerImpl::prepare_inputs(batch)
       │                                  │  rec_worker_impl.cpp:3093
       │                                  │
       │                                  └─ work_pipelines_[0]->prepare_inputs(batch)
       │                                       │  rec_worker_impl.cpp:3095
       │                                       │
       │                                       └─ OneRecWorkPipeline::prepare_inputs(batch)
       │                                            │  rec_worker_impl.cpp:645
       │                                            │
       │                                            └─ batch.prepare_rec_forward_input(...)
       │                                                 │  rec_worker_impl.cpp:651
       │                                                 │
       │                                                 └─④ Batch::prepare_rec_forward_input()
       │                                                      │  batch.cpp:155
       │                                                      │
       │                                                      ├─ RecBatchInputBuilder::create(rec_type, ...)
       │                                                      │    │  batch.cpp:179
       │                                                      │    │
       │                                                      │    └─ switch(rec_type):
       │                                                      │         rec_batch_input_builder.cpp:41
       │                                                      │         case kOneRec → new OneRecBatchInputBuilder(...)
       │                                                      │         rec_batch_input_builder.cpp:55
       │                                                      │
       │                                                      └─ builder->build_rec_forward_input()
       │                                                           │  batch.cpp:189
       │                                                           │
       │                                                           └─⑤ OneRecBatchInputBuilder::build_rec_forward_input()
       │                                                                │  onerec_batch_input_builder.cpp:103
       │                                                                │
       │                                                                ├─ 遍历 sequence_groups_ 中的每个 sequence:
       │                                                                │    onerec_batch_input_builder.cpp:200-224
       │                                                                │
       │                                                                │    ├─ auto mm_data = sequence->mm_data();
       │                                                                │    │    从 Sequence.mm_data_ 取出 MMData（拷贝）
       │                                                                │    │    onerec_batch_input_builder.cpp:211
       │                                                                │    │
       │                                                                │    ├─ mm_data.get<torch::Tensor>("sparse_embedding")
       │                                                                │    │    从 MMDict 中按 key 查找 torch::Tensor
       │                                                                │    │    onerec_batch_input_builder.cpp:213
       │                                                                │    │    └─ cache_data.encoder_sparse_embeddings.push_back(...)
       │                                                                │    │         onerec_batch_input_builder.cpp:215
       │                                                                │    │
       │                                                                │    └─ mm_data.get<torch::Tensor>("decoder_context_embedding")
       │                                                                │         onerec_batch_input_builder.cpp:219
       │                                                                │         └─ cache_data.decoder_context_embeddings.push_back(...)
       │                                                                │              onerec_batch_input_builder.cpp:222
       │                                                                │
       │                                                                ├─ 拼接 encoder sparse embedding:
       │                                                                │    onerec_batch_input_builder.cpp:944-947
       │                                                                │    onerec_params.encoder_sparse_embedding =
       │                                                                │        torch::cat(cache_data.encoder_sparse_embeddings, dim=0);
       │                                                                │    // 多个 sequence 的 embedding 沿第 0 维拼接
       │                                                                │
       │                                                                ├─ 拼接 decoder context embedding:
       │                                                                │    onerec_batch_input_builder.cpp:950-1005
       │                                                                │    onerec_params.decoder_context_embedding =
       │                                                                │        torch::cat(cache_data.decoder_context_embeddings, dim=0);
       │                                                                │    // 或按 [bs, group_width, ctx+seq, hidden] reshape
       │                                                                │
       │                                                                └─ 返回 ForwardInput
       │
       └─⑥ OneRecWorkPipeline::prepare_work_before_execute()
            │  rec_worker_impl.cpp:658
            │
            └─ inputs.to(runtime_.worker.device(), ...)
                 │  rec_worker_impl.cpp:424  ← H2D
                 │
                 ├─ ForwardInput::to(device)
                 │    forward_params.h:418
                 │
                 └─ OneRecModelInputParams::to(device)
                      model_input_params.h:79
                      ├─ encoder_sparse_embedding.to(device)  → NPU
                      └─ decoder_context_embedding.to(device) → NPU
```

**数据流转小结**：`Batch` 本身不拷贝 tensor 数据，它只持有 `Sequence*` 指针（通过 `sequence_groups_`）。真正的 MMData tensor 始终驻留在 `Sequence.mm_data_` 中，直到 `OneRecBatchInputBuilder` 在步骤⑤中按 key 取出并 `torch::cat` 拼接为 batch tensor，再在步骤⑥中 H2D 搬到 device。

#### 输出链路（完整调用链）

模型推理完成后，输出数据从 NPU tensor 经过采样、token 回写、结果封装，最终序列化为 HTTP 响应返回客户端。

```
OneRecWorkPipeline::step()
  │  rec_worker_impl.cpp:715
  │
  │  模型前向完成后:
  │
  ├─⑦ hidden_states → logits
  │    rec_worker_impl.cpp:832-836
  │    logits = runtime_.model->logits(hidden_states, selected_token_idxes)
  │    │
  │    └─ RecForCausalLMImplBase::logits()
  │         rec_model_base.h:56
  │         ├─ h = hidden_states * scale_factor  (tie_word_embeddings 时)
  │         ├─ h = h.index_select(0, selected_idxes)
  │         └─ return lm_head_(h)  →  logits Tensor [num_seq, vocab_size]
  │
  ├─⑧ logits → SampleOutput
  │    rec_worker_impl.cpp:845-851
  │    sample_output = rec_sampler_->forward(logits, sampling_params, filter_mask)
  │    │
  │    └─ RecSampler::forward()
  │         ├─ top_k / top_p / temperature 采样
  │         ├─ filter_mask 约束解码（可选）
  │         └─ 返回 SampleOutput { next_tokens, logprobs, top_tokens, top_logprobs }
  │              // 此时 next_tokens 仍在 NPU 上
  │
  ├─ 构造 ForwardOutput:
  │    rec_worker_impl.cpp:838-851
  │    output.sample_output = sample_output
  │    output.logits = logits
  │    output.do_sample = sampling_params.do_sample
  │    return output
  │
  └─ 返回 ForwardOutput (NPU tensor)
       │
       ▼
OneRecPrefillOnlyEnginePipeline::get_model_output()
  │  rec_engine.cpp:758
  │
  ├─⑨ D2H: sample_output 从 NPU 搬到 CPU
  │    rec_engine.cpp:790-805
  │    sample_output.next_tokens = safe_to(sample_output.next_tokens, torch::kCPU)
  │    sample_output.logprobs    = safe_to(sample_output.logprobs, torch::kCPU)
  │    sample_output.top_tokens  = safe_to(sample_output.top_tokens, torch::kCPU)
  │    sample_output.top_logprobs = safe_to(sample_output.top_logprobs, torch::kCPU)
  │    // Device synchronize 确保 D2H 完成
  │    Device(engine_.workers_[0]->device()).synchronize_default_stream()
  │
  └─ 返回 ForwardOutput (CPU tensor)
       │
       ▼
OneRecPrefillOnlyEnginePipeline::step()
  │  rec_engine.cpp:697
  │
  │  prefill 阶段:
  ├─⑩ batches[0].process_sample_output(prefill_output.sample_output, false)
  │    rec_engine.cpp:719
  │
  │  decode 阶段 (kRecDecodeSteps 轮循环):
  │  for (i = 0; i < kRecDecodeSteps; ++i):
  │    forward_inputs = workers_[0]->prepare_inputs(batches[0])
  │    decode_output = get_model_output(forward_inputs)
  │    batches[0].process_sample_output(decode_output.sample_output, false, ...)
  │    rec_engine.cpp:741-744
  │
  └─ batches[0].finish()
       rec_engine.cpp:749
       │
       ▼
Batch::process_sample_output(sample_output)
  │  batch.cpp:631
  │
  ├─ 遍历 output_targets_（每个 target 映射到一个 Sequence*）:
  │    batch.cpp:650-681
  │    for (output_idx = 0; output_idx < output_targets_.size(); ++output_idx) {
  │        seq = output_targets_[output_idx].sequence;
  │
  │        token = build_token(output_idx,
  │            sample_output.next_tokens,     // CPU Int tensor
  │            sample_output.logprobs,
  │            sample_output.top_tokens,
  │            sample_output.top_logprobs);
  │
  │        append_token_for_sequence(seq, token, ...)
  │    }
  │
  └─ append_token_for_sequence():
       batch.cpp:713-735
       └─ seq->append_token(token)
            │  sequence.cpp:325
            │
            ├─ tokens_[num_tokens_++] = token.id    // 写入 Sequence.tokens_
            ├─ token_to_count_map_[token.id]++      // 更新频率（repetition penalty 用）
            ├─ logprob_state_->update_logprob(...)   // 记录 logprob（可选）
            └─ finish_status_invalidated_ = true     // 标记需要重新检查停止条件
                 │
                 └─ Sequence::finished() 被调用时:
                      sequence.cpp:738-745
                      stopping_checker->check(tokens(), num_prompt_tokens_)
                      → 检查 stop_token / EOS / max_length
                      → 命中则 finished_ = true, finish_reason_ = STOP/LENGTH
       │
       ▼
Batch::finish()
  │  batch.cpp:816
  │
  ├─ for (auto* sg : sequence_groups_) sg->finish()
  │    sequences_group.cpp:425-429
  │    └─ for (auto& seq : sequences_) seq->finish()
  │
  └─ for (auto* seq : sequences) seq->finish()
       sequence.cpp:821-827
       ├─ finished_ = true
       └─ finish_reason_ = STOP (若未设置)
       │
       ▼
FixedStepsScheduler::step() 的 lambda 后处理
  │  fixed_steps_scheduler.cpp:361-378
  │
  │  engine_->step(batches) 返回后:
  │
  ├─ for (auto& request : requests):
  │    request->update_connection_status()     // 检查客户端是否断开
  │    if (request->finished() || request->cancelled()):
  │        kv_cache_manager_->deallocate(request.get())  // 释放 KV cache blocks
  │        finished_requests.emplace_back(request)
  │
  └─ response_processor_->process_completed_requests(finished_requests)
       │  fixed_steps_scheduler.cpp:377
       │
       ▼
AsyncResponseProcessor::process_completed_requests()
  │  async_response_processor.cpp:166
  │
  └─ process_completed_request(request)
       │  async_response_processor.cpp:77
       │
       │  在 response_threadpool_ 中异步执行:
       │
       ├─⑪ RequestOutput = request->generate_output(*tokenizer_)
       │    │  async_response_processor.cpp:103-104
       │    │
       │    └─ Request::generate_output()
       │         request.cpp:156-183
       │         ├─ 统计 usage (prompt_tokens, generated_tokens, total_tokens)
       │         ├─ output.request_id = request_id_
       │         ├─ output.status = Status(OK)
       │         ├─ output.finished = finished()
       │         │
       │         └─ sequences_group_->generate_outputs(output.outputs, tokenizer)
       │              │  sequences_group.cpp:97
       │              │
       │              └─ for each sequence:
       │                   seq->generate_output(tokenizer)
       │                   │  sequence.cpp:617
       │                   │
       │                   └─ generate_onerec_output(ids, size, tokenizer, output)
       │                        │  sequence.cpp:648 → 165
       │                        │
       │                        ├─ output.token_ids = ids.slice(num_prompt_tokens_, size)
       │                        │    // 截取生成部分的 token IDs
       │                        │
       │                        ├─ output.token_ids_logprobs = logprob_state_->get_logprobs()
       │                        │    // 可选: 每 token 的 logprob
       │                        │
       │                        └─ 若 enable_convert_tokens_to_item:
       │                             sequence.cpp:190-223
       │                             tokenizer.decode(token_slice) → item_ids
       │                             output.item_ids = item_ids.front()
       │                             output.item_ids_list = item_ids
       │                             // OneRec 特有: token IDs 解码为推荐 item ID
       │
       ├─⑫ request->state().output_func(req_output)
       │    │  async_response_processor.cpp:106
       │    │
       │    │  output_func 是在请求构建时设置的 OutputCallback lambda
       │    │  即 RecCompletionServiceImpl 中捕获的闭包:
       │    │  rec_completion_service_impl.cpp:327-354
       │    │
       │    └─ send_result_to_client_brpc_rec(call, request_id, created_time, model, req_output)
       │         │  rec_completion_service_impl.cpp:83
       │         │
       │         ├─ 填充 proto::CompletionResponse:
       │         │    response.set_object("text_completion")
       │         │    response.set_id(request_id)
       │         │    response.set_model(model)
       │         │
       │         ├─ 填充 choices:
       │         │    for (auto& output : req_output.outputs):
       │         │        choice.set_index(output.index)
       │         │        choice.set_text(output.text)
       │         │        choice.set_finish_reason(output.finish_reason)
       │         │
       │         ├─ 填充 usage:
       │         │    proto_usage.set_prompt_tokens / completion_tokens / total_tokens
       │         │
       │         ├─⑬ 填充 output_tensors (OneRec 特有):
       │         │    rec_completion_service_impl.cpp:116-253
       │         │
       │         │    output_tensor.set_name("rec_result")
       │         │
       │         │    若 enable_convert_tokens_to_item:
       │         │        output_tensor.set_datatype(INT64)
       │         │        for each item in outputs:
       │         │            output_context.mutable_int64_contents()->Add(item_id)
       │         │        // 推荐 item ID 写入 int64_contents
       │         │
       │         │    否则:
       │         │        output_tensor.set_datatype(INT32)
       │         │        for each output in outputs:
       │         │            context.mutable_int_contents()->Add(token_ids)
       │         │        // 原始 token IDs 写入 int_contents
       │         │
       │         │    可选: sku_logprobs tensor
       │         │
       │         └─⑭ call->write_and_finish(response)
       │              rec_completion_service_impl.cpp:255
       │              │
       │              └─ 序列化为 JSON → 写入 HTTP 响应体 → 关闭连接
       │
       └─ 请求完成，Request 对象析构
            → Sequence 析构 → mm_data_ 释放 → Tensor 引用计数归零
```

#### MMData 生命周期总结

| 阶段 | 存储位置 | 数据状态 |
|------|----------|----------|
| API 解析 | `proto::InferInputTensor.fp32_contents` | protobuf 内存，fp32 |
| 请求构建 | `MMData` → `RequestState.mm_data` | CPU bf16 Tensor，独立内存 |
| Sequence 创建 | `Sequence.mm_data_` | 从 RequestState 拷贝，CPU bf16 Tensor |
| Batch 打包 | `cache_data` → `OneRecModelInputParams` | torch::cat 拼接，CPU bf16 Tensor |
| H2D 传输 | `ForwardInput.to(device)` | NPU bf16 Tensor |
| 模型前向 | `OneRecModelImpl::forward()` | NPU bf16 Tensor |
| 请求完成 | `Sequence` 析构 | `mm_data_` 释放，Tensor 引用计数归零 |

### B.4 OneRec 特有数据流总结

```
输入侧:
  HTTP JSON → proto.input_tensors (fp32)
           → MMData{"encoder_sparse_embedding": CPU bf16 Tensor,
                     "decoder_context_embedding": CPU bf16 Tensor}
           → Sequence.mm_data_
           → OneRecModelInputParams.encoder_sparse_embedding (→ NPU)
           → OneRecModelInputParams.decoder_context_embedding (→ NPU)

输出侧:
  OneRecModelImpl::forward() → hidden_states (NPU)
           → logits(hidden_states, selected_token_idxes) → logits (NPU)
           → rec_sampler_->sample(logits) → SampleOutput.next_tokens (NPU)
           → Batch::process_sample_output → Sequence.tokens_ (CPU)
           → Sequence::generate_onerec_output → item_ids / item_ids_list (CPU)
           → proto.output_tensors["rec_result"] (int64_contents)
           → HTTP JSON Response
```
