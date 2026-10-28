# dqm-mini-llama

一个用 C++17 从零写的 LLaMA 式推理引擎：自己解析 GGUF、自己实现 BPE 分词与前向计算，CPU 与 CUDA 两套后端可切换，
能在本地笔记本上直接跑通 Qwen2-0.5B（Q8_0 量化）的对话。

项目目标不是"再造一个 llama.cpp"，而是把推理链路上的每一块自己写一遍：张量、算子、量化、KV cache、
采样、后端接口、精度校验，都得说得清为什么这么做。

## 能做什么

- **模型格式**：手写 GGUF 二进制解析（元数据、张量表、对齐），加载 Qwen2 权重并做张量映射；
- **分词**：自实现 BPE，支持 GGUF 词表与特殊标记，按 Qwen2 的 chat template 拼对话；
- **推理**：Prefill / Decode 两段执行流，KV cache + RoPE 位置编码，前缀命中只 prefill 新增 token；
- **量化**：F32、Q8_0、Q4_0 三条权重路径，标量实现为准，SIMD 版本与标量逐位对齐；
- **并行**：多线程 matmul（线程局部累加）+ AVX2/FMA 向量化，四档 matmul 模式可切换对比；
- **CUDA 后端**：手写 RMSNorm、SiLU、Elementwise、RoPE、注意力与量化 matmul 核函数，
  `CudaTensor` 常驻显存，热路径上不再搬激活值；
- **可插拔后端**：CMake 开关隔离 CPU / CUDA 构建环境，不开 CUDA 也能完整编译与测试；
- **工程闭环**：GoogleTest 用例 + CTest + ASan 构建选项 + CLI smoke，`scripts/` 下还有 logits 对照与基准脚本。

## 性能（Qwen2-0.5B-Instruct-Q8_0）

环境：Intel i7-13650HX（14 核 20 线程）、RTX 4060 Laptop、远程 RTX 4090D；4 轮对话共 prompt 863 tokens、
生成 198 tokens，统计不含模型加载。完整报告见 [`docs/benchmarks/2026-10-28-qwen2-0.5b-benchmark.md`](docs/benchmarks/2026-10-28-qwen2-0.5b-benchmark.md)。

| 配置 | tokens/s | 相对 naive |
|---|---|---|
| CPU · naive（单线程标量） | 2.58 | 1.00× |
| CPU · SIMD（单线程 AVX2） | 5.97 | 2.31× |
| CPU · threaded（20 线程标量） | 6.25 | 2.42× |
| CPU · threaded-SIMD（20 线程 AVX2） | **6.98** | **2.71×** |
| CUDA · RTX 4060 Laptop | 38.01 | 5.4×（对最快 CPU） |
| CUDA · RTX 4090D | **60.95** | **8.7×**（对最快 CPU） |

一句话结论：CPU 上 decode 卡在内存带宽而不是算力，SIMD 与多线程叠加只有 2.7 倍；换到 GPU 才是量级变化。

## 构建

```bash
# CPU（Release，x86_64 上自动开 -mavx2 -mfma）
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 打开 ASan（越界 / 悬垂指针排查）
cmake -B build-asan -DDQM_LLAMA_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-asan -j

# 带 CUDA 后端（需要 CUDA Toolkit，默认按 sm_89 编译，可用 -DCMAKE_CUDA_ARCHITECTURES 覆盖）
cmake -B build-cuda -DDQM_LLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j
```

## 运行

```bash
# 对话（默认 CPU、threaded-simd）
./build/dqm-llama --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf --prompt "你好，简单介绍一下你自己" --n-predict 64

# 指定后端与并行方式
./build/dqm-llama --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
    --backend cpu --matmul-mode threaded-simd --threads 20 --n-predict 64

# GPU
./build/dqm-llama --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf --backend cuda --n-predict 64

# 看设备信息 / 导出 logits 做精度对照
./build/dqm-llama --cuda-info
./build/dqm-llama --model <model> --prompt "hi" --dump-logits
```

常用参数：`--backend cpu|cuda`、`--matmul-mode naive|simd|threaded|threaded-simd`、`--threads N`、
`--n-predict N`、`--quant`、`--raw-prompt`（不走 chat template）、`--config`、`--dump-logits`。
`--help` 有完整说明。

## 测试

```bash
cd build && ctest --output-on-failure     # 全量：GoogleTest 用例 + CUDA 冒烟（开了 CUDA 才有）
./build/test-Gtest --gtest_filter='Quant*' # 单跑某组
```

- `tests/` 下是逐模块的 gtest 用例：GGUF 解析、BPE、前向、KV cache、量化、采样、线程池、后端接口等；
- 精度以 **CPU logits 为基准**：`scripts/compare_logits.py`、`scripts/reference_forward.py`、
  `scripts/make_golden.py` 负责生成与比对；
- `scripts/` 里还有 CLI 冒烟与基准脚本（`test_cli_help.py`、`test_cuda_benchmarks.py` 等）。

## 目录

```
include/dqm_llama/   公共头：张量、GGUF、量化、KV cache、后端接口、采样、CUDA 封装
src/                 实现；cuda_*.cc/.cu 为 CUDA 后端
tests/               GoogleTest 用例
scripts/             模型/词表导出、logits 对照、基准与冒烟脚本
docs/benchmarks/     性能测试报告
third_party/         googletest
models/              本地模型权重（不入库）
```

## 已知限制

- **Prefill 没有批量收益**：`ForwardBatch` 目前逐 token 调 `ForwardToken`，602 token 的首轮 prefill 要几秒到几分钟，
  批量 GEMM 是最值得做的结构性优化；
- **CUDA 侧偏 kernel 效率**：4090D 只比 4060 快 1.6 倍，说明小 batch 下 kernel 启动与同步占了大头，需要算子融合与批量；
- **tokenize 偏慢**：每轮约 0.75–1.04 s，值得单独排查；
- 目前只对齐 Qwen2 系列（chat template、张量命名），其它模型要改映射。

## 许可与致谢

本人独立实现；设计与思路参考 `llama.cpp`（GGUF 格式、量化方案、算子切分）与相关公开论文。
模型权重不入库，需自行下载后放在 `models/chat/`。
