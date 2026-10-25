#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "dqm_llama/batch.h"
#include "dqm_llama/backend.h"
#include "dqm_llama/chat.h"
#include "dqm_llama/context.h"
#include "dqm_llama/debug.h"
#include "dqm_llama/forward.h"
#include "dqm_llama/gguf.h"
#include "dqm_llama/loader.h"
#include "dqm_llama/matmul_dispatch.h"
#include "dqm_llama/model.h"
#include "dqm_llama/prompt_builder.h"
#include "dqm_llama/request_context.h"
#include "dqm_llama/gguf_loader.h"
#include "dqm_llama/gguf_tokenizer.h"
#include "dqm_llama/sampler.h"
#include "dqm_llama/terminal.h"
#include "dqm_llama/tokenizer.h"
#include "dqm_llama/thread_pool.h"

namespace dqm {

namespace {

// ── 参数解析 ────────────────────────────────────────────────────────────

// 解析十进制整数参数（范围 [0, 1000000]，拒绝负数）。
bool ParseIntArg(const char* text, int& value) {
  char* end = nullptr;
  int64_t parsed = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  if (parsed < 0 || parsed > 1000000) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

// 解析非负浮点参数（范围 [0, 10000]，拦 inf / NaN / ERANGE）。
bool ParseFloatArg(const char* text, float& value) {
  char* end = nullptr;
  errno = 0;
  double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    return false;
  }
  if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 ||
      parsed > 10000.0) {
    return false;
  }
  value = static_cast<float>(parsed);
  return true;
}

// 解析非负无符号整数参数（显式拒绝负号 —— strtoull 会把它回绕成巨大值）。
bool ParseUintArg(const char* text, unsigned int& value) {
  if (text[0] == '-' || text[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  uint64_t parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  value = static_cast<unsigned int>(parsed);
  return true;
}

// ── 后端参数 ────────────────────────────────────────────────────────────

bool ParseBackendArg(const char* text, BackendConfig& config) {
  BackendKind kind;
  if (!ParseBackendKind(text, kind)) {
    return false;
  }
  config.kind = kind;
  return true;
}

bool ParseDeviceArg(const char* text, BackendConfig& config) {
  int device_id = 0;
  if (!ParseIntArg(text, device_id) || device_id < 0) {
    return false;
  }
  config.device_id = device_id;
  config.device_id_set = true;
  return true;
}

bool ValidateBackendOrPrint(const BackendConfig& config) {
  try {
    ValidateBackend(config);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Backend setup failed: " << e.what() << "\n";
    return false;
  }
}

// 后端确定之后把权重复制到设备（CPU 后端直接返回 true）。
bool PrepareCudaWeightsOrPrint(MiniLlamaModel& model,
                               const BackendConfig& config) {
  if (config.kind != BackendKind::kCuda) {
    return true;
  }
  try {
    UploadModelWeightsToCuda(model, config.device_id);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "CUDA weight upload failed: " << e.what() << "\n";
    return false;
  }
}

std::string FormatMb(size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
  return out.str();
}

void PrintCudaWeightSummary(const MiniLlamaModel& model,
                            const std::string& indent = "") {
  std::cout << indent
            << "uploaded weights: " << ModelCudaUploadedWeightCount(model)
            << "\n";
  std::cout << indent << "gpu memory used: " << ModelCudaMemoryBytes(model)
            << " bytes (" << FormatMb(ModelCudaMemoryBytes(model)) << ")\n";
}

struct LinearWeightTypeCounts {
  size_t f32 = 0;
  size_t q8_0 = 0;
  size_t q4_0 = 0;
  size_t q4_1 = 0;

  size_t total() const { return f32 + q8_0 + q4_0 + q4_1; }
};

void CountLinearWeight(const QuantizedTensor& weight,
                       LinearWeightTypeCounts& counts) {
  switch (weight.type) {
    case QuantType::kF32:
      counts.f32 += 1;
      return;
    case QuantType::kQ80:
      counts.q8_0 += 1;
      return;
    case QuantType::kQ40:
      counts.q4_0 += 1;
      return;
    case QuantType::kQ41:
      counts.q4_1 += 1;
      return;
  }
}

LinearWeightTypeCounts ModelLinearWeightTypeCounts(
    const MiniLlamaModel& model) {
  LinearWeightTypeCounts counts;
  CountLinearWeight(model.lm_head, counts);
  for (const auto& lw : model.layers) {
    CountLinearWeight(lw.wq, counts);
    CountLinearWeight(lw.wk, counts);
    CountLinearWeight(lw.wv, counts);
    CountLinearWeight(lw.wo, counts);
    CountLinearWeight(lw.w_gate, counts);
    CountLinearWeight(lw.w_up, counts);
    CountLinearWeight(lw.w_down, counts);
  }
  return counts;
}

void PrintCudaExecutionSummary(const MiniLlamaModel& model,
                               const std::string& indent = "") {
  PrintCudaWeightSummary(model, indent);
  LinearWeightTypeCounts counts = ModelLinearWeightTypeCounts(model);
  size_t uploaded_f32 = ModelCudaUploadedF32WeightCount(model);
  size_t uploaded_q8_0 = ModelCudaUploadedQ80WeightCount(model);
  size_t uploaded_q4_0 = ModelCudaUploadedQ40WeightCount(model);
  size_t uploaded_q4_1 = ModelCudaUploadedQ41WeightCount(model);
  std::cout << indent << "cuda f32 Linear weights: " << uploaded_f32 << "/"
            << counts.total() << "\n";
  if (counts.q8_0 > 0) {
    std::cout << indent << "cuda q8_0 Linear weights: " << uploaded_q8_0 << "/"
              << counts.q8_0 << "\n";
  }
  if (counts.q4_0 > 0) {
    std::cout << indent << "cuda q4_0 Linear weights: " << uploaded_q4_0 << "/"
              << counts.q4_0 << "\n";
  }
  if (counts.q4_1 > 0) {
    std::cout << indent << "cuda q4_1 Linear weights: " << uploaded_q4_1 << "/"
              << counts.q4_1 << "\n";
  }
  size_t q8_0_fallback =
      counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
  size_t q4_0_fallback =
      counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
  size_t q4_1_fallback =
      counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;
  if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
    std::cout << indent
              << "cuda fallback: unsupported or missing quantized Linear "
                 "weights use CPU path"
              << " (Q8_0=" << q8_0_fallback << ", Q4_0=" << q4_0_fallback
              << ", Q4_1=" << q4_1_fallback << ")\n";
  }
}

void PrintCudaRuntimeSummary(const MiniLlamaModel& model,
                             const std::string& indent = "") {
  std::cout << indent << "cuda Linear calls: " << ModelCudaLinearCalls(model)
            << "\n";
  std::cout << indent
            << "cuda activation calls: " << ModelCudaActivationCalls(model)
            << "\n";
  std::cout << indent
            << "cuda attention calls: " << ModelCudaAttentionCalls(model)
            << "\n";
  std::cout << indent << "cpu attention fallback calls: "
            << ModelCudaAttentionCpuFallbacks(model) << "\n";
  std::cout << indent
            << "cuda kv write bytes: " << ModelCudaKvCacheWriteBytes(model)
            << "\n";
  std::cout << indent
            << "cuda kv read bytes: " << ModelCudaKvCacheReadBytes(model)
            << "\n";
  std::cout << indent
            << "host->device copies: " << ModelCudaHostToDeviceCopies(model)
            << " (" << ModelCudaHostToDeviceBytes(model) << " bytes)\n";
  std::cout << indent
            << "device->host copies: " << ModelCudaDeviceToHostCopies(model)
            << " (" << ModelCudaDeviceToHostBytes(model) << " bytes)\n";
}

// --quant q8_0|q4_0：把 Linear 权重就地转成量化态（必须在 CUDA 上传之前）。
void ApplyQuantOverride(MiniLlamaModel& model, const std::string& quant_type) {
  if (quant_type.empty()) {
    return;
  }
  if (quant_type == "q8_0") {
    QuantizeModelToQ80(model);
    return;
  }
  if (quant_type == "q4_0") {
    QuantizeModelToQ40(model);
    return;
  }
  throw std::runtime_error("unsupported quant type: " + quant_type);
}

// --dump-logits：把每步 logits 原样写到二进制文件（Python 对拍用）。
void DumpLogits(const Tensor& logits, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open logits dump file: " + path);
  }
  out.write(reinterpret_cast<const char*>(logits.data.data()),
            static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("failed to write logits to: " + path);
  }
}

void DumpGeneratedTokens(const std::vector<int>& generated,
                         const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open generation token dump file: " +
                             path);
  }
  for (size_t i = 0; i < generated.size(); ++i) {
    if (i > 0) {
      out << " ";
    }
    out << generated[i];
  }
  out << "\n";
  if (!out.good()) {
    throw std::runtime_error("failed to write generation tokens to: " + path);
  }
}

void EnsureDumpDirectory(const std::string& path) {  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error) {
    throw std::runtime_error("failed to create logits dump directory: " + path +
                             ": " + error.message());
  }
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("logits dump path is not a directory: " + path);
  }
}

Tensor RunLogitsForTokens(const MiniLlamaModel& model,
                          const std::vector<int>& tokens) {
  MiniLlamaContext ctx(&model);
  MiniBatch batch = MiniBatch::FromTokens(tokens, 0);
  return ForwardBatch(ctx, model, batch);
}

std::pair<float, float> LogitsError(const Tensor& baseline,
                                    const Tensor& candidate) {
  if (baseline.shape != candidate.shape) {
    throw std::runtime_error(
        "LogitsError: shape mismatch, baseline=" + baseline.ShapeStringShort() +
        ", candidate=" + candidate.ShapeStringShort());
  }
  float max_err = 0.0f;
  float sum_err = 0.0f;
  for (size_t i = 0; i < baseline.size(); ++i) {
    float err = std::abs(baseline.data[i] - candidate.data[i]);
    max_err = std::max(max_err, err);
    sum_err += err;
  }
  return {max_err, sum_err / static_cast<float>(baseline.size())};
}

// 打印 run 子命令的用法（-h 或参数出错时调用）。
void PrintRunUsage(const char* program) {
  std::cout
      << "Usage: " << program << " run <model_dir> [options]\n"
      << "\n"
      << "Options:\n"
      << "  --temperature <float> Sampling temperature (default: 0 = greedy)\n"
      << "  --top-k <int>         Top-k cutoff (default: 0 = disabled)\n"
      << "  --seed <uint>         RNG seed (default: 0 = random_device)\n"
      << "  -n, --n-predict <int> Max tokens per turn (default: 64)\n"
      << "  --tokenizer <path>    Tokenizer file (default: auto-detect)\n"
      << "  --request             Show per-request trace info\n"
      << "  --matmul-mode <name>  Matmul strategy: "
         "naive|threaded|simd|threaded-simd\n"
      << "  --threads <int>       Number of threads (0 = auto)\n"
      << "  --backend cpu|cuda    Execution backend (default: cpu)\n"
      << "  --device <int>        CUDA device id for --backend cuda (default: 0)\n"
      << "  -h, --help            Show this help\n";
}

// 非交互单次生成的用法（冒烟测试用）。
void PrintGenerateUsage(const char* program) {
  std::cout
      << "Usage: " << program << " generate [model_dir] [options]\n"
      << "\n"
      << "Options:\n"
      << "  --model <path|dir>    Model weights (.bin/.gguf) or directory\n"
      << "  --config <path>       Model config JSON (overrides <dir>/model.json)\n"
      << "  --raw-prompt          Encode the prompt directly (skip chat template)\n"
      << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <int> Number of tokens to generate (default: 16)\n"
      << "  --temperature <float> Sampling temperature (default: 0 = greedy)\n"
      << "  --top-k <int>         Top-k cutoff (default: 0 = disabled)\n"
      << "  --seed <uint>         RNG seed (default: 0 = random_device)\n"
      << "  --tokenizer <path>    Tokenizer file (default: auto-detect)\n"
      << "  --quant q8_0|q4_0     Quantize loaded Linear weights first\n"
      << "  --dump-logits <dir>   Dump per-step logits to a directory\n"
      << "  --matmul-mode <name>  Matmul strategy: "
         "naive|threaded|simd|threaded-simd\n"
      << "  --threads <int>       Number of threads (0 = auto)\n"
      << "  --backend cpu|cuda    Execution backend (default: cpu)\n"
      << "  --device <int>        CUDA device id for --backend cuda (default: 0)\n"
      << "  -h, --help            Show this help\n";
}

// bench 子命令的用法（跑 prefill + decode，打吞吐与内存统计）。
void PrintBenchUsage(const char* program) {
  std::cout
      << "Usage: " << program << " bench [model_dir] [options]\n"
      << "\n"
      << "Options:\n"
      << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <int> Number of tokens to generate (default: 64)\n"
      << "  --seed <uint>         RNG seed (default: 0 = random_device)\n"
      << "  --tokenizer <path>    Tokenizer file (default: auto-detect)\n"
      << "  --quant q8_0|q4_0     Quantize loaded Linear weights first\n"
      << "  --verbose             Print debug dumps after each step\n"
      << "  --matmul-mode <name>  Matmul strategy: "
         "naive|threaded|simd|threaded-simd\n"
      << "  --threads <int>       Number of threads (0 = auto)\n"
      << "  --backend cpu|cuda    Execution backend (default: cpu)\n"
      << "  --device <int>        CUDA device id for --backend cuda (default: 0)\n"
      << "  -h, --help            Show this help\n";
}

// ── 模型装配 ────────────────────────────────────────────────────────────

// 一次「加载」的产物：模型 + 词表 + 聊天模板。
struct LoadedModel {
  MiniLlamaModel model;
  std::unique_ptr<ITokenizer> tokenizer;
  // tokenizer 来自 GGUF 时，从它读到的聊天模板（Jinja 串 / "qwen2" / 空）。
  std::string chat_template;
};

// 把 manifest 里的词表路径（相对 model.json 所在目录）解析成绝对路径。
std::string ResolveManifestTokenizerPath(const std::string& config_path) {
  try {
    ModelManifest manifest = ParseManifest(config_path);
    if (manifest.tokenizer.path.empty()) {
      return "";
    }
    std::filesystem::path vocab(manifest.tokenizer.path);
    if (vocab.is_absolute()) {
      return vocab.string();
    }
    return (std::filesystem::path(config_path).parent_path() / vocab).string();
  } catch (const std::exception&) {
    return "";
  }
}

// 文件是否以 GGUF magic 开头。
bool IsGgufFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }
  char magic[4];
  return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

// 在目录里找一个 .gguf（按名字排序取第一个）；没有返回 ""。
std::string FindGgufInDirectory(const std::string& path) {
  std::vector<std::filesystem::path> candidates;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
      candidates.push_back(entry.path());
    }
  }
  if (candidates.empty()) {
    return "";
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.front().string();
}

// 装配「模型 + tokenizer」：**优先 GGUF**，起不来再回落 JSON+BIN。
// path 可以是「目录」（含 .gguf 或 model.json+model.bin）、「.gguf 本身」、或「model.bin 本身」。
LoadedModel LoadModelAndTokenizer(const std::string& path,
                                  const std::string& explicit_tokenizer_path,
                                  const std::string& explicit_config_path = "") {
  LoadedModel result;

  std::filesystem::path dir =
      std::filesystem::is_directory(path)
          ? std::filesystem::path(path)
          : std::filesystem::path(path).parent_path();
  bool path_is_gguf = !std::filesystem::is_directory(path) && IsGgufFile(path);

  // 1. 找 GGUF：path 本身是 .gguf，或目录里有 .gguf。
  std::string gguf_path;
  if (path_is_gguf) {
    gguf_path = path;
  } else if (std::filesystem::is_directory(path)) {
    gguf_path = FindGgufInDirectory(path);
  }

  // 2. 优先从 GGUF 建模型；metadata 空 / 加载失败就当没有，转 JSON+BIN。
  if (!gguf_path.empty()) {
    result.model = LoadGgufModel(gguf_path);
  }

  // 3. 回落：JSON + BIN（目录里找，或 path 本身就是 .bin）。
  if (!result.model.loaded) {
    bool use_dir_layout =
        std::filesystem::is_directory(path) || path_is_gguf;
    std::string model_path =
        use_dir_layout ? (dir / "model.bin").string() : path;
    std::string config_path = explicit_config_path.empty()
                                  ? (dir / "model.json").string()
                                  : explicit_config_path;
    result.model = LoadModel(config_path, model_path);
    if (!result.model.loaded) {
      return result;
    }
  }

  // 4. tokenizer：显式 > 目录 vocab.json > GGUF 里的词表 > manifest 里的路径。
  std::string tokenizer_path = explicit_tokenizer_path;
  if (tokenizer_path.empty()) {
    std::filesystem::path auto_vocab = dir / "vocab.json";
    if (std::filesystem::exists(auto_vocab)) {
      tokenizer_path = auto_vocab.string();
    }
  }
  if (tokenizer_path.empty()) {
    tokenizer_path = gguf_path;
  }
  if (tokenizer_path.empty()) {
    tokenizer_path = ResolveManifestTokenizerPath((dir / "model.json").string());
  }

  if (IsGgufFile(tokenizer_path)) {
    result.tokenizer = CreateGgufTokenizer(tokenizer_path);
    if (result.tokenizer) {
      result.chat_template = LoadChatTemplateFromGguf(tokenizer_path);
    }
  } else {
    result.tokenizer = CreateTokenizer(tokenizer_path);
  }

  if (!result.tokenizer) {
    result.model.load_error = "Failed to load tokenizer";
    result.model.loaded = false;
  }
  return result;
}

// ── 小工具 ──────────────────────────────────────────────────────────────

// 求两个 token 序列的公共前缀长度 —— 前缀复用那个 min() 的第二个来源。
// 注意：和 RadixTree 里的同名 private 函数无关。
size_t CommonPrefixLength(const std::vector<int>& a,
                          const std::vector<int>& b) {
  size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) {
    ++i;
  }
  return i;
}

// 打印一次请求的阶段耗时追踪（每轮收尾调一次）。
void PrintRequestTrace(const RequestContext& request) {
  for (const std::string& line : FormatRequestTraceEvents(request)) {
    std::cout << line << "\n";
  }
  std::cout << FormatRequestTraceSummary(request) << "\n";
}

}  // namespace

// 交互式聊天主循环：读一行 → 分发命令 / 跑一轮推理 → 流式输出。
//正常用户输入 程序名 run 模型 -n 16
static int RunChat(int argc, char** argv) {
  
  //要不要提示
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintRunUsage(argv[0]);
    return 0;
  }
  //命令太少
  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintRunUsage(argv[0]);
    return 1;
  }

  //开始正式接收命令参数
  //先设置默认参数
  std::string model_dir = argv[2];
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  int max_response_tokens = 64;
  std::string tokenizer_path;
  bool show_request = false;
  int n_threads = 0;
  MatmulMode matmul_mode = DefaultMatmulMode();
  BackendConfig backend_config;

  //然后逐个输入参数字符串比较，提取命令
  for (int i = 3; i < argc; i ++) {
    if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      //检测并赋值
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], max_response_tokens) ||
          max_response_tokens <= 0) {
        std::cerr << "Invalid --n-predict value. Expected a positive integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads)) {
        std::cerr << "Invalid --threads value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--matmul-mode") == 0 && i + 1 < argc) {
      if (!ParseMatmulMode(argv[++i], matmul_mode)) {
        std::cerr << "Invalid --matmul-mode value. Expected one of: "
                     "naive, threaded, simd, threaded-simd.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--request") == 0) {
      show_request = true;
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintRunUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintRunUsage(argv[0]);
      return 1;
    }
  }

  SetThreadCount(n_threads);
  SetDefaultMatmulMode(matmul_mode);
  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  //加载模型
  //这个方法里有默认的tokenizer_path路径，该项可以为空
  LoadedModel lm = LoadModelAndTokenizer(model_dir, tokenizer_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  if (!PrepareCudaWeightsOrPrint(lm.model, backend_config)) {
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) { 
    std::cerr << "model vocab too small\n";
    return 1; 
  }

  //开始初始化数据结构
  MiniLlamaModel& model = lm.model;
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

  PromptBuilder builder;
  // 有 GGUF 模板就交给模板引擎，否则留空走默认 plain。
  if (!lm.chat_template.empty()) {
    builder.SetChatTemplate(lm.chat_template);
  }

  Terminal term;
  ChatSession session;
  session.sampling_params.temperature = temperature;
  session.sampling_params.top_k       = top_k;
  session.sampling_params.seed        = seed;

  if (lm.chat_template.empty()) {
    session.AddMessage("system", "You are a helpful assistant.");
  }

  //打印头部提示信息
  term.PrintMessage("dqm-mini-llama.cpp chat");                     
  term.PrintMessage("Type /help for commands, /exit to quit.\n");  
  term.PrintMessage("threads: " + std::to_string(GetThreadCount()));
  term.PrintMessage("matmul mode: " + MatmulModeName(DefaultMatmulMode()));
  term.PrintMessage("backend: " + BackendKindName(backend_config.kind));
  term.PrintMessage(BackendExecutionNote(backend_config));
  if (backend_config.kind == BackendKind::kCuda) {
    term.PrintMessage("cuda: " + CudaDeviceSummary(backend_config.device_id));
    term.PrintMessage("cuda weights: " +
                      std::to_string(ModelCudaUploadedWeightCount(model)) +
                      " (" +
                      std::to_string(ModelCudaMemoryBytes(model) /
                                     (1024 * 1024)) +
                      " MB)");
  }
  if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
    term.PrintMessage("Tiny teaching model: random weights, small context window, ...");
    term.PrintMessage("Real chat demo: ./build/<name> run models/chat -n 8\n");
  }

  //循环主骨架
  MiniLlamaContext ctx(&model);

  while (true) {
    term.PrintUserPrompt();
    std::string input = term.ReadLine();
    if (input.empty() && std::cin.eof()) {
      break;
    }

    if (input == "/help")   { 
      term.PrintHelp();   
      continue; 
    } 

    if (input == "/exit")   {
      term.PrintMessage("Goodbye.");
      break;
    }

    if (input == "/clear") {
      session.Clear();
      ctx = MiniLlamaContext(&model);
      if (lm.chat_template.empty()) {
        session.AddMessage("system", "You are a helpful assistant.");
      }
      term.PrintMessage("Chat history cleared.");
      continue;
    }

    if (input == "/stats")  { 
      term.PrintStats(session); 
      continue; 
    }

    if (input == "/params") { 
      term.PrintParams(session.sampling_params); 
      continue; 
    }

    if (!input.empty() && input[0] == '/') { 
      term.PrintMessage("Unknown command: " + input);
      continue; 
    }

    if (input.empty()) { 
      continue;
    } 

    //创建请求上下文
    RequestContext request = StartRequest("run", "cpu", model_dir);
    
    //开启一轮对话
    //先提取副本出来操作，避免中途出错无法回滚
    std::vector<ChatMessage> candidate_messages = session.messages;
    candidate_messages.push_back({"user", input});

    //拼接prompt
    auto stage_start = RequestClock::now();
    std::string prompt_text = builder.Build(candidate_messages);
    request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
                        "messages=" +
                            std::to_string(candidate_messages.size()));

    //编码，把token转化成token_id
    stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt_text);
    request.tokenize_ms = ElapsedMs(stage_start);
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                        "prompt");

    //检验长度：放不下就丢掉本轮（副本还没提交，历史仍然干净）
    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      request.SetError("prompt uses " + std::to_string(tokens.size()) +
                       " tokens, context window is " +
                       std::to_string(model.config.max_seq_len) +
                       ". Use /clear or a shorter prompt.");
      request.Finish();
      if (show_request) {
        PrintRequestTrace(request);
      }
      term.PrintMessage("Error: " + request.error);
      continue;
    }

    //本轮生成长度 = 上下文剩余空间 与 -n 取小
    int max_response = model.config.max_seq_len - static_cast<int>(tokens.size());
    if (max_response > max_response_tokens) {
      max_response = max_response_tokens;
    }
    
    //message中存的是原始信息
    session.messages = candidate_messages;

    //每轮重建取词器
    MiniSampler sampler(session.sampling_params);
    Tensor logits;
    auto start = std::chrono::steady_clock::now();

    //前缀复用(重建 ctx 只是图省事；这套机制真正施展的场景是多会话共享 KV)
    //两个来源取交集：RadixTree 登记簿说的，和 ctx 手上真有的
    size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
    size_t context_prefix_len = CommonPrefixLength(ctx.token_history, tokens);
    size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);

    if (prefix_len >= tokens.size()) {
      //整段都命中 ⇒ new_prompt_tokens 为空 ⇒ prefill 拿不到 logits
      //⇒ decode 启动不了。放弃这次复用，退回全量重算。
      prefix_len = 0;
      ctx = MiniLlamaContext(&model);
    } else if (prefix_len == 0) {
      //一个前缀都用不上。ForwardBatch 只往账本里追加，所以账本必须清零
      //—— 重建 ctx 一条语句把账本、pos、KV 一起归零。
      ctx = MiniLlamaContext(&model);
    } else if (prefix_len < ctx.token_history.size()) {
      //真正能复用的情形：只留下前缀，丢掉后面用不上的，再回退下标
      ctx.token_history.resize(prefix_len);
      ctx.pos = static_cast<int>(prefix_len - 1);
    }

    //从重复前缀之后算新的就行
    std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
    session.SetTokenHistory(tokens);//session记录完整的tokens

    MiniBatch prefill = MiniBatch::FromTokens(
        new_prompt_tokens, static_cast<int>(prefix_len));

    //prefill：只算缺的那段，位置从 prefix_len 接着编号
    stage_start = RequestClock::now();
    logits = ForwardBatch(ctx, model, prefill);
    request.prefill_ms = ElapsedMs(stage_start);
    request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
    request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                        "radix_hit=" + std::to_string(cached_prefix_len) +
                            ", prefix_reuse=" + std::to_string(prefix_len));
    ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());

    //decode
    std::vector<int> generated_ids;
    std::string stream_reply;
    int generated_count = 0;

    term.PrintAssistantPrefix();
    try {
      for (int i = 0; i < max_response; i ++) {
        //先把下一个tokenid取出来
        stage_start = RequestClock::now();
        int next_token = sampler.Sample(logits, session.sampling_params);
        request.sample_ms += ElapsedMs(stage_start);
        tokens.push_back(next_token);
        session.AppendToken(next_token);
        generated_count ++;

        //进行字符串转换和输出(每次tokenizer都全部算一次)
        if (next_token != tokenizer->eos_id()) {
          generated_ids.push_back(next_token);
          std::string current_reply = tokenizer->Decode(generated_ids);
          if (current_reply.size() > stream_reply.size()) {
            //从上次的结尾输出
            term.PrintTokenText(current_reply.substr(stream_reply.size()));
            term.Flush();
            stream_reply = current_reply;
          }
        }

        //计算下一个token
        MiniBatch decode_batch = MiniBatch::Single(
            next_token, static_cast<int>(tokens.size() - 1));
        stage_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, decode_batch);
        double decode_ms = ElapsedMs(stage_start);
        request.decode_ms += decode_ms;
        request.RecordEvent("decode", decode_ms, 1,
                            "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;

        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    } catch (const std::exception& e) {
      //推理半路抛了：ctx 处于半推进状态，绝不能登记前缀。丢本轮、保住会话。
      term.NewLine();
      request.SetError("Inference error: " + std::string(e.what()));
      request.Finish();
      if (show_request) {
        PrintRequestTrace(request);
      }
      term.PrintMessage(request.error);
      continue;
    }

    //收尾：补打尾巴、写历史、记统计、登记前缀
    std::string assistant_reply = tokenizer->Decode(generated_ids);
    if (assistant_reply.size() > stream_reply.size()) {
      //流式阶段没能吐出来的尾巴（解不完整的那点）
      term.PrintTokenText(assistant_reply.substr(stream_reply.size()));
    }
    term.NewLine();
    term.NewLine();

    auto end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    session.AddMessage("assistant", assistant_reply);
    session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                       generated_count, elapsed_ms);
    session.RecordPrefix(session.token_history);

    request.generated_tokens = generated_count;
    request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                        "generated_tokens");
    request.Finish();
    if (show_request) {
      PrintRequestTrace(request);
    }

    

  }

  return 0;
}

// ── generate：非交互单次生成（冒烟测试用）───────────────────────────────
// 给一句 prompt（当一条 user 消息，走 chat 模板），prefill + decode 跑完就退，不读 stdin。
static int RunGenerate(int argc, char** argv) {
  // 模型路径：argv[2] 若不是以 '-' 开头就当模型路径，否则用默认。
  std::string model_path = "models/tiny";
  int arg_start = 2;
  if (argc >= 3 && argv[2][0] != '-') {
    model_path = argv[2];
    arg_start = 3;
  }

  std::string prompt = "hello";
  int n_predict = 16;
  float temperature = 0.0f;
  int top_k = 0;
  std::string config_path;
  bool raw_prompt = false;
  unsigned int seed = 0;
  std::string tokenizer_path;
  std::string quant_type;
  std::string dump_logits_dir;
  int n_threads = 0;
  MatmulMode matmul_mode = DefaultMatmulMode();
  BackendConfig backend_config;

  for (int i = arg_start; i < argc; ++i) {
    if ((std::strcmp(argv[i], "-p") == 0 ||
         std::strcmp(argv[i], "--prompt") == 0) &&
        i + 1 < argc) {
      prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict)) {
        std::cerr << "Invalid --n-predict value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads)) {
        std::cerr << "Invalid --threads value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--matmul-mode") == 0 && i + 1 < argc) {
      if (!ParseMatmulMode(argv[++i], matmul_mode)) {
        std::cerr << "Invalid --matmul-mode value. Expected one of: "
                     "naive, threaded, simd, threaded-simd.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--raw-prompt") == 0) {
      raw_prompt = true;
    } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant_type = argv[++i];
      if (quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value. Supported values: q8_0, q4_0.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
      dump_logits_dir = argv[++i];
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintGenerateUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintGenerateUsage(argv[0]);
      return 1;
    }
  }

  SetThreadCount(n_threads);
  SetDefaultMatmulMode(matmul_mode);
  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  LoadedModel lm =
      LoadModelAndTokenizer(model_path, tokenizer_path, config_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  try {
    ApplyQuantOverride(lm.model, quant_type);
  } catch (const std::exception& e) {
    std::cerr << "Quantization failed: " << e.what() << "\n";
    return 1;
  }
  if (!PrepareCudaWeightsOrPrint(lm.model, backend_config)) {
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    std::cerr << "model vocab too small\n";
    return 1;
  }

  MiniLlamaModel& model = lm.model;
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

  RequestContext request =
      StartRequest("generate", BackendKindName(backend_config.kind), model_path);

  // 把 prompt 当一条 user 消息，交给 chat 模板拼成规范 prompt（与 run 同款）。
  // 模板为空时手塞 system（plain 模式不自己注入）。
  // --raw-prompt：跳过模板，直接编码原文（参考 generate 的行为，便于对拍）。
  auto stage_start = RequestClock::now();
  std::string prompt_text;
  std::vector<ChatMessage> messages;
  if (raw_prompt) {
    prompt_text = prompt;
  } else {
    PromptBuilder builder;
    if (!lm.chat_template.empty()) {
      builder.SetChatTemplate(lm.chat_template);
    }
    if (lm.chat_template.empty()) {
      messages.push_back({"system", "You are a helpful assistant."});
    }
    messages.push_back({"user", prompt});
    prompt_text = builder.Build(messages);
  }
  request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
                      "messages=" + std::to_string(messages.size()));

  stage_start = RequestClock::now();
  std::vector<int> tokens = tokenizer->Encode(prompt_text);
  request.tokenize_ms = ElapsedMs(stage_start);
  request.prompt_tokens = static_cast<int>(tokens.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                      "prompt");

  std::cout << "prompt: " << prompt << "\n";
  std::cout << "tokens: [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << tokens[i];
  }
  std::cout << "]\n";
  std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k
            << ", seed=" << seed << "\n";
  std::cout << "quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n";
  std::cout << "threads: " << GetThreadCount() << "\n";
  std::cout << "matmul mode: " << MatmulModeName(DefaultMatmulMode()) << "\n";
  std::cout << "backend: " << BackendKindName(backend_config.kind) << "\n";
  std::cout << BackendExecutionNote(backend_config) << "\n";
  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "cuda: " << CudaDeviceSummary(backend_config.device_id) << "\n";
    PrintCudaExecutionSummary(model);
  }

  if (tokens.empty()) {
    request.SetError("Empty prompt (tokenizer produced no tokens).");
    request.Finish();
    PrintRequestTrace(request);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    request.SetError("Prompt is too long for max_seq_len=" +
                     std::to_string(model.config.max_seq_len) + ".");
    request.Finish();
    PrintRequestTrace(request);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    request.SetError("Requested tokens exceed context window.");
    request.Finish();
    PrintRequestTrace(request);
    std::cerr << request.error << "\n";
    return 1;
  }

  MiniLlamaContext ctx(&model);
  SamplingParams sampling_params;
  sampling_params.temperature = temperature;
  sampling_params.top_k = top_k;
  sampling_params.seed = seed;
  MiniSampler sampler(sampling_params);

  size_t prompt_len = tokens.size();
  int step = 0;
  try {
    std::cout << "prefill...\n";
    stage_start = RequestClock::now();
    Tensor logits;
    if (!dump_logits_dir.empty()) {
      // 逐步 prefill：每步的 logits 都要落盘，不能只留最后一个 token 的。
      EnsureDumpDirectory(dump_logits_dir);
      for (size_t i = 0; i < prompt_len; ++i) {
        MiniBatch prefill_step =
            MiniBatch::FromTokens({tokens[i]}, static_cast<int>(i));
        logits = ForwardBatch(ctx, model, prefill_step);
        DumpLogits(logits, dump_logits_dir + "/logits_step" +
                               std::to_string(step) + ".bin");
        ++step;
      }
    } else {
      MiniBatch prefill = MiniBatch::FromTokens(tokens, 0);
      logits = ForwardBatch(ctx, model, prefill);
    }
    request.prefill_ms = ElapsedMs(stage_start);
    request.prefill_tokens = static_cast<int>(prompt_len);
    request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                        "batch");
    ctx.n_prefill_tokens += static_cast<int>(prompt_len);

    std::cout << "decode loop...\n";
    if (n_predict == 0) {
      std::cout << "decode loop skipped.\n";
    } else {
    stage_start = RequestClock::now();
    int next_token = sampler.Sample(logits, sampling_params);
    request.sample_ms += ElapsedMs(stage_start);
    tokens.push_back(next_token);

    for (int i = 1; i < n_predict; ++i) {
      MiniBatch decode_batch = MiniBatch::Single(
          tokens.back(), static_cast<int>(tokens.size() - 1));
      stage_start = RequestClock::now();
      logits = ForwardBatch(ctx, model, decode_batch);
      if (!dump_logits_dir.empty()) {
        DumpLogits(logits, dump_logits_dir + "/logits_step" +
                               std::to_string(step) + ".bin");
        ++step;
      }
      double decode_ms = ElapsedMs(stage_start);
      request.decode_ms += decode_ms;
      request.RecordEvent("decode", decode_ms, 1,
                          "pos=" + std::to_string(tokens.size() - 1));
      ++ctx.n_decode_tokens;
      ++request.decode_tokens;

      stage_start = RequestClock::now();
      next_token = sampler.Sample(logits, sampling_params);
      request.sample_ms += ElapsedMs(stage_start);
      tokens.push_back(next_token);
      if (next_token == tokenizer->eos_id()) {
        break;
      }
    }
    }
  } catch (const std::exception& e) {
    request.SetError("Inference failed: " + std::string(e.what()));
    request.Finish();
    PrintRequestTrace(request);
    std::cerr << request.error << "\n";
    return 1;
  }

  std::vector<int> generated(
      tokens.begin() + static_cast<std::ptrdiff_t>(prompt_len), tokens.end());
  // 末尾若停在 EOS，去掉它（与 run 同款），输出不带特殊 token。
  if (!generated.empty() && generated.back() == tokenizer->eos_id()) {
    generated.pop_back();
  }
  if (!dump_logits_dir.empty()) {
    try {
      DumpGeneratedTokens(generated,
                          dump_logits_dir + "/generation_tokens.txt");
    } catch (const std::exception& e) {
      request.SetError("Logits dump failed: " + std::string(e.what()));
      request.Finish();
      PrintRequestTrace(request);
      std::cerr << request.error << "\n";
      return 1;
    }
  }
  request.generated_tokens = static_cast<int>(generated.size());
  request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                      "generated_tokens");
  request.Finish();

  std::cout << "\ngenerated tokens: [";
  for (size_t i = 0; i < generated.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << generated[i];
  }
  std::cout << "]\n";
  std::cout << "generated text: \"" << tokenizer->Decode(generated) << "\"\n";
  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "\ncuda runtime:\n";
    PrintCudaRuntimeSummary(model, "  ");
  }
  PrintRequestTrace(request);
  return 0;
}

// ── bench：跑 prefill + decode，打吞吐与内存统计 ────────────────────────
// 给一句 prompt（直接 Encode，不走 chat 模板 —— 同参考），用 debug.h 的
// RunBenchmark 跑完，打印结果。
static int RunBench(int argc, char** argv) {
  // 模型路径：argv[2] 不以 '-' 开头就当路径，否则用默认（与 generate 同款）。
  std::string model_path = "models/tiny";
  int arg_start = 2;
  if (argc >= 3 && argv[2][0] != '-') {
    model_path = argv[2];
    arg_start = 3;
  }

  std::string prompt = "hello";
  int n_predict = 64;
  unsigned int seed = 0;
  std::string tokenizer_path;
  bool verbose = false;
  std::string quant_type;
  std::string config_path;
  int n_threads = 0;
  MatmulMode matmul_mode = DefaultMatmulMode();
  BackendConfig backend_config;

  for (int i = arg_start; i < argc; ++i) {
    if ((std::strcmp(argv[i], "-p") == 0 ||
         std::strcmp(argv[i], "--prompt") == 0) &&
        i + 1 < argc) {
      prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict) || n_predict <= 0) {
        std::cerr << "Invalid --n-predict value. Expected a positive integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads)) {
        std::cerr << "Invalid --threads value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--matmul-mode") == 0 && i + 1 < argc) {
      if (!ParseMatmulMode(argv[++i], matmul_mode)) {
        std::cerr << "Invalid --matmul-mode value. Expected one of: "
                     "naive, threaded, simd, threaded-simd.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant_type = argv[++i];
      if (quant_type != "q8_0" && quant_type != "q4_0") {
        std::cerr << "Invalid --quant value. Supported values: q8_0, q4_0.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintBenchUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintBenchUsage(argv[0]);
      return 1;
    }
  }

  SetThreadCount(n_threads);
  SetDefaultMatmulMode(matmul_mode);
  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  LoadedModel lm =
      LoadModelAndTokenizer(model_path, tokenizer_path, config_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  // 量化前留一份原始模型，用于量化误差对比。
  MiniLlamaModel baseline_model = lm.model;
  try {
    ApplyQuantOverride(lm.model, quant_type);
  } catch (const std::exception& e) {
    std::cerr << "Quantization failed: " << e.what() << "\n";
    return 1;
  }
  if (!PrepareCudaWeightsOrPrint(lm.model, backend_config)) {
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    std::cerr << "model vocab too small\n";
    return 1;
  }

  MiniLlamaModel& model = lm.model;
  std::unique_ptr<ITokenizer>& tokenizer = lm.tokenizer;

  std::vector<int> tokens = tokenizer->Encode(prompt);
  if (tokens.empty()) {
    std::cerr << "Empty prompt (tokenizer produced no tokens).\n";
    return 1;
  }
  if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
    std::cerr << "Prompt too long for max_seq_len=" << model.config.max_seq_len
              << ".\n";
    return 1;
  }
  // 生成数不越过上下文窗口（够不着就夹到剩余空间）。
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    n_predict = model.config.max_seq_len - static_cast<int>(tokens.size());
  }

  std::cout << "Benchmark: " << model_path << "\n";
  std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size()
            << " tokens)\n";
  std::cout << "  n_predict: " << n_predict << "\n";
  std::cout << "  seed: " << seed << "\n";
  std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";
  std::cout << "  quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n\n";
  std::cout << "  threads: " << GetThreadCount() << "\n\n";
  std::cout << "  matmul mode: " << MatmulModeName(DefaultMatmulMode())
            << "\n\n";
  std::cout << "  backend: " << BackendKindName(backend_config.kind) << "\n\n";
  std::cout << "  " << BackendExecutionNote(backend_config) << "\n\n";
  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "  cuda: " << CudaDeviceSummary(backend_config.device_id)
              << "\n\n";
    PrintCudaExecutionSummary(model, "  ");
    std::cout << "\n";
    ResetModelCudaRuntimeStats(model);
  }

  const BenchmarkResult result =
      RunBenchmark(model, tokens, n_predict, seed, verbose);

  std::cout << "Results:\n";
  std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
  std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
  std::cout << "  decode tokens:     " << result.n_decode_tokens << "\n";
  std::cout << "  prefill time:      " << std::fixed << std::setprecision(2)
            << result.prefill_ms << " ms\n";
  std::cout << "  decode time:       " << std::fixed << std::setprecision(2)
            << result.decode_ms << " ms\n";
  std::cout << "  total time:        " << std::fixed << std::setprecision(2)
            << (result.prefill_ms + result.decode_ms) << " ms\n";
  std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2)
            << result.tokens_per_sec() << "\n";
  std::cout << "  tokens/s (decode): " << std::fixed << std::setprecision(2)
            << result.decode_tokens_per_sec() << "\n";

  const size_t actual_bytes = ModelWeightBytes(model);
  const size_t f32_bytes = ModelWeightBytesF32(model);
  const double mib = 1024.0 * 1024.0;
  std::cout << "\n  weight memory:\n";
  std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << static_cast<double>(actual_bytes) / mib
            << " MB)\n";
  std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << static_cast<double>(f32_bytes) / mib
            << " MB)\n";
  if (actual_bytes > 0) {
    std::cout << "    savings:   " << std::fixed << std::setprecision(2)
              << static_cast<double>(f32_bytes) /
                     static_cast<double>(actual_bytes)
              << "x compression\n";
  }

  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "\n  cuda runtime:\n";
    PrintCudaRuntimeSummary(model, "    ");
  }

  if (!quant_type.empty()) {
    try {
      Tensor baseline_logits = RunLogitsForTokens(baseline_model, tokens);
      Tensor quant_logits = RunLogitsForTokens(model, tokens);
      auto [max_err, mean_err] = LogitsError(baseline_logits, quant_logits);

      std::cout << "\n  logits error vs model-native:\n";
      std::cout << "    max:  " << std::scientific << std::setprecision(3)
                << max_err << "\n";
      std::cout << "    mean: " << std::scientific << std::setprecision(3)
                << mean_err << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to compute logits error: " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}

// ── --cuda-info：列出设备 ────────────────────────────────────────────────
static int RunCudaInfo() {
  try {
    int count = CudaDeviceCount();
    std::cout << "CUDA devices: " << count << "\n";
    for (int device_id = 0; device_id < count; ++device_id) {
      CudaDeviceInfo info = CudaGetDeviceInfo(device_id);
      double total_gb = static_cast<double>(info.total_memory_bytes) /
                        (1024.0 * 1024.0 * 1024.0);
      std::cout << "device " << info.id << ": " << info.name << "\n"
                << "  device name: " << info.name << "\n"
                << "  compute capability: " << info.compute_major << "."
                << info.compute_minor << "\n"
                << "  total memory: " << info.total_memory_bytes << " bytes ("
                << std::fixed << std::setprecision(2) << total_gb << " GB)\n"
                << "  driver version: " << info.driver_version << "\n"
                << "  runtime version: " << info.runtime_version << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CUDA info failed: " << e.what() << "\n";
    return 1;
  }
}

// ── inspect：看 manifest / 模型配置 ─────────────────────────────────────
static void PrintInspectUsage(const char* prog) {
  std::cout << "Usage: " << prog << " inspect <model-path|dir> [options]\n"
            << "Options:\n"
            << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
               "requires -DDQM_LLAMA_CUDA=ON)\n"
            << "  --device <n>         CUDA device id for --backend cuda "
               "(default: 0)\n"
            << "  -h, --help           Show this help\n";
}

static MiniLlamaModel LoadModelForInspect(const std::string& path) {
  std::string model_path = path;
  if (std::filesystem::is_directory(path)) {
    std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
    std::filesystem::path json_path =
        std::filesystem::path(path) / "model.json";
    if (std::filesystem::exists(bin_path) &&
        std::filesystem::exists(json_path)) {
      return LoadModel(json_path.string(), bin_path.string());
    }
    std::string gguf_path = FindGgufInDirectory(path);
    if (!gguf_path.empty()) {
      model_path = gguf_path;
    } else {
      return LoadModel(json_path.string(), bin_path.string());
    }
  }

  if (IsGgufFile(model_path)) {
    return LoadGgufModel(model_path);
  }

  std::filesystem::path model_file(model_path);
  std::string config_path = (model_file.parent_path() / "model.json").string();
  return LoadModel(config_path, model_path);
}

static int RunInspect(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintInspectUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model path.\n";
    PrintInspectUsage(argv[0]);
    return 1;
  }

  std::string model_path = argv[2];
  BackendConfig backend_config;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintInspectUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintInspectUsage(argv[0]);
      return 1;
    }
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  if (std::filesystem::is_directory(model_path)) {
    std::filesystem::path config_path =
        std::filesystem::path(model_path) / "model.json";
    if (std::filesystem::exists(config_path) &&
        !InspectModel(config_path.string())) {
      return 1;
    }
  }

  MiniLlamaModel model = LoadModelForInspect(model_path);
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  std::cout << "backend: " << BackendKindName(backend_config.kind) << "\n";
  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "cuda: " << CudaDeviceSummary(backend_config.device_id)
              << "\n";
    if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
      return 1;
    }
    PrintCudaExecutionSummary(model);
  }

  return 0;
}

// ── inspect-gguf：打印 GGUF 元数据与张量列表 ────────────────────────────
static int RunInspectGguf(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
    return 1;
  }
  GgufReader reader;
  if (!reader.Load(argv[2])) {
    std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
    return 1;
  }
  InspectGguf(reader);
  return 0;
}

}  // namespace dqm

namespace {

void PrintTopLevelUsage(const char* program, std::ostream& out) {
  out << "Usage: " << program << " <command> [options]\n"
      << "Commands:\n"
      << "  run <model_dir>               Interactive chat\n"
      << "  generate [model_dir] -p <s>   One-shot generation (smoke)\n"
      << "  bench [model_dir]             Benchmark prefill + decode\n"
      << "  inspect <model-path|dir>      Show model config / tensors\n"
      << "  inspect-gguf <path>           Show GGUF metadata / tensors\n"
      << "  --cuda-info                   Show CUDA device information\n"
      << "\nRun '<command> --help' for command-specific options.\n";
}

}  // namespace

// 程序入口：把 argv[1] 分发给子命令。
int main(int argc, char** argv) {
  if (argc < 2) {
    PrintTopLevelUsage(argv[0], std::cerr);
    return 1;
  }

  const std::string command = argv[1];
  if (command == "-h" || command == "--help") {
    PrintTopLevelUsage(argv[0], std::cout);
    return 0;
  }
  if (command == "run") {
    return dqm::RunChat(argc, argv);
  }
  if (command == "generate") {
    return dqm::RunGenerate(argc, argv);
  }
  if (command == "bench") {
    return dqm::RunBench(argc, argv);
  }
  if (command == "inspect") {
    return dqm::RunInspect(argc, argv);
  }
  if (command == "inspect-gguf") {
    return dqm::RunInspectGguf(argc, argv);
  }
  if (command == "--cuda-info") {
    return dqm::RunCudaInfo();
  }
  if (!command.empty() && command[0] == '-') {
    // 兼容旧调用：第一个参数就是选项时，默认按 generate 处理。
    std::vector<char*> shifted;
    shifted.reserve(static_cast<size_t>(argc) + 1);
    shifted.push_back(argv[0]);
    shifted.push_back(const_cast<char*>("generate"));
    for (int i = 1; i < argc; ++i) {
      shifted.push_back(argv[i]);
    }
    shifted.push_back(nullptr);
    return dqm::RunGenerate(argc + 1, shifted.data());
  }

  std::cerr << "Unknown command: " << command << "\n";
  return 1;
}
