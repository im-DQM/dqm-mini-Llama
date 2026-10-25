#pragma once

#include <string>
#include <vector>

#include "dqm_llama/kv_cache.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"

namespace dqm {

// 一次 benchmark 的计时统计。
struct BenchmarkResult {
  int n_prompt_tokens = 0;
  int n_generated_tokens = 0;
  int n_decode_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;

  // 总吞吐：prefill + decode 一起算，分母是两段耗时之和。
  double tokens_per_sec() const {
    const double total_ms = prefill_ms + decode_ms;
    if (total_ms <= 0.0) {
      return 0.0;
    }
    return static_cast<double>(n_generated_tokens) * 1000.0 / total_ms;
  }

  // 纯 decode 吞吐：只算 decode 段，比 tokens_per_sec 更能反映逐 token 的代价。
  double decode_tokens_per_sec() const {
    if (decode_ms <= 0.0 || n_decode_tokens <= 0) {
      return 0.0;
    }
    return static_cast<double>(n_decode_tokens) * 1000.0 / decode_ms;
  }
};

// 调试转储：直接把信息打到 stdout。
void DumpTensorShape(const Tensor& t, const std::string& name);
void DumpLogitsTopK(const Tensor& logits, int k);
void DumpKvCacheInfo(const KvCache& cache, int current_pos = -1);

// 跑一遍 prefill + decode，返回计时统计。
// verbose 为 true 时，每步后打调试转储。
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose);

}  // namespace dqm
