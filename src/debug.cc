#include "dqm_llama/debug.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "dqm_llama/context.h"
#include "dqm_llama/forward.h"
#include "dqm_llama/sampler.h"

namespace dqm {

// ---------------------------------------------------------------------------
// DumpTensorShape
// ---------------------------------------------------------------------------
void DumpTensorShape(const Tensor& t, const std::string& name) {
  std::cout << "  tensor: " << name << " shape=[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << t.shape[i];
  }
  std::cout << "] dtype=f32 size=" << t.size() << "\n";
}

// ---------------------------------------------------------------------------
// DumpLogitsTopK
// ---------------------------------------------------------------------------
void DumpLogitsTopK(const Tensor& logits, int k) {
  if (logits.num_dims() != 1) {
    std::cout << "  logits: invalid shape for top-k dump\n";
    return;
  }
  const int vocab_size = logits.shape[0];
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  // 收集 (value, id) 对，再按 value 降序部分排序。
  std::vector<std::pair<float, int>> pairs;
  pairs.reserve(static_cast<size_t>(vocab_size));
  for (int i = 0; i < vocab_size; ++i) {
    pairs.push_back({logits.data[static_cast<size_t>(i)], i});
  }

  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  std::cout << "  logits top-" << k << ":\n";
  for (int i = 0; i < k; ++i) {
    const std::pair<float, int>& entry = pairs[static_cast<size_t>(i)];
    std::cout << "    [" << std::setw(3) << i << "] id=" << std::setw(4)
              << entry.second << " value=" << std::fixed << std::setprecision(6)
              << entry.first << "\n";
  }
}

// ---------------------------------------------------------------------------
// DumpKvCacheInfo
// ---------------------------------------------------------------------------
void DumpKvCacheInfo(const KvCache& cache, int current_pos) {
  const int n_layers = cache.keys.shape[0];
  const int seq_len = cache.keys.shape[1];
  const int n_kv_heads = cache.keys.shape[2];
  const int head_dim = cache.keys.shape[3];
  std::cout << "  KV cache: layers=" << n_layers << " seq_len=" << seq_len
            << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << "\n";
  if (current_pos >= 0) {
    std::cout << "    current_pos: " << current_pos << "\n";
  }
  std::cout << "    keys shape: [" << cache.keys.shape[0] << ", "
            << cache.keys.shape[1] << ", " << cache.keys.shape[2] << ", "
            << cache.keys.shape[3] << "]\n";
  std::cout << "    values shape: [" << cache.values.shape[0] << ", "
            << cache.values.shape[1] << ", " << cache.values.shape[2] << ", "
            << cache.values.shape[3] << "]\n";
}

// ---------------------------------------------------------------------------
// RunBenchmark
// ---------------------------------------------------------------------------
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict, unsigned int seed, bool verbose) {
  BenchmarkResult result;
  result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

  MiniLlamaContext ctx(&model);
  MiniSampler sampler(seed);
  SamplingParams params;
  params.seed = seed;

  if (verbose) {
    std::cout << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
  }

  // Prefill：整段 prompt 一次前向。
  const auto prefill_start = std::chrono::steady_clock::now();
  const MiniBatch prefill = MiniBatch::FromTokens(prompt_tokens, 0);
  Tensor logits = ForwardBatch(ctx, model, prefill);
  const auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();

  if (verbose) {
    DumpTensorShape(logits, "logits");
    DumpLogitsTopK(logits, 5);
    DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
  }

  // Decode：prompt 之后逐 token 前向。第一个 token 来自 prefill 的 logits，
  // 所以它计入 n_generated_tokens 但不计入 n_decode_tokens（没走 decode 循环）。
  if (n_predict > 0 && result.n_prompt_tokens < model.config.max_seq_len) {
    std::vector<int> tokens = prompt_tokens;
    int next_token = sampler.Sample(logits, params);
    tokens.push_back(next_token);
    ++result.n_generated_tokens;

    if (verbose) {
      std::cout << "[verbose] Decode step 0: token=" << next_token << "\n";
    }

    const auto decode_start = std::chrono::steady_clock::now();
    for (int i = 1;
         i < n_predict &&
         static_cast<int>(tokens.size()) < model.config.max_seq_len;
         ++i) {
      const MiniBatch decode =
          MiniBatch::Single(tokens.back(), static_cast<int>(tokens.size()) - 1);
      logits = ForwardBatch(ctx, model, decode);
      ++result.n_decode_tokens;
      next_token = sampler.Sample(logits, params);
      tokens.push_back(next_token);
      ++result.n_generated_tokens;

      if (verbose) {
        std::cout << "[verbose] Decode step " << i << ": token=" << next_token
                  << "\n";
        DumpLogitsTopK(logits, 5);
      }
    }

    const auto decode_end = std::chrono::steady_clock::now();
    if (result.n_decode_tokens > 0) {
      result.decode_ms =
          std::chrono::duration<double, std::milli>(decode_end - decode_start)
              .count();
    }
  }

  if (verbose) {
    DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
  }

  return result;
}

}  // namespace dqm
