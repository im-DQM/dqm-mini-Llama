#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "dqm_llama/debug.h"
#include "dqm_llama/kv_cache.h"
#include "dqm_llama/loader.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"
#include "dqm_llama/tokenizer.h"

using dqm::AsciiTokenizer;
using dqm::BenchmarkResult;
using dqm::DumpKvCacheInfo;
using dqm::DumpLogitsTopK;
using dqm::DumpTensorShape;
using dqm::KvCache;
using dqm::LoadModel;
using dqm::MiniLlamaModel;
using dqm::RunBenchmark;
using dqm::Tensor;

#ifndef DQM_TEST_DATA_DIR
#define DQM_TEST_DATA_DIR "."
#endif

namespace {

std::string DataPath(const std::string& rel) {
  return std::string(DQM_TEST_DATA_DIR) + "/" + rel;
}

std::string TinyModelJson() {
  return DataPath("models/tiny/model.json");
}

std::string TinyModelBin() {
  return DataPath("models/tiny/model.bin");
}

}  // namespace

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------
TEST(Debug, BenchmarkResultZero) {
  const BenchmarkResult r;
  EXPECT_EQ(r.n_prompt_tokens, 0);
  EXPECT_EQ(r.n_generated_tokens, 0);
  EXPECT_EQ(r.n_decode_tokens, 0);
  EXPECT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
  EXPECT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
}

TEST(Debug, BenchmarkResultTokensPerSec) {
  BenchmarkResult r;
  r.n_prompt_tokens = 10;
  r.n_generated_tokens = 100;
  r.n_decode_tokens = 100;
  r.prefill_ms = 50.0;
  r.decode_ms = 450.0;
  EXPECT_NEAR(r.tokens_per_sec(), 200.0, 1e-6);  // 100 / 0.5
  EXPECT_NEAR(r.decode_tokens_per_sec(), 1000.0 / 4.5, 1e-6);
}

TEST(Debug, BenchmarkResultNoDecode) {
  BenchmarkResult r;
  r.n_generated_tokens = 0;
  r.n_decode_tokens = 0;
  r.prefill_ms = 100.0;
  r.decode_ms = 0.0;
  EXPECT_NEAR(r.tokens_per_sec(), 0.0, 1e-6);
  EXPECT_NEAR(r.decode_tokens_per_sec(), 0.0, 1e-6);
}

// ---------------------------------------------------------------------------
// DumpTensorShape（冒烟：只验证不崩）
// ---------------------------------------------------------------------------
TEST(Debug, DumpTensorShapeSmoke) {
  const Tensor t({2, 3, 4}, 1.0f);
  DumpTensorShape(t, "TestTensor");
}

// ---------------------------------------------------------------------------
// DumpLogitsTopK（冒烟）
// ---------------------------------------------------------------------------
TEST(Debug, DumpLogitsTopKSmoke) {
  Tensor logits({10}, 0.0f);
  for (int i = 0; i < 10; ++i) {
    logits.data[i] = static_cast<float>(i);
  }
  DumpLogitsTopK(logits, 3);
}

TEST(Debug, DumpLogitsTopKLargerThanVocab) {
  Tensor logits({5}, 0.0f);
  for (int i = 0; i < 5; ++i) {
    logits.data[i] = static_cast<float>(i);
  }
  DumpLogitsTopK(logits, 10);  // k > vocab_size，应被夹到 vocab_size
}

// ---------------------------------------------------------------------------
// DumpKvCacheInfo（冒烟）
// ---------------------------------------------------------------------------
TEST(Debug, DumpKvCacheInfoSmoke) {
  const KvCache cache(2, 16, 4, 8);
  DumpKvCacheInfo(cache);
}

// ---------------------------------------------------------------------------
// RunBenchmark（tiny 模型集成冒烟）
// ---------------------------------------------------------------------------
TEST(Debug, RunBenchmarkSmoke) {
  const MiniLlamaModel model = LoadModel(TinyModelJson(), TinyModelBin());
  ASSERT_TRUE(model.loaded) << "failed to load tiny model";

  const AsciiTokenizer tokenizer;
  const std::vector<int> tokens = tokenizer.Encode("hi");
  const BenchmarkResult result = RunBenchmark(model, tokens, 4, 42, false);

  EXPECT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
  EXPECT_EQ(result.n_generated_tokens, 4);
  EXPECT_EQ(result.n_decode_tokens, 3);  // 首个 token 来自 prefill，不算 decode
  EXPECT_GE(result.prefill_ms, 0.0);
  EXPECT_GE(result.decode_ms, 0.0);
  EXPECT_GE(result.tokens_per_sec(), 0.0);
}

TEST(Debug, RunBenchmarkZeroPredict) {
  const MiniLlamaModel model = LoadModel(TinyModelJson(), TinyModelBin());
  ASSERT_TRUE(model.loaded) << "failed to load tiny model";

  const AsciiTokenizer tokenizer;
  const std::vector<int> tokens = tokenizer.Encode("hi");
  const BenchmarkResult result = RunBenchmark(model, tokens, 0, 42, false);

  EXPECT_EQ(result.n_prompt_tokens, static_cast<int>(tokens.size()));
  EXPECT_EQ(result.n_generated_tokens, 0);
  EXPECT_EQ(result.n_decode_tokens, 0);
  EXPECT_GE(result.prefill_ms, 0.0);
  EXPECT_DOUBLE_EQ(result.decode_ms, 0.0);
}
