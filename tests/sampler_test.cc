#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "dqm_llama/sampler.h"
#include "dqm_llama/tensor.h"

using dqm::MiniSampler;
using dqm::SamplingParams;
using dqm::Tensor;

namespace {

Tensor MakeLogits(const std::vector<float>& values) {
  Tensor t({static_cast<int>(values.size())}, 0.0f);
  for (size_t i = 0; i < values.size(); ++i) {
    t.data[i] = values[i];
  }
  return t;
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

}  // namespace

// ===========================================================================
// greedy
// ===========================================================================

TEST(Sampler, GreedyReturnsArgMaxIndex) {
  const Tensor logits = MakeLogits({-1.0f, 3.0f, 2.0f, 5.0f, 0.0f});
  EXPECT_EQ(MiniSampler::SampleGreedy(logits), 3);
}

TEST(Sampler, GreedyTieReturnsSmallestIndex) {
  EXPECT_EQ(MiniSampler::SampleGreedy(MakeLogits({2.0f, 2.0f, 2.0f})), 0);
  EXPECT_EQ(MiniSampler::SampleGreedy(MakeLogits({1.0f, 7.0f, 7.0f, 0.0f})), 1);
}

// ===========================================================================
// Sample 的分发
// ===========================================================================

TEST(Sampler, SampleDefaultsToGreedy) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({0.0f, 9.0f, 1.0f});
  SamplingParams params;  // 全 0 => 贪心
  EXPECT_EQ(s.Sample(logits, params), 1);
}

TEST(Sampler, TemperatureZeroIsGreedy) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({0.0f, 1.0f, 9.0f});
  SamplingParams params;
  params.temperature = 0.0f;
  params.top_k = 0;
  EXPECT_EQ(s.Sample(logits, params), 2);
}

TEST(Sampler, TopKOneIsGreedy) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({5.0f, 0.0f, 0.0f, 3.0f});
  SamplingParams params;
  params.temperature = 2.0f;  // 有温度也不影响：top_k == 1 一定贪心
  params.top_k = 1;
  EXPECT_EQ(s.Sample(logits, params), 0);
  EXPECT_EQ(s.SampleTopK(logits, 2.0f, 1), 0);
}

TEST(Sampler, SampleTopKZeroDisablesTopK) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({0.0f, 0.0f, 0.0f, 0.0f});
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 0;  // Sample 里 0 = 关闭，不抛
  const int token = s.Sample(logits, params);
  EXPECT_GE(token, 0);
  EXPECT_LT(token, 4);
}

TEST(Sampler, TinyPositiveTemperatureIsGreedy) {
  const Tensor logits = MakeLogits({0.0f, 0.0f, 4.0f});
  MiniSampler s(1u);
  EXPECT_EQ(s.SampleTemperature(logits, 1e-7f), 2);  // 低于 kGreedyTemperature
}

// ===========================================================================
// 校验
// ===========================================================================

TEST(Sampler, RejectsEmptyLogits) {
  const Tensor empty;
  MiniSampler s(1u);
  EXPECT_THROW(MiniSampler::SampleGreedy(empty), std::runtime_error);
  EXPECT_THROW(s.SampleTemperature(empty, 1.0f), std::runtime_error);
}

TEST(Sampler, RejectsNonFiniteLogits) {
  MiniSampler s(1u);
  EXPECT_THROW(MiniSampler::SampleGreedy(MakeLogits({1.0f, kNaN})),
               std::runtime_error);
  EXPECT_THROW(MiniSampler::SampleGreedy(MakeLogits({kInf, 0.0f})),
               std::runtime_error);
  EXPECT_THROW(MiniSampler::SampleGreedy(MakeLogits({-kInf, 0.0f})),
               std::runtime_error);
  EXPECT_THROW(s.SampleTemperature(MakeLogits({0.0f, kNaN}), 1.0f),
               std::runtime_error);
}

TEST(Sampler, RejectsBadTemperature) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({1.0f, 2.0f});
  EXPECT_THROW(s.SampleTemperature(logits, -1.0f), std::runtime_error);
  EXPECT_THROW(s.SampleTemperature(logits, kNaN), std::runtime_error);
  EXPECT_THROW(s.SampleTemperature(logits, kInf), std::runtime_error);

  SamplingParams params;
  params.temperature = -0.5f;
  EXPECT_THROW(s.Sample(logits, params), std::runtime_error);
}

TEST(Sampler, RejectsNegativeTopK) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({1.0f, 2.0f, 3.0f});
  EXPECT_THROW(s.SampleTopK(logits, 1.0f, -1), std::runtime_error);

  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = -3;
  EXPECT_THROW(s.Sample(logits, params), std::runtime_error);
}

TEST(Sampler, SampleTopKZeroThrows) {
  MiniSampler s(1u);
  const Tensor logits = MakeLogits({1.0f, 2.0f, 3.0f});
  EXPECT_THROW(s.SampleTopK(logits, 1.0f, 0), std::runtime_error);
}

// ===========================================================================
// 可复现性 / RNG 状态
// ===========================================================================

TEST(Sampler, SameSeedProducesSameSequence) {
  const Tensor logits = MakeLogits({0.5f, 1.5f, -1.0f, 2.0f, 0.0f});
  MiniSampler a(42u);
  MiniSampler b(42u);
  SamplingParams params;
  params.temperature = 1.0f;
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(a.Sample(logits, params), b.Sample(logits, params));
  }
}

TEST(Sampler, DifferentSeedsDiverge) {
  const Tensor logits = MakeLogits({0.0f, 0.0f, 0.0f, 0.0f});
  MiniSampler a(1u);
  MiniSampler b(2u);
  SamplingParams params;
  params.temperature = 1.0f;
  bool diverged = false;
  for (int i = 0; i < 64 && !diverged; ++i) {
    if (a.Sample(logits, params) != b.Sample(logits, params)) {
      diverged = true;
    }
  }
  EXPECT_TRUE(diverged);
}

TEST(Sampler, GreedyDoesNotAdvanceRng) {
  const Tensor logits = MakeLogits({0.2f, 0.9f, -0.4f});
  SamplingParams greedy;  // 全 0
  SamplingParams hot;
  hot.temperature = 1.0f;

  MiniSampler a(7u);
  for (int i = 0; i < 5; ++i) {
    a.Sample(logits, greedy);  // 贪心路径不该推进 rng_
  }
  const int from_a = a.Sample(logits, hot);

  MiniSampler b(7u);
  const int from_b = b.Sample(logits, hot);

  EXPECT_EQ(from_a, from_b);
}

TEST(Sampler, ParamsConstructorUsesOnlySeed) {
  const Tensor logits = MakeLogits({0.0f, 1.0f, 0.0f, 2.0f});
  SamplingParams params;
  params.temperature = 0.7f;  // 不参与播种
  params.top_k = 2;           // 不存进对象
  params.seed = 99u;
  MiniSampler from_params(params);
  MiniSampler from_seed(99u);

  SamplingParams sample_params;
  sample_params.temperature = 1.0f;
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(from_params.Sample(logits, sample_params),
              from_seed.Sample(logits, sample_params));
  }
}

TEST(Sampler, SeedZeroIsUsable) {
  const Tensor logits = MakeLogits({0.1f, 0.2f, 0.3f});
  MiniSampler s(0u);  // 走 random_device，不可复现，但必须能用
  EXPECT_GE(s.SampleTemperature(logits, 1.0f), 0);
}

// ===========================================================================
// top-k
// ===========================================================================

TEST(Sampler, TopKRestrictsToTopKValues) {
  const Tensor logits = MakeLogits({0.0f, 5.0f, 1.0f, 4.0f, 2.0f, 3.0f});
  // 值降序：5(idx1) > 4(idx3) > 3(idx5) => top-3 集合 = {1, 3, 5}
  MiniSampler s(3u);
  for (int i = 0; i < 100; ++i) {
    const int token = s.SampleTopK(logits, 1.0f, 3);
    EXPECT_TRUE(token == 1 || token == 3 || token == 5) << "token=" << token;
  }
}

TEST(Sampler, TopKEqualLogitsTieBreakKeepsSmallestIndices) {
  const Tensor logits = MakeLogits({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  MiniSampler s(5u);
  for (int i = 0; i < 100; ++i) {
    const int token = s.SampleTopK(logits, 1.0f, 2);
    // 全相等时按 token id 升序稳定截断 => 候选恒为 {0, 1}
    EXPECT_TRUE(token == 0 || token == 1) << "token=" << token;
  }
}

TEST(Sampler, TopKLargerThanVocabIsClamped) {
  const Tensor logits = MakeLogits({1.0f, 2.0f, 3.0f, 4.0f});
  MiniSampler s(9u);
  const int token = s.SampleTopK(logits, 1.0f, 100);
  EXPECT_GE(token, 0);
  EXPECT_LT(token, 4);
}

// ===========================================================================
// temperature
// ===========================================================================

TEST(Sampler, TemperatureSampleReturnsValidId) {
  const Tensor logits = MakeLogits({0.3f, -0.2f, 0.8f});
  MiniSampler s(11u);
  for (int i = 0; i < 50; ++i) {
    const int token = s.SampleTemperature(logits, 1.0f);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 3);
  }
}

TEST(Sampler, TemperatureFavorsHighLogit) {
  const Tensor logits = MakeLogits({0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  MiniSampler s(13u);
  const int draws = 200;
  int hits = 0;
  for (int i = 0; i < draws; ++i) {
    if (s.SampleTemperature(logits, 1.0f) == 3) {
      ++hits;
    }
  }
  // exp(5) 远大于其余四个 exp(0)：理论上 ~97%
  EXPECT_GT(hits, draws * 3 / 4);
}
