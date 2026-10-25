#include "dqm_llama/sampler.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dqm_llama/ops.h"

namespace dqm {

namespace { 

// 判定「温度低到等同贪心」的阈值。
// 出现在三处判断：SampleFromCandidates / SampleTemperature / Sample。
constexpr float kGreedyTemperature = 1e-6f;

// 校验 logits：非空、且每个元素都有限。
void ValidateLogits(const Tensor& logits) {
  if (logits.size() == 0) {
    throw std::runtime_error("sampler: cannot sample from empty logits");
  }
  
  for (auto num : logits.data) {
    if (!std::isfinite(num)) {
      throw std::runtime_error("sampler: logits must be finite");
    }
  }
}

// 校验温度：有限且非负。注意 0 合法（= 贪心）。
void ValidateTemperature(float temperature) {
  if (temperature < 0.0f || !std::isfinite(temperature)) {
    throw std::runtime_error(
        "sampler: temperature must be finite and non-negative");
  }
}

// 校验 top_k：非负。注意 0 合法（= 关闭）。
void ValidateTopK(int top_k) {
  if (top_k < 0) {
    throw std::runtime_error("sampler: top_k must be non-negative");
  }
}
// 核心采样 —— 唯一真正掷骰子的地方。candidates 可能已被 top-k 截短。
// rng 是引用：要推进它的内部状态；传值就白抽了。
int SampleFromCandidates(const std::vector<std::pair<float, int>>& candidates,
                         float temperature, std::mt19937& rng) {
  // TODO(dqm):
  //   1) temperature < kGreedyTemperature -> 线性扫找 first 最大者返回（平手取先出现）
  //   2) 求全部候选的 max（数值稳定）
  //   3) probs[i] = exp((candidates[i].first - max) / temperature)，累加 sum
  //   4) uniform_real_distribution<float>(0,1) 抽 r
  //   5) 累积 probs[i]/sum，第一个使 r <= cumulative 的返回；走完兜底返回 back()

  //如果temperature < kGreedyTemperature，走greedy
  if (temperature < kGreedyTemperature) {
    int best = candidates[0].second;
    float max_val = candidates[0].first;
    for (const auto& pair : candidates) {
      if (pair.first > max_val) {
        best = pair.second;
        max_val = pair.first;
      }
    }
    return best;
  }

  //先减去最大值再/temperature再取指数
  float max_val = candidates[0].first;
  for (const auto& pair : candidates) {
    if (pair.first > max_val) {
      max_val = pair.first;
    }
  }

  std::vector<float> prob(candidates.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < candidates.size(); i ++) {
    prob[i] = std::exp((candidates[i].first - max_val) / temperature);
    sum += prob[i];
  }

  //轮盘取法
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);
  float cumulative = 0.0f;
  for (size_t i = 0; i < candidates.size(); i ++) {
    cumulative += prob[i] / sum;
    if (r <= cumulative) {
      return candidates[i].second;
    }
  }

  return candidates.back().second;
}

}  // namespace

MiniSampler::MiniSampler(unsigned int seed) {
  if (seed != 0) {
    rng_.seed(seed);
  } else {
    std::random_device rd;
    rng_.seed(rd());
  }
}

MiniSampler::MiniSampler(const SamplingParams& params) : MiniSampler(params.seed) {}

int MiniSampler::SampleGreedy(const Tensor& logits) {
  ValidateLogits(logits);
  return ArgMax(logits);
}

int MiniSampler::SampleTemperature(const Tensor& logits, float temperature) {
  ValidateLogits(logits);
  ValidateTemperature(temperature);
  if (temperature < kGreedyTemperature) {
    return SampleGreedy(logits);
  }

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.data.size());
  for (size_t i = 0; i < logits.data.size(); i ++){
    candidates.emplace_back(std::make_pair(logits.data[i], static_cast<int>(i)));
  }

  return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::SampleTopK(const Tensor& logits, float temperature,
                            int top_k) {
  ValidateLogits(logits);
  ValidateTemperature(temperature);
  ValidateTopK(top_k);

  if (top_k == 0) {
    throw std::runtime_error(
        "sampler: top_k must be positive when calling SampleTopK");
  }
  const int vocab = static_cast<int>(logits.data.size());
  if (top_k > vocab) {
    top_k = vocab;
  }

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.data.size());
  for (size_t i = 0; i < logits.data.size(); i ++){
    candidates.emplace_back(std::make_pair(logits.data[i], static_cast<int>(i)));
  }
  std::sort(candidates.begin(), candidates.end(), 
    [](const auto& a, const auto& b){
      if (a.first == b.first) {
        return a.second < b.second;
      }
      return a.first > b.first;
  });

  candidates.resize(top_k);
  return SampleFromCandidates(candidates, temperature, rng_);
}

int MiniSampler::Sample(const Tensor& logits, const SamplingParams& params) {
  //每个分支都会检验logits,而且这个检验特别贵
  ValidateTemperature(params.temperature);
  ValidateTopK(params.top_k);
  if (params.temperature < kGreedyTemperature || params.top_k == 1) {
    return SampleGreedy(logits);
  }
  if (params.top_k > 1) {
    return SampleTopK(logits, params.temperature, params.top_k);
  }
  return SampleTemperature(logits, params.temperature);
}

}  // namespace dqm
