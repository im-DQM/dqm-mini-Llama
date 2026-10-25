#pragma once

#include <random>

#include "dqm_llama/tensor.h"

namespace dqm {

// 采样参数。三个成员都约定「0 = 特殊含义」，具体含义由构造函数与各方法的判断解释。
struct SamplingParams {
  float temperature = 0.0f;  // 0 表示贪心
  int top_k = 0;             // 0 表示关闭 top-k
  unsigned int seed = 0;     // 0 表示改用 random_device 取种子
};

// 采样器：把 logits 变成一个 token id。支持 greedy / temperature / top-k 三档。
// 只持有 rng_ 一个成员 —— 采样参数每次由调用方现传，不存进对象。
class MiniSampler {
 public:
  explicit MiniSampler(unsigned int seed = 0);
  explicit MiniSampler(const SamplingParams& params);

  // 主入口：按 params 分发到三档之一。
  int Sample(const Tensor& logits, const SamplingParams& params);

  // 贪心：恒返回 ArgMax。静态 —— 不掷骰子，不需要采样器对象。
  static int SampleGreedy(const Tensor& logits);

  // 温度采样（无 top-k）。
  int SampleTemperature(const Tensor& logits, float temperature);

  // top-k 采样（温度作用在截断后的 k 个候选内部）。
  int SampleTopK(const Tensor& logits, float temperature, int top_k);

 private:
  std::mt19937 rng_;
};

}  // namespace dqm
