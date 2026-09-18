#include "dqm_llama/ops.h"

#include <cmath>
#include <stdexcept>

#include "dqm_llama/matmul_dispatch.h"

namespace dqm {

namespace {

// 原地按"相邻配对"旋转一个 [n_heads, head_dim] 张量。
void ApplyRopeNormalToHeads(Tensor& x, int pos, float theta) {
  int n_heads = x.shape[0];
  int head_dim = x.shape[1];
  for (int h = 0; h < n_heads; ++h) {
    for (int i = 0; i < head_dim; i += 2) {
      float freq = 1.0f / std::pow(
                               theta, static_cast<float>(i) /
                                          static_cast<float>(head_dim));
      const float angle = static_cast<float>(pos) * freq;
      float c = std::cos(angle);
      float s = std::sin(angle);
      float x0 = x.data[h * head_dim + i];
      float x1 = x.data[h * head_dim + i + 1];
      x.data[h * head_dim + i] = x0 * c - x1 * s;
      x.data[h * head_dim + i + 1] = x0 * s + x1 * c;
    }
  }
}

// 原地按"前后半配对"旋转一个 [n_heads, head_dim] 张量。
void ApplyRopeNeoXToHeads(Tensor& x, int pos, float theta) {
  int n_heads = x.shape[0];
  int head_dim = x.shape[1];
  int half = head_dim / 2;
  for (int h = 0; h < n_heads; ++h) {
    int base = h * head_dim;
    for (int i = 0; i < half; ++i) {
      float freq = 1.0f / std::pow(
                               theta, static_cast<float>(2 * i) /
                                          static_cast<float>(head_dim));
      const float angle = static_cast<float>(pos) * freq;
      float c = std::cos(angle);
      float s = std::sin(angle);
      float x0 = x.data[base + i];
      float x1 = x.data[base + half + i];
      x.data[base + i] = x0 * c - x1 * s;
      x.data[base + half + i] = x0 * s + x1 * c;
    }
  }
}

}  // namespace

Tensor Matmul(const Tensor& a, const Tensor& b) {
  return MatmulDispatch(a, b, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
  return LinearDispatch(x, weight, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const QuantizedTensor& weight) {
  return LinearDispatch(x, weight, DefaultMatmulMode());
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
  if (x.num_dims() != 1 || weight.num_dims() != 1) {
    throw std::runtime_error("ops: RmsNorm: dim must be 1");
  }
  const int dim = static_cast<int>(x.size());
  if (weight.size() != static_cast<size_t>(dim)) {
    throw std::runtime_error("ops: RmsNorm: dim not compare");
  }

  Tensor res({dim}, 0.0f);
  float mean = 0;
  for (int i = 0; i < dim; i ++) {
    mean += x.data[i] * x.data[i];
  }
  mean /= static_cast<float>(dim);
  float rms = std::sqrt(mean + eps);
  //浮点数乘法快于除法
  float scale = 1 / rms;
  for (int i = 0; i < dim; i ++) {
    res.data[i] = (x.data[i]) * scale * weight[i];
  }
  return res;
}

Tensor Softmax(const Tensor& x) {
  if (x.num_dims() != 1) {
    throw std::runtime_error("Softmax: expected 1D tensor");
  }
  if (x.data.size() == 0) {
    throw std::runtime_error("Softmax: empty tensor");
  }
  int n = x.shape[0];

  float max_val = x.data[0];
  for (int i = 1; i < n; ++i) {
    if (x.data[i] > max_val) {
      max_val = x.data[i];
    }
  }

  Tensor y({n}, 0.0f);
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    y.data[i] = std::exp(x.data[i] - max_val);
    sum += y.data[i];
  }
  for (int i = 0; i < n; ++i) {
    y.data[i] /= sum;
  }
  return y;
}

Tensor Silu(const Tensor& x) {
  Tensor y(x.shape, 0.0f);
  for (size_t i = 0; i < x.data.size(); ++i) {
    float val = x.data[i];
    y.data[i] = val / (1.0f + std::exp(-val));
  }
  return y;
}

Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("ElementwiseMul: shape mismatch");
  }
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.data.size(); ++i) {
    y.data[i] = a.data[i] * b.data[i];
  }
  return y;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
  if (gate.shape != up.shape) {
    throw std::runtime_error("SwiGlu: shape mismatch");
  }
  Tensor y(gate.shape, 0.0f);
  for (size_t i = 0; i < gate.data.size(); ++i) {
    float val = gate.data[i];
    float sigmoid = 1.0f / (1.0f + std::exp(-val));
    y.data[i] = val * sigmoid * up.data[i];
  }
  return y;
}

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
  if (q.num_dims() != 2 || k.num_dims() != 2) {
    throw std::runtime_error("Rope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("Rope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("Rope: theta must be finite and positive");
  }

  int head_dim = q.shape[1];
  int k_head_dim = k.shape[1];
  if (head_dim != k_head_dim) {
    throw std::runtime_error("Rope: q and k head_dim mismatch");
  }
  if (head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("Rope: head_dim must be positive and even");
  }

  if (rope_type == RopeType::kNeoX) {
    ApplyRopeNeoXToHeads(q, pos, theta);
    ApplyRopeNeoXToHeads(k, pos, theta);
  } else {
    ApplyRopeNormalToHeads(q, pos, theta);
    ApplyRopeNormalToHeads(k, pos, theta);
  }
}

int ArgMax(const Tensor& x) {
  if (x.size() == 0) {
    throw std::runtime_error("ArgMax: empty tensor");
  }
  int best = 0;
  float best_val = x.data[0];
  for (size_t i = 1; i < x.data.size(); ++i) {
    if (x.data[i] > best_val) {
      best_val = x.data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

}  // namespace dqm
