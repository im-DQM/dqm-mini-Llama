#include "dqm_llama/matmul_dispatch.h"

#include <stdexcept>

namespace dqm {

namespace {

// 前置：a、b 已确认为 2D 且内维匹配（校验在 MatmulDispatch 做）。
Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int k = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int t = 0; t < k; ++t) {
        sum += a.data[i * k + t] * b.data[t * n + j];
      }
      c.data[i * n + j] = sum;
    }
  }
  return c;
}


// 校验 x（1D 的 [k]，或 2D 的 [1, k]）与 weight 的形状 [out_features,
// in_features] 是否匹配。两个 LinearDispatch 重载共用。
void ValidateLinearShapes(const Tensor& x, const std::vector<int>& weight_shape) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error("matmul_dispatch: in LinearDispatch, weight_shape_size must be 2");
  }
  if (x.shape.size() == 1) {
    if (x.shape[0] != weight_shape[1]) {
      throw std::runtime_error("matmul_dispatch: in LinearDispatch, uncompared dim");
    }
    return;
  }
  if (x.shape.size() == 2 && x.shape[0] == 1) {
    if (x.shape[1] != weight_shape[1]) {
      throw std::runtime_error("matmul_dispatch: in LinearDispatch, uncompared dim");
    }
    return;
  }
  throw std::runtime_error("matmul_dispatch: in LinearDispatch, x_dim_size too big, can only compare 1 dim");
}

// 前置：形状已校验。w 指向 [out_features, in_features] 的连续 float，
// 调用方保证至少 out_features * in_features 个元素。
Tensor LinearNaive(const Tensor& x, const float* w, int out_features,
                   int in_features) {
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  for (int j = 0; j < out_features; ++j) {
    float sum = 0.0f;
    for (int t = 0; t < in_features; ++t) {
      sum += x.data[t] * w[j * in_features + t];
    }
    result.data[j] = sum;
  }
  return result;
}

}  // namespace

MatmulMode DefaultMatmulMode() {
  return MatmulMode::kNaive;
}

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  //校验
  //先做本地函数调用
  if (mode != MatmulMode::kNaive) {
    throw std::logic_error("other mode not implemente");
  }
  auto a_shape = a.shape;
  auto b_shape = b.shape;
  if (a_shape.size() != 2) {
    throw std::runtime_error("matmul_dispatch: in matmulDispatch, shape_size must be 2");
  }
  if (a_shape.size() != b_shape.size()) {
    throw std::runtime_error("matmul_dispatch: in matmulDispatch, a_shape_size not compare b_shape_size");
  }
  if (a_shape[1] != b_shape[0]) {
    throw std::runtime_error("matmul_dispatch: in matmulDispatch, a_shape not compare b_shape");
  }
  return MatmulNaive(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  if (mode != MatmulMode::kNaive) {
    throw std::logic_error("matmul_dispatch: in LinearDispatch, other mode not implement");
  }
  ValidateLinearShapes(x, weight.shape);
  return LinearNaive(x, weight.data.data(), weight.shape[0], weight.shape[1]);
}

Tensor LinearDispatch(const Tensor& x, const QuantizedTensor& weight,
                      MatmulMode mode) {
  if (mode != MatmulMode::kNaive) {
    throw std::logic_error("matmul_dispatch: in LinearDispatch, other mode not implement");
  }
  if (weight.type != QuantType::kF32) {
    throw std::runtime_error("matmul_dispatch: in LinearDispatch, quantized weight type not implemented");
  }
  ValidateLinearShapes(x, weight.shape);
  return LinearNaive(x, weight.f32_data.data(), weight.shape[0],
                     weight.shape[1]);
}

}  // namespace dqm
