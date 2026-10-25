#include "dqm_llama/matmul_dispatch.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/quant.h"
#include "dqm_llama/thread_pool.h"

// 平台 SIMD 头：只有对应的编译开关打开时才引入（见 CMakeLists 的 -mavx2 -mfma）。
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define DQM_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define DQM_USE_AVX2 1
#endif

namespace dqm {

namespace {

// ---------------------------------------------------------------------------
// Naive 实现（保留：可读基准 + SIMD 不可用时的兜底）
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// 多线程实现（并行单位 = 输出行）
// ---------------------------------------------------------------------------

Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int k = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  ParallelFor(m, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int t = 0; t < k; ++t) {
          sum += a.data[i * k + t] * b.data[t * n + j];
        }
        c.data[i * n + j] = sum;
      }
    }
  });
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

Tensor LinearThreaded(const Tensor& x, const float* w, int out_features,
                      int in_features) {
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  // 并行单位 = 输出神经元：每个 j 只写 result[j]，互不重叠。
  ParallelFor(out_features, [&](int begin, int end) {
    for (int j = begin; j < end; ++j) {
      float sum = 0.0f;
      for (int t = 0; t < in_features; ++t) {
        sum += x.data[t] * w[j * in_features + t];
      }
      result.data[j] = sum;
    }
  });
  return result;
}

// ---------------------------------------------------------------------------
// SIMD 点积（平台相关）。要求 a、b 各含至少 n 个连续 float。
// ---------------------------------------------------------------------------

#ifdef DQM_USE_NEON
float DotSimdNeon(const float* a, const float* b, int n) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum = vfmaq_f32(sum, va, vb);
  }
  float32x2_t r = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
  r = vpadd_f32(r, r);
  float result = vget_lane_f32(r, 0);
  for (; i < n; ++i) {  // 尾部退回标量
    result += a[i] * b[i];
  }
  return result;
}
#endif

#ifdef DQM_USE_AVX2
float DotSimdAvx2(const float* a, const float* b, int n) {
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  // 水平归约：256 -> 128 -> 标量
  __m128 sum_lo = _mm256_castps256_ps128(sum);
  __m128 sum_hi = _mm256_extractf128_ps(sum, 1);
  sum_lo = _mm_add_ps(sum_lo, sum_hi);
  sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
  sum_lo = _mm_hadd_ps(sum_lo, sum_lo);
  float result = _mm_cvtss_f32(sum_lo);
  for (; i < n; ++i) {  // 尾部退回标量
    result += a[i] * b[i];
  }
  return result;
}
#endif

// 无 SIMD 开关时退化为标量点积（仍正确，只是没加速）。
float DotSimd(const float* a, const float* b, int n) {
#if defined(DQM_USE_AVX2)
  return DotSimdAvx2(a, b, n);
#elif defined(DQM_USE_NEON)
  return DotSimdNeon(a, b, n);
#else
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
#endif
}

// ---------------------------------------------------------------------------
// SIMD 实现（Linear 专用：权重按行连续，两个操作数都能向量化）
// ---------------------------------------------------------------------------

Tensor LinearSimd(const Tensor& x, const float* w, int out_features,
                  int in_features) {
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  for (int j = 0; j < out_features; ++j) {
    result.data[j] =
        DotSimd(x.data.data(), w + j * in_features, in_features);
  }
  return result;
}

Tensor LinearThreadedSimd(const Tensor& x, const float* w, int out_features,
                          int in_features) {
  bool is_1d = (x.num_dims() == 1);

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{1, out_features},
                0.0f);
  ParallelFor(out_features, [&](int begin, int end) {
    for (int j = begin; j < end; ++j) {
      result.data[j] =
          DotSimd(x.data.data(), w + j * in_features, in_features);
    }
  });
  return result;
}

// F32 权重的四档分发（两个 LinearDispatch 重载共用内核）。
Tensor LinearDispatchF32(const Tensor& x, const float* w, int out_features,
                         int in_features, MatmulMode mode) {
  switch (mode) {
    case MatmulMode::kNaive:
      return LinearNaive(x, w, out_features, in_features);
    case MatmulMode::kThreaded:
      return LinearThreaded(x, w, out_features, in_features);
    case MatmulMode::kSimd:
      return LinearSimd(x, w, out_features, in_features);
    case MatmulMode::kThreadedSimd:
      return LinearThreadedSimd(x, w, out_features, in_features);
  }
  return LinearThreadedSimd(x, w, out_features, in_features);
}

}  // namespace

namespace {

// 主推理路径的默认策略；SetDefaultMatmulMode() 可覆盖。
MatmulMode g_default_matmul_mode = MatmulMode::kThreadedSimd;

}  // namespace

MatmulMode DefaultMatmulMode() { return g_default_matmul_mode; }

void SetDefaultMatmulMode(MatmulMode mode) { g_default_matmul_mode = mode; }

bool ParseMatmulMode(const std::string& text, MatmulMode& mode) {
  if (text == "naive") {
    mode = MatmulMode::kNaive;
    return true;
  }
  if (text == "threaded") {
    mode = MatmulMode::kThreaded;
    return true;
  }
  if (text == "simd") {
    mode = MatmulMode::kSimd;
    return true;
  }
  if (text == "threaded-simd") {
    mode = MatmulMode::kThreadedSimd;
    return true;
  }
  return false;
}

std::string MatmulModeName(MatmulMode mode) {
  switch (mode) {
    case MatmulMode::kNaive:
      return "naive";
    case MatmulMode::kThreaded:
      return "threaded";
    case MatmulMode::kSimd:
      return "simd";
    case MatmulMode::kThreadedSimd:
      return "threaded-simd";
  }
  return "unknown";
}

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
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
  switch (mode) {
    case MatmulMode::kNaive:
      return MatmulNaive(a, b);
    case MatmulMode::kThreaded:
    case MatmulMode::kSimd:
    case MatmulMode::kThreadedSimd:
      // 通用 matmul 里 b 是「按列」访问（b[t*n + j]，随 t 跳 n 步）不连续，
      // SIMD 难以套用 —— kSimd / kThreadedSimd 统一退回多线程（与参考一致）。
      return MatmulThreaded(a, b);
  }
  return MatmulThreaded(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  ValidateLinearShapes(x, weight.shape);
  return LinearDispatchF32(x, weight.data.data(), weight.shape[0],
                           weight.shape[1], mode);
}

Tensor LinearDispatch(const Tensor& x, const QuantizedTensor& weight,
                      MatmulMode mode) {
  switch (weight.type) {
    case QuantType::kF32:
      ValidateLinearShapes(x, weight.shape);
      return LinearDispatchF32(x, weight.f32_data.data(), weight.shape[0],
                               weight.shape[1], mode);
    case QuantType::kQ80:
      return LinearQ80(x, weight.q8_0_data, weight.shape);
    case QuantType::kQ40:
      return LinearQ40(x, weight.q4_0_data, weight.shape);
    case QuantType::kQ41:
      return LinearQ41(x, weight.q4_1_data, weight.shape);
  }
  throw std::runtime_error(
      "matmul_dispatch: in LinearDispatch, unknown quantized weight type");
}

}  // namespace dqm
