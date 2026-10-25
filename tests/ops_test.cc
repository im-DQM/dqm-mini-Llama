#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "dqm_llama/model.h"
#include "dqm_llama/ops.h"
#include "dqm_llama/tensor.h"

using dqm::ArgMax;
using dqm::ElementwiseMul;
using dqm::Linear;
using dqm::Matmul;
using dqm::RmsNorm;
using dqm::Rope;
using dqm::RopeType;
using dqm::Silu;
using dqm::Softmax;
using dqm::SwiGlu;
using dqm::Tensor;

// ===========================================================================
// 壳函数：转发 smoke
// ===========================================================================

TEST(OpsShell, MatmulForwardsToDispatch) {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f; a[1] = 2.0f;
  a[2] = 3.0f; a[3] = 4.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 5.0f; b[1] = 6.0f;
  b[2] = 7.0f; b[3] = 8.0f;

  Tensor c = Matmul(a, b);
  EXPECT_NEAR(c.At2(0, 0), 19.0f, 1e-5f);
  EXPECT_NEAR(c.At2(1, 1), 50.0f, 1e-5f);
}

TEST(OpsShell, LinearForwardsToDispatch) {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
  Tensor w({3, 3}, 0.0f);
  w[0] = 1.0f; w[4] = 1.0f; w[8] = 1.0f;  // identity

  Tensor y = Linear(x, w);
  EXPECT_EQ(y.num_dims(), 1);
  EXPECT_NEAR(y[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y[2], 3.0f, 1e-5f);
}

// ===========================================================================
// RmsNorm
// ===========================================================================

TEST(RmsNorm, UnitInputGivesUnitOutput) {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  ASSERT_EQ(y.shape[0], 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(y[i], 1.0f, 1e-4f);
  }
}

TEST(RmsNorm, ScaledInput) {
  Tensor x({4}, 0.0f);
  x[0] = 2.0f;
  Tensor w({4}, 1.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  EXPECT_NEAR(y[0], 2.0f, 1e-4f);
  EXPECT_NEAR(y[1], 0.0f, 1e-4f);
}

TEST(RmsNorm, WeightScalesOutput) {
  Tensor x({4}, 1.0f);
  Tensor w({4}, 2.0f);
  Tensor y = RmsNorm(x, w, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(y[i], 2.0f, 1e-4f);
  }
}

TEST(RmsNorm, RejectsShapeMismatch) {
  Tensor x({4}, 0.0f);
  Tensor w({3}, 0.0f);
  EXPECT_THROW(RmsNorm(x, w, 1e-5f), std::runtime_error);
}

// ===========================================================================
// Softmax
// ===========================================================================

TEST(Softmax, KnownValues) {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
  Tensor y = Softmax(x);
  ASSERT_EQ(y.shape[0], 3);
  EXPECT_NEAR(y[0], 0.09003057f, 1e-5f);
  EXPECT_NEAR(y[1], 0.24472847f, 1e-5f);
  EXPECT_NEAR(y[2], 0.66524096f, 1e-5f);
  EXPECT_NEAR(y[0] + y[1] + y[2], 1.0f, 1e-5f);
}

TEST(Softmax, UniformInputGivesUniformOutput) {
  Tensor x({3}, 0.0f);
  Tensor y = Softmax(x);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(y[i], 1.0f / 3.0f, 1e-5f);
  }
}

TEST(Softmax, RejectsNon1D) {
  Tensor x({2, 3}, 0.0f);
  EXPECT_THROW(Softmax(x), std::runtime_error);
}

TEST(Softmax, RejectsEmpty) {
  Tensor x;
  x.shape = {0};
  EXPECT_THROW(Softmax(x), std::runtime_error);
}

// ===========================================================================
// Silu
// ===========================================================================

TEST(Silu, KnownValues) {
  Tensor x({1}, 0.0f);
  EXPECT_NEAR(Silu(x)[0], 0.0f, 1e-5f);

  x[0] = 1.0f;
  EXPECT_NEAR(Silu(x)[0], 0.7310586f, 1e-5f);
}

// ===========================================================================
// SwiGlu
// ===========================================================================

TEST(SwiGlu, KnownValues) {
  Tensor gate({3}, 0.0f);
  gate[0] = 0.0f; gate[1] = 1.0f; gate[2] = 2.0f;
  Tensor up({3}, 0.0f);
  up[0] = 1.0f; up[1] = 2.0f; up[2] = 3.0f;

  Tensor y = SwiGlu(gate, up);
  EXPECT_NEAR(y[0], 0.0f, 1e-5f);
  EXPECT_NEAR(y[1], 0.7310586f * 2.0f, 1e-5f);
  EXPECT_NEAR(y[2], 1.7615942f * 3.0f, 1e-4f);
}

TEST(SwiGlu, RejectsShapeMismatch) {
  Tensor a({3}, 0.0f);
  Tensor b({4}, 0.0f);
  EXPECT_THROW(SwiGlu(a, b), std::runtime_error);
}

// ===========================================================================
// ElementwiseMul
// ===========================================================================

TEST(ElementwiseMul, KnownValues) {
  Tensor a({3}, 0.0f);
  a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;
  Tensor b({3}, 0.0f);
  b[0] = 4.0f; b[1] = 5.0f; b[2] = 6.0f;

  Tensor y = ElementwiseMul(a, b);
  EXPECT_NEAR(y[0], 4.0f, 1e-5f);
  EXPECT_NEAR(y[1], 10.0f, 1e-5f);
  EXPECT_NEAR(y[2], 18.0f, 1e-5f);
}

TEST(ElementwiseMul, RejectsShapeMismatch) {
  Tensor a({3}, 0.0f);
  Tensor b({4}, 0.0f);
  EXPECT_THROW(ElementwiseMul(a, b), std::runtime_error);
}

// ===========================================================================
// ArgMax
// ===========================================================================

TEST(ArgMax, ReturnsIndexOfMaximum) {
  Tensor x({5}, 0.0f);
  x[0] = 1.0f; x[1] = 5.0f; x[2] = 3.0f; x[3] = 5.0f; x[4] = 2.0f;
  EXPECT_EQ(ArgMax(x), 1);  // 并列取第一个
}

TEST(ArgMax, RejectsEmpty) {
  Tensor x;
  EXPECT_THROW(ArgMax(x), std::runtime_error);
}

// ===========================================================================
// Rope
// ===========================================================================

TEST(Rope, PreservesNorm) {
  int n_heads = 2;
  int head_dim = 4;
  Tensor q({n_heads, head_dim}, 1.0f);
  Tensor k({n_heads, head_dim}, 1.0f);

  Rope(q, k, 5, 10000.0f);

  for (int h = 0; h < n_heads; ++h) {
    float q_norm = 0.0f;
    float k_norm = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      EXPECT_TRUE(std::isfinite(q.At2(h, d)));
      EXPECT_TRUE(std::isfinite(k.At2(h, d)));
      q_norm += q.At2(h, d) * q.At2(h, d);
      k_norm += k.At2(h, d) * k.At2(h, d);
    }
    EXPECT_NEAR(q_norm, 4.0f, 1e-5f);
    EXPECT_NEAR(k_norm, 4.0f, 1e-5f);
  }
}

TEST(Rope, NeoXMatchesSplitHalfPairs) {
  Tensor q({1, 4}, 0.0f);
  q[0] = 1.0f; q[1] = 2.0f; q[2] = 3.0f; q[3] = 4.0f;
  Tensor k = q;

  Rope(q, k, 1, 10000.0f, RopeType::kNeoX);

  float c0 = std::cos(1.0f);
  float s0 = std::sin(1.0f);
  float c1 = std::cos(0.01f);
  float s1 = std::sin(0.01f);

  EXPECT_NEAR(q[0], 1.0f * c0 - 3.0f * s0, 1e-5f);
  EXPECT_NEAR(q[2], 1.0f * s0 + 3.0f * c0, 1e-5f);
  EXPECT_NEAR(q[1], 2.0f * c1 - 4.0f * s1, 1e-5f);
  EXPECT_NEAR(q[3], 2.0f * s1 + 4.0f * c1, 1e-5f);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(k[i], q[i], 1e-5f);
  }
}

// 位置必须改变旋转结果：pos=0 旋转角为 0（恒等），pos>0 才真正旋转。
TEST(Rope, PositionChangesRotation) {
  Tensor base({1, 4}, 0.0f);
  base[0] = 1.0f; base[1] = 2.0f; base[2] = 3.0f; base[3] = 4.0f;

  Tensor q0 = base;
  Tensor k0 = base;
  Tensor q1 = base;
  Tensor k1 = base;

  Rope(q0, k0, 0, 10000.0f);
  Rope(q1, k1, 1, 10000.0f);

  // pos=0：旋转角恒为 0，q/k 原地不动
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(q0[i], base[i], 1e-6f);
  }

  // pos 不同 -> 旋转角不同 -> 结果不同
  bool moved = false;
  for (int i = 0; i < 4; ++i) {
    if (std::fabs(q1[i] - q0[i]) > 1e-6f) {
      moved = true;
    }
  }
  EXPECT_TRUE(moved);
}

TEST(Rope, RejectsNon2D) {
  Tensor q({4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(Rope, RejectsMismatchedHeadDim) {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 6}, 0.0f);
  EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(Rope, RejectsOddHeadDim) {
  Tensor q({2, 3}, 0.0f);
  Tensor k({2, 3}, 0.0f);
  EXPECT_THROW(Rope(q, k, 0, 10000.0f), std::runtime_error);
}

TEST(Rope, RejectsNegativePosition) {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  EXPECT_THROW(Rope(q, k, -1, 10000.0f), std::out_of_range);
}

TEST(Rope, RejectsNonpositiveTheta) {
  Tensor q({2, 4}, 0.0f);
  Tensor k({2, 4}, 0.0f);
  EXPECT_THROW(Rope(q, k, 0, 0.0f), std::runtime_error);
}
