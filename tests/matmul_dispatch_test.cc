#include <gtest/gtest.h>

#include <stdexcept>

#include "dqm_llama/matmul_dispatch.h"
#include "dqm_llama/tensor.h"

using dqm::LinearDispatch;
using dqm::MatmulDispatch;
using dqm::MatmulMode;
using dqm::Tensor;

namespace {

// 用已知值填一个 F32 张量：data[i] = i * step + base
Tensor Filled(const std::vector<int>& shape, float base, float step) {
  Tensor t(shape, 0.0f);
  for (size_t i = 0; i < t.size(); ++i) {
    t.data[i] = static_cast<float>(i) * step + base;
  }
  return t;
}

}  // namespace

// ===========================================================================
// Matmul：正确性
// ===========================================================================

TEST(MatmulNaive, IdentityLeavesMatrixUnchanged) {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f; a[1] = 0.0f;
  a[2] = 0.0f; a[3] = 1.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 1.0f; b[1] = 2.0f;
  b[2] = 3.0f; b[3] = 4.0f;

  Tensor c = MatmulDispatch(a, b, MatmulMode::kNaive);

  EXPECT_EQ(c.num_dims(), 2);
  EXPECT_EQ(c.shape[0], 2);
  EXPECT_EQ(c.shape[1], 2);
  EXPECT_NEAR(c.At2(0, 0), 1.0f, 1e-5f);
  EXPECT_NEAR(c.At2(0, 1), 2.0f, 1e-5f);
  EXPECT_NEAR(c.At2(1, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(c.At2(1, 1), 4.0f, 1e-5f);
}

TEST(MatmulNaive, TwoByTwoHandComputed) {
  Tensor a({2, 2}, 0.0f);
  a[0] = 1.0f; a[1] = 2.0f;
  a[2] = 3.0f; a[3] = 4.0f;
  Tensor b({2, 2}, 0.0f);
  b[0] = 5.0f; b[1] = 6.0f;
  b[2] = 7.0f; b[3] = 8.0f;

  Tensor c = MatmulDispatch(a, b, MatmulMode::kNaive);

  EXPECT_NEAR(c.At2(0, 0), 19.0f, 1e-5f);
  EXPECT_NEAR(c.At2(0, 1), 22.0f, 1e-5f);
  EXPECT_NEAR(c.At2(1, 0), 43.0f, 1e-5f);
  EXPECT_NEAR(c.At2(1, 1), 50.0f, 1e-5f);
}

TEST(MatmulNaive, RectangularShapes) {
  // a: [2, 3], b: [3, 4] -> c: [2, 4]
  Tensor a({2, 3}, 0.0f);
  a[0] = 1.0f; a[1] = 2.0f; a[2] = 3.0f;
  a[3] = 4.0f; a[4] = 5.0f; a[5] = 6.0f;
  Tensor b = Filled({3, 4}, 0.0f, 1.0f);

  Tensor c = MatmulDispatch(a, b, MatmulMode::kNaive);

  EXPECT_EQ(c.shape[0], 2);
  EXPECT_EQ(c.shape[1], 4);
  // c[0,0] = 1*0 + 2*4 + 3*8 = 32
  EXPECT_NEAR(c.At2(0, 0), 32.0f, 1e-5f);
  // c[1,0] = 4*0 + 5*4 + 6*8 = 68
  EXPECT_NEAR(c.At2(1, 0), 68.0f, 1e-5f);
}

// ===========================================================================
// Matmul：错误
// ===========================================================================

TEST(MatmulErrors, RejectsNon2DInput) {
  Tensor a({4}, 0.0f);
  Tensor b({2, 4}, 0.0f);
  EXPECT_THROW(MatmulDispatch(a, b, MatmulMode::kNaive), std::runtime_error);
}

TEST(MatmulErrors, RejectsInnerDimensionMismatch) {
  Tensor a({2, 3}, 0.0f);
  Tensor b({2, 3}, 0.0f);
  EXPECT_THROW(MatmulDispatch(a, b, MatmulMode::kNaive), std::runtime_error);
}

// ===========================================================================
// Linear：正确性
// ===========================================================================

TEST(LinearNaive, OneDInputKeepsRankOne) {
  Tensor x({3}, 0.0f);
  x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
  Tensor w({3, 3}, 0.0f);
  w[0] = 1.0f; w[4] = 1.0f; w[8] = 1.0f;  // identity

  Tensor y = LinearDispatch(x, w, MatmulMode::kNaive);

  EXPECT_EQ(y.num_dims(), 1);
  EXPECT_EQ(y.shape[0], 3);
  EXPECT_NEAR(y[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y[1], 2.0f, 1e-5f);
  EXPECT_NEAR(y[2], 3.0f, 1e-5f);
}

TEST(LinearNaive, RowVectorInputKeepsRankTwo) {
  Tensor x({1, 3}, 0.0f);
  x[0] = 1.0f; x[1] = 2.0f; x[2] = 3.0f;
  Tensor w({2, 3}, 0.0f);
  w[0] = 1.0f;              // row 0: [1, 0, 0] -> picks x[0]
  w[4] = 1.0f;              // row 1: [0, 1, 0] -> picks x[1]

  Tensor y = LinearDispatch(x, w, MatmulMode::kNaive);

  EXPECT_EQ(y.num_dims(), 2);
  EXPECT_EQ(y.shape[0], 1);
  EXPECT_EQ(y.shape[1], 2);
  EXPECT_NEAR(y[0], 1.0f, 1e-5f);
  EXPECT_NEAR(y[1], 2.0f, 1e-5f);
}

// ===========================================================================
// Linear：错误
// ===========================================================================

TEST(LinearErrors, RejectsNon2DWeight) {
  Tensor x({3}, 0.0f);
  Tensor w({3}, 0.0f);
  EXPECT_THROW(LinearDispatch(x, w, MatmulMode::kNaive), std::runtime_error);
}

TEST(LinearErrors, Rejects2DInputWithLeadingDimNotOne) {
  Tensor x({3, 4}, 0.0f);   // 2D 但首维 != 1
  Tensor w({2, 4}, 0.0f);
  EXPECT_THROW(LinearDispatch(x, w, MatmulMode::kNaive), std::runtime_error);
}

TEST(LinearErrors, RejectsInnerDimensionMismatch) {
  Tensor x({3}, 0.0f);
  Tensor w({2, 4}, 0.0f);
  EXPECT_THROW(LinearDispatch(x, w, MatmulMode::kNaive), std::runtime_error);
}

// ===========================================================================
// 默认档 & 未实现档
// ===========================================================================

TEST(DispatchDefault, DefaultModeIsNaiveThisRound) {
  EXPECT_EQ(dqm::DefaultMatmulMode(), MatmulMode::kNaive);
}

TEST(DispatchUnimplemented, MatmulUnimplementedModesThrowLogicError) {
  Tensor a({2, 2}, 0.0f);
  Tensor b({2, 2}, 0.0f);
  EXPECT_THROW(MatmulDispatch(a, b, MatmulMode::kThreaded), std::logic_error);
  EXPECT_THROW(MatmulDispatch(a, b, MatmulMode::kSimd), std::logic_error);
  EXPECT_THROW(MatmulDispatch(a, b, MatmulMode::kThreadedSimd), std::logic_error);
}

TEST(DispatchUnimplemented, LinearUnimplementedModesThrowLogicError) {
  Tensor x({3}, 0.0f);
  Tensor w({3, 3}, 0.0f);
  EXPECT_THROW(LinearDispatch(x, w, MatmulMode::kThreaded), std::logic_error);
  EXPECT_THROW(LinearDispatch(x, w, MatmulMode::kSimd), std::logic_error);
  EXPECT_THROW(LinearDispatch(x, w, MatmulMode::kThreadedSimd), std::logic_error);
}
