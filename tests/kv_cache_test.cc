#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "dqm_llama/kv_cache.h"
#include "dqm_llama/tensor.h"

using dqm::KvCache;
using dqm::Tensor;

// ===========================================================================
// 构造
// ===========================================================================

TEST(KvCache, ConstructsExpectedStorageShape) {
  KvCache cache(2, 8, 3, 4);
  EXPECT_EQ(cache.keys.shape, (std::vector<int>{2, 8, 3, 4}));
  EXPECT_EQ(cache.values.shape, (std::vector<int>{2, 8, 3, 4}));
  EXPECT_EQ(cache.keys.size(), 192u);
  EXPECT_EQ(cache.values.size(), 192u);
  for (size_t i = 0; i < cache.keys.size(); ++i) {
    EXPECT_NEAR(cache.keys.data[i], 0.0f, 1e-6f);
    EXPECT_NEAR(cache.values.data[i], 0.0f, 1e-6f);
  }
}

// ===========================================================================
// Write + Ptr 回读
// ===========================================================================

TEST(KvCache, WriteAndPtrReadback) {
  KvCache cache(2, 4, 2, 3);
  Tensor k({2, 3}, 0.0f);
  Tensor v({2, 3}, 0.0f);
  for (int i = 0; i < 6; ++i) {
    k[i] = static_cast<float>(i + 1);    // 1..6
    v[i] = static_cast<float>(100 + i);  // 100..105
  }

  cache.Write(1, 2, k, v);

  const float* key0 = cache.KeyPtr(1, 2, 0);
  const float* key1 = cache.KeyPtr(1, 2, 1);
  const float* val0 = cache.ValuePtr(1, 2, 0);
  const float* val1 = cache.ValuePtr(1, 2, 1);

  EXPECT_NEAR(key0[0], 1.0f, 1e-6f);
  EXPECT_NEAR(key0[1], 2.0f, 1e-6f);
  EXPECT_NEAR(key0[2], 3.0f, 1e-6f);
  EXPECT_NEAR(key1[0], 4.0f, 1e-6f);
  EXPECT_NEAR(key1[1], 5.0f, 1e-6f);
  EXPECT_NEAR(key1[2], 6.0f, 1e-6f);

  EXPECT_NEAR(val0[0], 100.0f, 1e-6f);
  EXPECT_NEAR(val0[1], 101.0f, 1e-6f);
  EXPECT_NEAR(val0[2], 102.0f, 1e-6f);
  EXPECT_NEAR(val1[0], 103.0f, 1e-6f);
  EXPECT_NEAR(val1[1], 104.0f, 1e-6f);
  EXPECT_NEAR(val1[2], 105.0f, 1e-6f);
}

// ===========================================================================
// 隔离性：写一个槽不污染别的槽
// ===========================================================================

TEST(KvCache, WritePreservesOtherPositions) {
  KvCache cache(1, 3, 1, 2);
  Tensor k({1, 2}, 0.0f);
  Tensor v({1, 2}, 0.0f);
  k[0] = 7.0f;
  k[1] = 8.0f;
  v[0] = 9.0f;
  v[1] = 10.0f;

  cache.Write(0, 1, k, v);

  const float* untouched_key = cache.KeyPtr(0, 0, 0);
  const float* untouched_value = cache.ValuePtr(0, 2, 0);
  EXPECT_NEAR(untouched_key[0], 0.0f, 1e-6f);
  EXPECT_NEAR(untouched_key[1], 0.0f, 1e-6f);
  EXPECT_NEAR(untouched_value[0], 0.0f, 1e-6f);
  EXPECT_NEAR(untouched_value[1], 0.0f, 1e-6f);
}

// ===========================================================================
// 错误：越界下标
// ===========================================================================

TEST(KvCache, RejectsBadIndices) {
  KvCache cache(1, 2, 1, 2);
  Tensor k({1, 2}, 0.0f);
  Tensor v({1, 2}, 0.0f);

  EXPECT_THROW(cache.Write(-1, 0, k, v), std::out_of_range);  // layer 下溢
  EXPECT_THROW(cache.Write(0, 2, k, v), std::out_of_range);   // pos 上溢
  EXPECT_THROW(cache.KeyPtr(0, 0, 1), std::out_of_range);     // head 上溢
  EXPECT_THROW(cache.ValuePtr(0, -1, 0), std::out_of_range);  // pos 负
}

// ===========================================================================
// 错误：k / v 形状不对
// ===========================================================================

TEST(KvCache, RejectsBadTensorShapes) {
  KvCache cache(1, 2, 1, 2);
  Tensor good({1, 2}, 0.0f);
  Tensor rank_3({1, 1, 2}, 0.0f);
  Tensor wrong_heads({2, 2}, 0.0f);
  Tensor wrong_dim({1, 3}, 0.0f);

  EXPECT_THROW(cache.Write(0, 0, rank_3, good), std::out_of_range);
  EXPECT_THROW(cache.Write(0, 0, wrong_heads, wrong_heads), std::out_of_range);
  EXPECT_THROW(cache.Write(0, 0, wrong_dim, wrong_dim), std::out_of_range);
}

// ===========================================================================
// 错误：cache 本身不是 4D
// ===========================================================================

TEST(KvCache, RejectsNon4DCache) {
  KvCache empty;  // 默认构造：keys / values 是空张量（0 维）
  Tensor k({1, 2}, 0.0f);
  Tensor v({1, 2}, 0.0f);

  EXPECT_THROW(empty.Write(0, 0, k, v), std::runtime_error);
  EXPECT_THROW(empty.KeyPtr(0, 0, 0), std::runtime_error);
}
