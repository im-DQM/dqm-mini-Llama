#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

using dqm::BlockQ40;
using dqm::BlockQ41;
using dqm::BlockQ80;
using dqm::kQ40BlockSize;
using dqm::kQ41BlockSize;
using dqm::kQ80BlockSize;
using dqm::QuantizedTensor;
using dqm::QuantType;
using dqm::Tensor;
using dqm::ToQuantizedTensor;
using dqm::ToTensor;

// ===========================================================================
// 内存布局（编译期契约）
//
// 这和 ggml / llama.cpp 的 block 布局是二进制兼容的约定：将来 loader 按
// 字节偏移去解析 GGUF 里的量化权重，块大小错一个字节，整条链全错。
// 所以把 sizeof 钉死 —— 编译器一旦插入 padding 改变布局，这里就红。
// ===========================================================================

TEST(QuantizedTensorBlockLayout, BlockSizes) {
  EXPECT_EQ(sizeof(BlockQ80), sizeof(uint16_t) + kQ80BlockSize);
  EXPECT_EQ(sizeof(BlockQ40), sizeof(uint16_t) + kQ40BlockSize / 2u);
  EXPECT_EQ(sizeof(BlockQ41), 2u * sizeof(uint16_t) + kQ41BlockSize / 2u);
}

// 三个常量都等于 32：它们形容的是「一块覆盖多少个元素」，不是「一块多少字节」
TEST(QuantizedTensorBlockLayout, BlockSizeConstants) {
  EXPECT_EQ(kQ80BlockSize, 32);
  EXPECT_EQ(kQ40BlockSize, 32);
  EXPECT_EQ(kQ41BlockSize, 32);
}

// Q4 系一个字节装两个 4-bit 值 -> 32 个元素只用 16 字节
TEST(QuantizedTensorBlockLayout, Q4PacksTwoValuesPerByte) {
  EXPECT_EQ(sizeof(BlockQ40) - sizeof(uint16_t), kQ40BlockSize / 2u);
  EXPECT_EQ(sizeof(BlockQ41) - 2u * sizeof(uint16_t), kQ41BlockSize / 2u);
}

// ===========================================================================
// 成员函数：默认状态 / 形状工具
// ===========================================================================

TEST(QuantizedTensorHelpers, DefaultConstructedIsEmptyF32) {
  QuantizedTensor q;
  EXPECT_EQ(q.type, QuantType::kF32);
  EXPECT_TRUE(q.shape.empty());
  EXPECT_TRUE(q.f32_data.empty());
  EXPECT_TRUE(q.q8_0_data.empty());
  EXPECT_TRUE(q.q4_0_data.empty());
  EXPECT_TRUE(q.q4_1_data.empty());
  EXPECT_EQ(q.num_dims(), 0);
}

TEST(QuantizedTensorHelpers, NumElementsMultipliesShape) {
  QuantizedTensor q;
  q.shape = {5};
  EXPECT_EQ(q.num_elements(), 5u);
  q.shape = {2, 3};
  EXPECT_EQ(q.num_elements(), 6u);
  q.shape = {2, 3, 4};
  EXPECT_EQ(q.num_elements(), 24u);
  q.shape = {2, 3, 4, 5};
  EXPECT_EQ(q.num_elements(), 120u);
}

// 空 shape 时「累乘」按乘法单位元算 —— 结果是 1，不是 0。
// （⚠️ 待拍板：参考实现就是这个行为，但它和 Tensor::num_elements()
//  = data.size() = 0 不一致。你若想要 0，改这条测试 + 实现里特判。）
TEST(QuantizedTensorHelpers, NumElementsOfEmptyShapeIsOne) {
  QuantizedTensor q;
  EXPECT_EQ(q.num_elements(), 1u);
}

TEST(QuantizedTensorHelpers, NumDimsTracksShapeRank) {
  QuantizedTensor q;
  EXPECT_EQ(q.num_dims(), 0);
  q.shape = {2, 3, 4};
  EXPECT_EQ(q.num_dims(), 3);
}

TEST(QuantizedTensorHelpers, ShapeStringShortFormats) {
  QuantizedTensor q;
  EXPECT_EQ(q.ShapeStringShort(), "[]");
  q.shape = {5};
  EXPECT_EQ(q.ShapeStringShort(), "[5]");
  q.shape = {2, 3};
  EXPECT_EQ(q.ShapeStringShort(), "[2, 3]");
  q.shape = {128, 32, 4};
  EXPECT_EQ(q.ShapeStringShort(), "[128, 32, 4]");
}

// ===========================================================================
// 成员函数：AssertShape
// ===========================================================================

TEST(QuantizedTensorHelpers, AssertShapeAcceptsMatchingShape) {
  QuantizedTensor q;
  q.shape = {2, 3};
  EXPECT_NO_THROW(q.AssertShape({2, 3}, "test_caller"));

  QuantizedTensor empty;
  EXPECT_NO_THROW(empty.AssertShape({}, "test_caller"));
}

TEST(QuantizedTensorHelpers, AssertShapeThrowsOnMismatch) {
  QuantizedTensor q;
  q.shape = {2, 3};

  try {
    q.AssertShape({4, 3}, "my_caller");
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("my_caller"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[4, 3]"), std::string::npos) << msg;  // 期望的
    EXPECT_NE(msg.find("[2, 3]"), std::string::npos) << msg;  // 实际的
  }
}

TEST(QuantizedTensorHelpers, AssertShapeThrowsOnRankMismatch) {
  QuantizedTensor q;
  q.shape = {2, 3};
  EXPECT_THROW(q.AssertShape({2, 3, 4}, "test_caller"), std::runtime_error);
  EXPECT_THROW(q.AssertShape({}, "test_caller"), std::runtime_error);
}

// ===========================================================================
// 带 tag 的容器语义
// ===========================================================================

// 聚合初始化按**声明顺序**填字段：第 1 个是 type，第 2 个是 shape，
// 后面四个 data vector 全部落到默认值（空）。
TEST(QuantizedTensorTagging, AggregateInitSetsTypeAndShape) {
  QuantizedTensor q{QuantType::kQ80, {2, 3}};
  EXPECT_EQ(q.type, QuantType::kQ80);
  EXPECT_EQ(q.shape, (std::vector<int>{2, 3}));
  EXPECT_TRUE(q.f32_data.empty());
  EXPECT_TRUE(q.q8_0_data.empty());
  EXPECT_TRUE(q.q4_0_data.empty());
  EXPECT_TRUE(q.q4_1_data.empty());
}

// ===========================================================================
// ToTensor：容器 -> 普通张量
// ===========================================================================

TEST(QuantizedTensorConvert, ToTensorCopiesF32Data) {
  QuantizedTensor q;
  q.type = QuantType::kF32;
  q.shape = {2, 3};
  q.f32_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  const Tensor t = ToTensor(q);

  EXPECT_EQ(t.shape, (std::vector<int>{2, 3}));
  ASSERT_EQ(t.size(), 6u);
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.data[i], static_cast<float>(i + 1), 1e-6f);
  }

  // 入参是 const 引用：调用后容器原封不动
  EXPECT_EQ(q.type, QuantType::kF32);
  EXPECT_EQ(q.f32_data.size(), 6u);
  EXPECT_NEAR(q.f32_data[5], 6.0f, 1e-6f);
}

TEST(QuantizedTensorConvert, ToTensorRejectsQuantizedTypes) {
  QuantizedTensor q;
  q.shape = {32};

  q.type = QuantType::kQ80;
  EXPECT_THROW(ToTensor(q), std::runtime_error);

  q.type = QuantType::kQ40;
  EXPECT_THROW(ToTensor(q), std::runtime_error);

  q.type = QuantType::kQ41;
  EXPECT_THROW(ToTensor(q), std::runtime_error);
}

// 契约点：f32_data 元素数和 shape 推出来的元素数对不上时**不抛异常**，
// 而是返回一个按 shape 尺寸、全 0 的张量。宽松行为，故意钉住 ——
// 免得以后有人觉得"这明明是 bug"顺手改成抛错，把上游依赖它的路径打翻。
TEST(QuantizedTensorConvert, ToTensorWithMismatchedDataIsAllZeros) {
  QuantizedTensor q;
  q.type = QuantType::kF32;
  q.shape = {2, 3};
  q.f32_data = {1.0f, 2.0f, 3.0f};  // 只有 3 个，但 shape 要 6 个

  const Tensor t = ToTensor(q);

  EXPECT_EQ(t.shape, (std::vector<int>{2, 3}));
  ASSERT_EQ(t.size(), 6u);
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(t.data[i], 0.0f, 1e-6f);
  }
}

// ===========================================================================
// ToQuantizedTensor：普通张量 -> 容器
// ===========================================================================

TEST(QuantizedTensorConvert, ToQuantizedTensorWrapsF32Tensor) {
  Tensor t({2, 3}, 0.0f);
  for (size_t i = 0; i < t.size(); ++i) {
    t.data[i] = static_cast<float>(i) * 1.5f;
  }

  const QuantizedTensor q = ToQuantizedTensor(t);

  EXPECT_EQ(q.type, QuantType::kF32);
  EXPECT_EQ(q.shape, (std::vector<int>{2, 3}));
  ASSERT_EQ(q.f32_data.size(), t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(q.f32_data[i], t.data[i], 1e-6f);
  }

  // 只有 F32 那一路被填，另外三个保持空
  EXPECT_TRUE(q.q8_0_data.empty());
  EXPECT_TRUE(q.q4_0_data.empty());
  EXPECT_TRUE(q.q4_1_data.empty());
}

// 深拷贝：动容器里的数据不该影响原张量
TEST(QuantizedTensorConvert, ToQuantizedTensorDeepCopies) {
  Tensor t({3}, 0.0f);
  t.data = {7.0f, 8.0f, 9.0f};

  QuantizedTensor q = ToQuantizedTensor(t);

  ASSERT_EQ(t.size(), 3u);
  EXPECT_NEAR(t.data[0], 7.0f, 1e-6f);
  EXPECT_NEAR(t.data[1], 8.0f, 1e-6f);
  EXPECT_NEAR(t.data[2], 9.0f, 1e-6f);

  q.f32_data[0] = -1.0f;
  EXPECT_NEAR(t.data[0], 7.0f, 1e-6f);
}

// loader 读 .bin 出来的 F32 权重 -> 塞进 LayerWeights -> 再取回来，
// 这条往返路径必须无损，否则权重会悄悄坏掉
TEST(QuantizedTensorConvert, F32RoundTripIsLossless) {
  Tensor t({2, 2}, 0.0f);
  t.data = {-1.25f, 0.0f, 3.5f, 42.0f};

  const Tensor back = ToTensor(ToQuantizedTensor(t));

  EXPECT_EQ(back.shape, t.shape);
  ASSERT_EQ(back.size(), t.size());
  for (size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(back.data[i], t.data[i], 1e-6f);
  }
}
