#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "dqm_llama/quant.h"
#include "dqm_llama/quantized_tensor.h"

// FP16 位模式常量（用 dqm::Block*::d 的 uint16_t 形式）
namespace {
constexpr uint16_t kFp16One = 0x3C00;   // 1.0
constexpr uint16_t kFp16Two = 0x4000;   // 2.0
constexpr uint16_t kFp16Half = 0x3800;  // 0.5
}  // namespace

// ---------------------------------------------------------------------------
// Q8_0：x = d * q，q 是 int8
// ---------------------------------------------------------------------------
TEST(Quant, DequantizeQ80Basic) {
  dqm::BlockQ80 block{};
  block.d = kFp16One;
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    block.qs[i] = static_cast<int8_t>(i - 16);
  }

  dqm::Tensor t = dqm::DequantizeFromQ80({block}, {dqm::kQ80BlockSize});

  ASSERT_EQ(t.shape, (std::vector<int>{dqm::kQ80BlockSize}));
  ASSERT_EQ(t.size(), 32u);
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    EXPECT_FLOAT_EQ(t.data[static_cast<size_t>(i)],
                    static_cast<float>(i - 16));
  }
}

// 2 行：每行一个块，各自用不同的 d。
TEST(Quant, DequantizeQ80TwoRows) {
  dqm::BlockQ80 a{};
  a.d = kFp16One;
  dqm::BlockQ80 b{};
  b.d = kFp16Two;
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    a.qs[i] = 1;
    b.qs[i] = 1;
  }

  dqm::Tensor t = dqm::DequantizeFromQ80({a, b}, {2, dqm::kQ80BlockSize});

  ASSERT_EQ(t.size(), 64u);
  EXPECT_FLOAT_EQ(t.data[0], 1.0f);   // 1.0 * 1
  EXPECT_FLOAT_EQ(t.data[32], 2.0f);  // 2.0 * 1
}

// 行尾不满一块：只写到 row_size 为止。
TEST(Quant, DequantizeQ80TailPartialBlock) {
  dqm::BlockQ80 a{};
  a.d = kFp16One;
  dqm::BlockQ80 b{};
  b.d = kFp16One;
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    a.qs[i] = static_cast<int8_t>(i);
    b.qs[i] = static_cast<int8_t>(i);
  }

  dqm::Tensor t = dqm::DequantizeFromQ80({a, b}, {40});

  ASSERT_EQ(t.size(), 40u);
  EXPECT_FLOAT_EQ(t.data[32], 0.0f);  // b.qs[0]
  EXPECT_FLOAT_EQ(t.data[39], 7.0f);  // b.qs[7]
}

// 块数不足 -> 抛 runtime_error
TEST(Quant, DequantizeQ80BlockCountMismatch) {
  dqm::BlockQ80 block{};
  EXPECT_THROW(dqm::DequantizeFromQ80({block}, {64}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Q4_0：x = d * (q - 8)，两个 4-bit 塞一字节
// ---------------------------------------------------------------------------
TEST(Quant, DequantizeQ40Basic) {
  dqm::BlockQ40 block{};
  block.d = kFp16One;
  for (int j = 0; j < dqm::kQ40BlockSize / 2; ++j) {
    int lo = j;       // 低半字节 -> 下标 j
    int hi = 15 - j;  // 高半字节 -> 下标 j + 16
    block.qs[j] = static_cast<uint8_t>((hi << 4) | lo);
  }

  dqm::Tensor t = dqm::DequantizeFromQ40({block}, {dqm::kQ40BlockSize});

  ASSERT_EQ(t.size(), 32u);
  for (int j = 0; j < dqm::kQ40BlockSize / 2; ++j) {
    EXPECT_FLOAT_EQ(t.data[static_cast<size_t>(j)],
                    static_cast<float>(j - 8));
    EXPECT_FLOAT_EQ(t.data[static_cast<size_t>(j + 16)],
                    static_cast<float>((15 - j) - 8));
  }
}

// ---------------------------------------------------------------------------
// Q4_1：x = d * q + m
// ---------------------------------------------------------------------------
TEST(Quant, DequantizeQ41Basic) {
  dqm::BlockQ41 block{};
  block.d = kFp16One;   // 1.0
  block.m = kFp16Half;  // 0.5
  for (int j = 0; j < dqm::kQ41BlockSize / 2; ++j) {
    block.qs[j] = static_cast<uint8_t>((j << 4) | j);
  }

  dqm::Tensor t = dqm::DequantizeFromQ41({block}, {dqm::kQ41BlockSize});

  ASSERT_EQ(t.size(), 32u);
  for (int j = 0; j < dqm::kQ41BlockSize / 2; ++j) {
    EXPECT_FLOAT_EQ(t.data[static_cast<size_t>(j)],
                    static_cast<float>(j) + 0.5f);
    EXPECT_FLOAT_EQ(t.data[static_cast<size_t>(j + 16)],
                    static_cast<float>(j) + 0.5f);
  }
}

// ---------------------------------------------------------------------------
// 真·量化 Linear（on-the-fly）：权重块保存在内存，边算边反量化
// ---------------------------------------------------------------------------
TEST(Quant, LinearQ80Basic) {
  dqm::BlockQ80 r0{};
  r0.d = kFp16One;  // 权重 1.0
  dqm::BlockQ80 r1{};
  r1.d = kFp16Two;  // 权重 2.0
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    r0.qs[i] = 1;
    r1.qs[i] = 1;
  }
  dqm::Tensor x({dqm::kQ80BlockSize}, 1.0f);  // 全 1

  dqm::Tensor y = dqm::LinearQ80(x, {r0, r1}, {2, dqm::kQ80BlockSize});

  ASSERT_EQ(y.shape, (std::vector<int>{2}));
  EXPECT_FLOAT_EQ(y.data[0], 32.0f);  // 1.0 * sum(x)=32
  EXPECT_FLOAT_EQ(y.data[1], 64.0f);  // 2.0 * sum(x)=32
}

TEST(Quant, LinearQ41Basic) {
  dqm::BlockQ41 r{};
  r.d = kFp16One;   // 1.0
  r.m = kFp16Half;  // 0.5  -> 权重 = 1*1 + 0.5 = 1.5
  for (int j = 0; j < dqm::kQ41BlockSize / 2; ++j) {
    r.qs[j] = static_cast<uint8_t>((1 << 4) | 1);
  }
  dqm::Tensor x({dqm::kQ41BlockSize}, 1.0f);

  dqm::Tensor y = dqm::LinearQ41(x, {r}, {1, dqm::kQ41BlockSize});

  ASSERT_EQ(y.shape, (std::vector<int>{1}));
  EXPECT_FLOAT_EQ(y.data[0], 1.5f * 32.0f);
}

TEST(Quant, LinearQ80DimensionMismatch) {
  dqm::BlockQ80 r{};
  dqm::Tensor x({16}, 1.0f);
  EXPECT_THROW(dqm::LinearQ80(x, {r}, {1, 32}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// 量化（写路径）：F32 -> 块。往返误差应 <= 块内 d/2。
// ---------------------------------------------------------------------------

// Q8_0 往返：max_abs = 1.0 -> d = 1/127，误差 <= d/2。
TEST(Quant, QuantizeQ80RoundTrip) {
  dqm::Tensor src({dqm::kQ80BlockSize}, 0.0f);
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    src.data[static_cast<size_t>(i)] = static_cast<float>(i - 16) / 16.0f;  // -1.0 .. 0.9375
  }

  std::vector<dqm::BlockQ80> blocks = dqm::QuantizeToQ80(src);
  ASSERT_EQ(blocks.size(), 1u);

  dqm::Tensor out = dqm::DequantizeFromQ80(blocks, src.shape);
  const float tol = (1.0f / 127.0f) * 0.5f + 1e-6f;
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    EXPECT_NEAR(out.data[static_cast<size_t>(i)],
                src.data[static_cast<size_t>(i)], tol);
  }
}

// d 的 FP16 位模式与上界夹取：max_abs = 127 -> d = 1.0（FP16 0x3C00），q = 127。
TEST(Quant, QuantizeQ80ScaleIsOne) {
  dqm::Tensor src({dqm::kQ80BlockSize}, 0.0f);
  src.data[0] = 127.0f;

  std::vector<dqm::BlockQ80> blocks = dqm::QuantizeToQ80(src);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks[0].d, kFp16One);
  EXPECT_EQ(blocks[0].qs[0], 127);

  dqm::Tensor out = dqm::DequantizeFromQ80(blocks, src.shape);
  EXPECT_FLOAT_EQ(out.data[0], 127.0f);
}

// 全零块：d = 0，反量化得全 0。
TEST(Quant, QuantizeQ80ZeroBlock) {
  dqm::Tensor src({dqm::kQ80BlockSize}, 0.0f);

  std::vector<dqm::BlockQ80> blocks = dqm::QuantizeToQ80(src);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks[0].d, 0);

  dqm::Tensor out = dqm::DequantizeFromQ80(blocks, src.shape);
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    EXPECT_FLOAT_EQ(out.data[static_cast<size_t>(i)], 0.0f);
  }
}

// 行尾不满一块：40 个元素 -> 2 块，尾部按 0 填。
TEST(Quant, QuantizeQ80PartialBlock) {
  dqm::Tensor src({40}, 1.0f);

  std::vector<dqm::BlockQ80> blocks = dqm::QuantizeToQ80(src);
  EXPECT_EQ(blocks.size(), 2u);

  dqm::Tensor out = dqm::DequantizeFromQ80(blocks, src.shape);
  ASSERT_EQ(out.size(), 40u);
  for (int i = 0; i < 40; ++i) {
    EXPECT_NEAR(out.data[static_cast<size_t>(i)], 1.0f, 0.01f);
  }
}

// 空张量 -> 空块序列。
TEST(Quant, QuantizeQ80Empty) {
  dqm::Tensor src;
  EXPECT_TRUE(dqm::QuantizeToQ80(src).empty());
}

// 2 行各自一块：块数 = 2，两行分别用自己的 d。
TEST(Quant, QuantizeQ80TwoRows) {
  dqm::Tensor src({2, dqm::kQ80BlockSize}, 0.0f);
  for (int i = 0; i < dqm::kQ80BlockSize; ++i) {
    src.data[static_cast<size_t>(i)] = 1.0f;  // 行 0：max_abs = 1
    src.data[static_cast<size_t>(dqm::kQ80BlockSize + i)] = 2.0f;  // 行 1
  }

  std::vector<dqm::BlockQ80> blocks = dqm::QuantizeToQ80(src);
  ASSERT_EQ(blocks.size(), 2u);

  dqm::Tensor out = dqm::DequantizeFromQ80(blocks, src.shape);
  EXPECT_NEAR(out.data[0], 1.0f, 0.01f);
  EXPECT_NEAR(out.data[static_cast<size_t>(dqm::kQ80BlockSize)], 2.0f, 0.02f);
}

// Q4_0 往返：max_abs = 1.0 -> d = 1/7，误差 <= d/2。
TEST(Quant, QuantizeQ40RoundTrip) {
  dqm::Tensor src({dqm::kQ40BlockSize}, 0.0f);
  for (int i = 0; i < dqm::kQ40BlockSize; ++i) {
    src.data[static_cast<size_t>(i)] = static_cast<float>(i - 16) / 16.0f;  // -1.0 .. 0.9375
  }

  std::vector<dqm::BlockQ40> blocks = dqm::QuantizeToQ40(src);
  ASSERT_EQ(blocks.size(), 1u);

  dqm::Tensor out = dqm::DequantizeFromQ40(blocks, src.shape);
  const float tol = (1.0f / 7.0f) * 0.5f + 1e-6f;
  for (int i = 0; i < dqm::kQ40BlockSize; ++i) {
    EXPECT_NEAR(out.data[static_cast<size_t>(i)],
                src.data[static_cast<size_t>(i)], tol);
  }
}

// Q4_0 量程与半字节顺序：max_abs = 7 -> d = 1.0；
// 下标 0 编进低半字节(q=15)，下标 16 编进高半字节(q=8) -> 0x8F。
TEST(Quant, QuantizeQ40ScaleAndPacking) {
  dqm::Tensor src({dqm::kQ40BlockSize}, 0.0f);
  src.data[0] = 7.0f;  // 低半字节 -> q = round(7*1 + 8) = 15

  std::vector<dqm::BlockQ40> blocks = dqm::QuantizeToQ40(src);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks[0].d, kFp16One);
  EXPECT_EQ(blocks[0].qs[0], static_cast<uint8_t>(15 | (8 << 4)));

  dqm::Tensor out = dqm::DequantizeFromQ40(blocks, src.shape);
  EXPECT_FLOAT_EQ(out.data[0], 7.0f);
}

// Q4_0 全零块：d = 0，反量化得全 0。
TEST(Quant, QuantizeQ40ZeroBlock) {
  dqm::Tensor src({dqm::kQ40BlockSize}, 0.0f);

  std::vector<dqm::BlockQ40> blocks = dqm::QuantizeToQ40(src);
  ASSERT_EQ(blocks.size(), 1u);
  EXPECT_EQ(blocks[0].d, 0);

  dqm::Tensor out = dqm::DequantizeFromQ40(blocks, src.shape);
  for (int i = 0; i < dqm::kQ40BlockSize; ++i) {
    EXPECT_FLOAT_EQ(out.data[static_cast<size_t>(i)], 0.0f);
  }
}

// Q4_0 空张量。
TEST(Quant, QuantizeQ40Empty) {
  dqm::Tensor src;
  EXPECT_TRUE(dqm::QuantizeToQ40(src).empty());
}
