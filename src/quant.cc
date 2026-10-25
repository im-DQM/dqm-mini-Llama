// quant.cc —— 反量化入口：块量化数据 -> F32 Tensor（Q8_0 / Q4_0 / Q4_1）。

#include "dqm_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "dqm_llama/thread_pool.h"

namespace dqm {

namespace {

// FP16 位模式 -> float。
float Fp16ToFloat(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = 0;

  if (exponent == 0) {
    // 非规格化数（含 ±0）
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      uint32_t exp32 = exponent + (127 - 15);
      bits = sign | (exp32 << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    // Inf / NaN
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    uint32_t exp32 = exponent + (127 - 15);
    bits = sign | (exp32 << 23) | (mantissa << 13);
  }

  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// float -> FP16 位模式（就近舍入到偶数；下溢走非规格化，上溢给 Inf）。
uint16_t FloatToFp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    uint32_t shifted = mantissa >> (1 - exponent);
    if ((shifted & 0x00001000u) != 0) {
      shifted += 0x00002000u;
    }
    return static_cast<uint16_t>(sign | (shifted >> 13));
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if ((mantissa & 0x00001000u) != 0) {
    mantissa += 0x00002000u;
    if ((mantissa & 0x00800000u) != 0) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }

  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

// 一行一块解码时共用的形状拆解：返回 (row_size, n_rows, row_blocks)。
// row_size = 行内元素数（最后一维）；非 2D 时整段算一行。
struct RowLayout {
  int row_size;
  int n_rows;
  int row_blocks;
};

RowLayout SplitRows(const std::vector<int>& shape, size_t total, int block_size) {
  RowLayout layout;
  layout.row_size =
      shape.size() >= 2 ? shape.back() : static_cast<int>(total);
  layout.n_rows =
      shape.size() >= 2 ? static_cast<int>(total) / layout.row_size : 1;
  layout.row_blocks = (layout.row_size + block_size - 1) / block_size;
  return layout;
}

size_t ExpectedBlocks(const RowLayout& layout) {
  return static_cast<size_t>(layout.n_rows) *
         static_cast<size_t>(layout.row_blocks);
}

}  // namespace

Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
  Tensor dst(shape, 0.0f);
  const size_t total = dst.size();
  if (total == 0) {
    return dst;
  }

  RowLayout layout = SplitRows(shape, total, kQ80BlockSize);
  size_t expected = ExpectedBlocks(layout);
  if (blocks.size() != expected) {
    throw std::runtime_error(
        "DequantizeFromQ80: block count mismatch: expected " +
        std::to_string(expected) + ", got " + std::to_string(blocks.size()));
  }

  size_t block_idx = 0;
  for (int row = 0; row < layout.n_rows; ++row) {
    int row_offset = row * layout.row_size;
    for (int rb = 0; rb < layout.row_blocks; ++rb) {
      int base = row_offset + rb * kQ80BlockSize;
      int k_end = std::min(base + kQ80BlockSize, row_offset + layout.row_size);
      const BlockQ80& block = blocks[block_idx++];
      float d = Fp16ToFloat(block.d);
      for (int k = base; k < k_end; ++k) {
        dst.data[static_cast<size_t>(k)] =
            d * static_cast<float>(block.qs[k - base]);
      }
    }
  }
  return dst;
}

Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape) {
  Tensor dst(shape, 0.0f);
  const size_t total = dst.size();
  if (total == 0) {
    return dst;
  }

  RowLayout layout = SplitRows(shape, total, kQ40BlockSize);
  size_t expected = ExpectedBlocks(layout);
  if (blocks.size() != expected) {
    throw std::runtime_error(
        "DequantizeFromQ40: block count mismatch: expected " +
        std::to_string(expected) + ", got " + std::to_string(blocks.size()));
  }

  size_t block_idx = 0;
  for (int row = 0; row < layout.n_rows; ++row) {
    int row_offset = row * layout.row_size;
    for (int rb = 0; rb < layout.row_blocks; ++rb) {
      int base = row_offset + rb * kQ40BlockSize;
      int k_end = std::min(base + kQ40BlockSize, row_offset + layout.row_size);
      const BlockQ40& block = blocks[block_idx++];
      float d = Fp16ToFloat(block.d);

      for (int j = 0; j < kQ40BlockSize / 2; ++j) {
        int idx0 = base + j;                             // 低半字节
        int idx1 = base + j + kQ40BlockSize / 2;         // 高半字节
        int q0 = static_cast<int>(block.qs[j] & 0x0F) - 8;
        int q1 = static_cast<int>(block.qs[j] >> 4) - 8;
        if (idx0 < k_end) {
          dst.data[static_cast<size_t>(idx0)] = d * static_cast<float>(q0);
        }
        if (idx1 < k_end) {
          dst.data[static_cast<size_t>(idx1)] = d * static_cast<float>(q1);
        }
      }
    }
  }
  return dst;
}

Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape) {
  Tensor dst(shape, 0.0f);
  const size_t total = dst.size();
  if (total == 0) {
    return dst;
  }

  RowLayout layout = SplitRows(shape, total, kQ41BlockSize);
  size_t expected = ExpectedBlocks(layout);
  if (blocks.size() != expected) {
    throw std::runtime_error(
        "DequantizeFromQ41: block count mismatch: expected " +
        std::to_string(expected) + ", got " + std::to_string(blocks.size()));
  }

  size_t block_idx = 0;
  for (int row = 0; row < layout.n_rows; ++row) {
    int row_offset = row * layout.row_size;
    for (int rb = 0; rb < layout.row_blocks; ++rb) {
      int base = row_offset + rb * kQ41BlockSize;
      int k_end = std::min(base + kQ41BlockSize, row_offset + layout.row_size);
      const BlockQ41& block = blocks[block_idx++];
      float d = Fp16ToFloat(block.d);
      float m = Fp16ToFloat(block.m);

      for (int j = 0; j < kQ41BlockSize / 2; ++j) {
        int idx0 = base + j;                             // 低半字节
        int idx1 = base + j + kQ41BlockSize / 2;         // 高半字节
        int q0 = static_cast<int>(block.qs[j] & 0x0F);
        int q1 = static_cast<int>(block.qs[j] >> 4);
        if (idx0 < k_end) {
          dst.data[static_cast<size_t>(idx0)] =
              d * static_cast<float>(q0) + m;
        }
        if (idx1 < k_end) {
          dst.data[static_cast<size_t>(idx1)] =
              d * static_cast<float>(q1) + m;
        }
      }
    }
  }
  return dst;
}

// ---------------------------------------------------------------------------
// 真·量化 Linear（块级 on-the-fly）
// ---------------------------------------------------------------------------

namespace {

// 单个块内第 idx 个元素的权重（Q8_0：x = d * q）。
float DequantQ80(const BlockQ80& block, int idx) {
  return Fp16ToFloat(block.d) * static_cast<float>(block.qs[idx]);
}

// Q4_0：x = d * (q - 8)，低半字节在前 16 个、高半字节在后 16 个。
float DequantQ40(const BlockQ40& block, int idx) {
  int q = idx < kQ40BlockSize / 2
              ? static_cast<int>(block.qs[idx] & 0x0F)
              : static_cast<int>(block.qs[idx - kQ40BlockSize / 2] >> 4);
  return Fp16ToFloat(block.d) * static_cast<float>(q - 8);
}

// Q4_1：x = d * q + m
float DequantQ41(const BlockQ41& block, int idx) {
  int q = idx < kQ41BlockSize / 2
              ? static_cast<int>(block.qs[idx] & 0x0F)
              : static_cast<int>(block.qs[idx - kQ41BlockSize / 2] >> 4);
  return Fp16ToFloat(block.d) * static_cast<float>(q) + Fp16ToFloat(block.m);
}

std::string ShapeStr(const std::vector<int>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      s += ",";
    }
    s += std::to_string(shape[i]);
  }
  s += "]";
  return s;
}

// 共用内核：权重是量化块 [out_features, in_features]，输入 [in_features]
// 或 [batch, in_features]，输出秩跟随输入。逐元素 on-the-fly 反量化。
template <typename BlockType, int kBlockSize>
Tensor LinearQuantizedImpl(const Tensor& x, const std::vector<BlockType>& blocks,
                           const std::vector<int>& weight_shape,
                           float (*dequant_fn)(const BlockType&, int)) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error("LinearQuantized: expected 2D weight shape");
  }

  int in_features;
  int rows = 1;
  bool is_1d = false;
  if (x.num_dims() == 1) {
    in_features = x.shape[0];
    is_1d = true;
  } else if (x.num_dims() == 2) {
    rows = x.shape[0];
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "LinearQuantized: expected x shape [in_features] or "
        "[batch, in_features], got " +
        x.ShapeStringShort());
  }

  int out_features = weight_shape[0];
  if (weight_shape[1] != in_features) {
    throw std::runtime_error("LinearQuantized: dimension mismatch x=" +
                             x.ShapeStringShort() +
                             " weight=" + ShapeStr(weight_shape));
  }

  int n_blocks_per_row = (in_features + kBlockSize - 1) / kBlockSize;
  size_t expected_blocks =
      static_cast<size_t>(out_features) * static_cast<size_t>(n_blocks_per_row);
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "LinearQuantized: block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  Tensor result(is_1d ? std::vector<int>{out_features}
                      : std::vector<int>{rows, out_features},
                0.0f);

  // 并行单位 = (row, j) 扁平对：每个输出元素只写 result[row, j]，互不重叠。
  // 必须是 rows * out_features 而非 rows —— decode 时 x 是 1D、rows = 1，
  // 只有按输出维展开才能拿到并行度。
  ParallelFor(rows * out_features, [&](int begin, int end) {
    for (int flat = begin; flat < end; ++flat) {
      const int row = flat / out_features;
      const int j = flat % out_features;
      const float* x_row =
          x.data.data() + static_cast<size_t>(row) * in_features;
      const int block_base = j * n_blocks_per_row;
      float sum = 0.0f;
      for (int b = 0; b < n_blocks_per_row; ++b) {
        const BlockType& block = blocks[static_cast<size_t>(block_base + b)];
        const int base_k = b * kBlockSize;
        const int k_end = std::min(base_k + kBlockSize, in_features);
        for (int k = base_k; k < k_end; ++k) {
          sum += dequant_fn(block, k - base_k) * x_row[k];
        }
      }
      result.data[static_cast<size_t>(row) * out_features + j] = sum;
    }
  });
  return result;
}

}  // namespace

Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ80, kQ80BlockSize>(x, weight, weight_shape,
                                                      DequantQ80);
}

Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ40, kQ40BlockSize>(x, weight, weight_shape,
                                                      DequantQ40);
}

Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearQuantizedImpl<BlockQ41, kQ41BlockSize>(x, weight, weight_shape,
                                                      DequantQ41);
}

// ---------------------------------------------------------------------------
// 块量化（写路径）：F32 -> Q8_0 / Q4_0
// ---------------------------------------------------------------------------

std::vector<BlockQ80> QuantizeToQ80(const Tensor& src) {
  if (src.size() == 0) {
    return {};
  }

  RowLayout layout = SplitRows(src.shape, src.size(), kQ80BlockSize);
  std::vector<BlockQ80> blocks;
  blocks.reserve(ExpectedBlocks(layout));

  for (int row = 0; row < layout.n_rows; ++row) {
    int row_offset = row * layout.row_size;
    for (int rb = 0; rb < layout.row_blocks; ++rb) {
      int base = row_offset + rb * kQ80BlockSize;
      int k_end = std::min(base + kQ80BlockSize, row_offset + layout.row_size);
      int block_len = k_end - base;

      float max_abs = 0.0f;
      for (int k = base; k < k_end; ++k) {
        float abs_val = std::abs(src.data[static_cast<size_t>(k)]);
        if (abs_val > max_abs) {
          max_abs = abs_val;
        }
      }

      BlockQ80 block{};
      if (max_abs > 0.0f) {
        float d = max_abs / 127.0f;
        block.d = FloatToFp16(d);
        float stored_d = Fp16ToFloat(block.d);
        float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;
        for (int i = 0; i < block_len; ++i) {
          int q = static_cast<int>(
              std::round(src.data[static_cast<size_t>(base + i)] * id));
          if (q > 127) {
            q = 127;
          } else if (q < -127) {
            q = -127;
          }
          block.qs[i] = static_cast<int8_t>(q);
        }
      }
      // 全零块：d 保持 0；尾部字节由 block{} 的零初始化覆盖。
      blocks.push_back(block);
    }
  }
  return blocks;
}

std::vector<BlockQ40> QuantizeToQ40(const Tensor& src) {
  if (src.size() == 0) {
    return {};
  }

  RowLayout layout = SplitRows(src.shape, src.size(), kQ40BlockSize);
  std::vector<BlockQ40> blocks;
  blocks.reserve(ExpectedBlocks(layout));

  for (int row = 0; row < layout.n_rows; ++row) {
    int row_offset = row * layout.row_size;
    for (int rb = 0; rb < layout.row_blocks; ++rb) {
      int base = row_offset + rb * kQ40BlockSize;
      int k_end = std::min(base + kQ40BlockSize, row_offset + layout.row_size);

      float max_abs = 0.0f;
      for (int k = base; k < k_end; ++k) {
        float abs_val = std::abs(src.data[static_cast<size_t>(k)]);
        if (abs_val > max_abs) {
          max_abs = abs_val;
        }
      }

      BlockQ40 block{};
      if (max_abs > 0.0f) {
        float d = max_abs / 7.0f;
        block.d = FloatToFp16(d);
        float stored_d = Fp16ToFloat(block.d);
        float id = stored_d == 0.0f ? 0.0f : 1.0f / stored_d;

        for (int j = 0; j < kQ40BlockSize / 2; ++j) {
          int idx0 = base + j;
          int idx1 = base + j + kQ40BlockSize / 2;
          float x0 =
              idx0 < k_end ? src.data[static_cast<size_t>(idx0)] * id : 0.0f;
          float x1 =
              idx1 < k_end ? src.data[static_cast<size_t>(idx1)] * id : 0.0f;
          int q0 = static_cast<int>(std::round(x0 + 8.0f));
          int q1 = static_cast<int>(std::round(x1 + 8.0f));
          if (q0 < 0) {
            q0 = 0;
          } else if (q0 > 15) {
            q0 = 15;
          }
          if (q1 < 0) {
            q1 = 0;
          } else if (q1 > 15) {
            q1 = 15;
          }
          block.qs[j] = static_cast<uint8_t>(q0 | (q1 << 4));
        }
      }
      blocks.push_back(block);
    }
  }
  return blocks;
}

}  // namespace dqm
