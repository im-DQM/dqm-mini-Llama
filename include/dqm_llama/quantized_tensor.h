#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/tensor.h"

namespace dqm {

// ---------------------------------------------------------------------------
// 量化类型
// ---------------------------------------------------------------------------
enum class QuantType {
  kF32,
  kQ80,
  kQ40,
  kQ41,
};

// ---------------------------------------------------------------------------
// Q8_0 block：每块 32 个元素
// ---------------------------------------------------------------------------
constexpr int kQ80BlockSize = 32;

struct BlockQ80 {
  uint16_t d;   // FP16 缩放因子
  int8_t qs[kQ80BlockSize];
};

// ---------------------------------------------------------------------------
// Q4_0 block：每块 32 个元素，打包进 16 字节
// ---------------------------------------------------------------------------
constexpr int kQ40BlockSize = 32;

struct BlockQ40 {
  uint16_t d;
  uint8_t qs[kQ40BlockSize / 2];
};

// ---------------------------------------------------------------------------
// Q4_1 block：比 Q4_0 多一个最小值 m
// ---------------------------------------------------------------------------
constexpr int kQ41BlockSize = 32;

struct BlockQ41 {
  uint16_t d;
  uint16_t m;
  uint8_t qs[kQ41BlockSize / 2];
};

// ---------------------------------------------------------------------------
// QuantizedTensor：带 tag 的容器
//   按 type 决定四个 data vector 里哪一个有效
// ---------------------------------------------------------------------------
struct QuantizedTensor {
  QuantType type = QuantType::kF32;
  std::vector<int> shape;

  std::vector<float> f32_data;       // type == kF32
  std::vector<BlockQ80> q8_0_data;   // type == kQ80
  std::vector<BlockQ40> q4_0_data;   // type == kQ40
  std::vector<BlockQ41> q4_1_data;   // type == kQ41

  size_t num_elements() const;
  std::string ShapeStringShort() const;
  void AssertShape(const std::vector<int>& expected, const char* caller) const;
  int num_dims() const { return static_cast<int>(shape.size()); }
};

// kF32 容器 -> 普通 Tensor
Tensor ToTensor(const QuantizedTensor& q);

// 普通 Tensor -> kF32 容器
QuantizedTensor ToQuantizedTensor(const Tensor& t);

}  // namespace dqm
