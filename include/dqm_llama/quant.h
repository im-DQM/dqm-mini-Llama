#pragma once

#include <vector>

#include "dqm_llama/quantized_tensor.h"

namespace dqm {

// 量化入口：F32 Tensor -> 块量化数据（Q8_0 / Q4_0）。
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src);
std::vector<BlockQ40> QuantizeToQ40(const Tensor& src);

// 反量化入口：块量化数据 -> F32 Tensor。
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape);
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape);
Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape);

// ---------------------------------------------------------------------------
// 真·量化 Linear（块级 on-the-fly）：权重以量化块保存，输入/输出是 F32，
// 不分配临时 F32 权重。weight_shape = [out_features, in_features]。
// ---------------------------------------------------------------------------
Tensor LinearQ80(const Tensor& x, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape);
Tensor LinearQ40(const Tensor& x, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape);
Tensor LinearQ41(const Tensor& x, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape);

}  // namespace dqm
