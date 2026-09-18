#pragma once

#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

namespace dqm {

// 矩阵乘 / 线性层的实现策略。
// 本轮只实现 kNaive；其余三档为预留槽位（调用即抛 std::logic_error）。
enum class MatmulMode {
  kNaive,         // 三重循环
  kThreaded,      // 预留：多线程
  kSimd,          // 预留：向量指令
  kThreadedSimd,  // 预留：多线程 + 向量指令
};

// 主推理路径默认使用的档。本轮返回 kNaive。
MatmulMode DefaultMatmulMode();

// c = a * b
// a: [m, k], b: [k, n] -> c: [m, n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// y = x * weight^T
// x: [in_features] 或 [1, in_features], weight: [out_features, in_features]
// 结果秩跟随 x。
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

// 同上，但权重是 QuantizedTensor 容器。
// 本轮只接受 type == kF32 —— 直接读 f32_data，不复制。
Tensor LinearDispatch(const Tensor& x, const QuantizedTensor& weight,
                      MatmulMode mode);

}  // namespace dqm
