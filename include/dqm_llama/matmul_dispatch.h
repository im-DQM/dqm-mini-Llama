#pragma once

#include <string>

#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

namespace dqm {

// 矩阵乘 / 线性层的实现策略。
// 四档均已接线：kNaive（三重循环）/ kThreaded（多线程）/ kSimd（向量指令）/
// kThreadedSimd（多线程 + 向量指令）。注意：通用 Matmul 的 kSimd/kThreadedSimd
// 会退回多线程（b 按列访问不连续）；只有 Linear 真正吃到 SIMD。
enum class MatmulMode {
  kNaive,         // 三重循环
  kThreaded,      // 多线程
  kSimd,          // 向量指令
  kThreadedSimd,  // 多线程 + 向量指令
};

// 主推理路径默认使用的档。未显式设置时为 kThreadedSimd。
MatmulMode DefaultMatmulMode();

// 覆盖主推理路径的默认档（供 CLI / 测试设置）。
void SetDefaultMatmulMode(MatmulMode mode);

// 策略名 -> 枚举："naive" / "threaded" / "simd" / "threaded-simd"。
// 合法返回 true 并写入 mode；否则返回 false 且不改动 mode。
bool ParseMatmulMode(const std::string& text, MatmulMode& mode);

// 枚举 -> 策略名（与 ParseMatmulMode 接受的名字一致）。
std::string MatmulModeName(MatmulMode mode);

// c = a * b
// a: [m, k], b: [k, n] -> c: [m, n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// y = x * weight^T
// x: [in_features] 或 [1, in_features], weight: [out_features, in_features]
// 结果秩跟随 x。
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

// 同上，但权重是 QuantizedTensor 容器。
// mode 只影响 kF32 那一路；量化态（kQ80/kQ40/kQ41）按 type 走各自的块级内核。
Tensor LinearDispatch(const Tensor& x, const QuantizedTensor& weight,
                      MatmulMode mode);

}  // namespace dqm
