#pragma once

#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"

namespace dqm {

// 矩阵乘：c = a * b
// a: [m, k], b: [k, n] -> c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// 线性投影：y = x * weight^T
// x: [in_features] 或 [1, in_features]
// weight: [out_features, in_features]
// 结果秩跟随 x：[out_features] 或 [1, out_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// 线性投影，权重是 QuantizedTensor 容器。
// 本轮只走 kF32 状态 —— 零拷贝读 f32_data，不经过 ToTensor。
Tensor Linear(const Tensor& x, const QuantizedTensor& weight);

// RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim], weight: [dim], 结果: [dim]
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// 1D 张量上的 Softmax
// x: [n] -> [n]
Tensor Softmax(const Tensor& x);

// SiLU 激活：Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

// 逐元素乘法
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU：SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE（旋转位置编码），原地作用于 q 和 k
// q: [n_heads, head_dim], k: [n_kv_heads, head_dim]
// pos: 当前 token 位置
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// ArgMax：返回扁平最大值的下标
int ArgMax(const Tensor& x);
    
}  // namespace dqm 
