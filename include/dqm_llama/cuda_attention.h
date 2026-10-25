// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_DQM_LLAMA_CUDA_ATTENTION_H_
#define INCLUDE_DQM_LLAMA_CUDA_ATTENTION_H_

#include "dqm_llama/cuda_kv_cache.h"
#include "dqm_llama/cuda_tensor.h"
#include "dqm_llama/tensor.h"

namespace dqm {

bool CudaAttentionBuilt();

Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache,
                           int layer, int pos, int n_heads, int n_kv_heads,
                           int head_dim, int device_id = 0);

CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache,
                                          int layer, int pos, int n_heads,
                                          int n_kv_heads, int head_dim,
                                          int device_id = 0);

}  // namespace dqm

#endif  // INCLUDE_DQM_LLAMA_CUDA_ATTENTION_H_
