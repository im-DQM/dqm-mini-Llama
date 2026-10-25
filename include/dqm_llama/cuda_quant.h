// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_DQM_LLAMA_CUDA_QUANT_H_
#define INCLUDE_DQM_LLAMA_CUDA_QUANT_H_

#include <vector>

#include "dqm_llama/cuda_tensor.h"
#include "dqm_llama/quantized_tensor.h"
#include "dqm_llama/tensor.h"

namespace dqm {

bool CudaQuantBuilt();

Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x,
                                    const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr,
                                 int device_id = 0);

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x,
                                    const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

}  // namespace dqm

#endif  // INCLUDE_DQM_LLAMA_CUDA_QUANT_H_
