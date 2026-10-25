// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the CUDA implementation.
// clang-format off
#include "dqm_llama/cuda_ops.h"
// clang-format on

#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/cuda_runtime.h"

namespace dqm {

namespace {

constexpr int kBlockSize = 256;

void CheckCudaOps(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA ops error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

void CheckLastKernel(const char* name) {
  CheckCudaOps(cudaGetLastError(), name);
}

__global__ void SumSquaresKernel(const float* x, float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(sum, x[i] * x[i]);
  }
}

__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              const float* sum, int n, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float scale = rsqrtf((*sum / static_cast<float>(n)) + eps);
    y[i] = x[i] * scale * weight[i];
  }
}

__global__ void SiluKernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

__global__ void ElementwiseMulKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * b[i];
  }
}

__global__ void ElementwiseAddKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

__global__ void EmbeddingLookupKernel(const float* embedding, float* y,
                                      int token_id, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    y[i] = embedding[token_id * dim + i];
  }
}

__global__ void SoftmaxMaxKernel(const float* x, float* max_out, int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_max = -3.402823466e+38F;
  for (int i = tid; i < n; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    *max_out = shared[0];
  }
}

__global__ void SoftmaxExpSumKernel(const float* x, float* y,
                                    const float* max_value, float* sum_out,
                                    int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_sum = 0.0f;
  float max_v = *max_value;
  for (int i = tid; i < n; i += blockDim.x) {
    float e = expf(x[i] - max_v);
    y[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *sum_out = shared[0];
  }
}

__global__ void SoftmaxNormKernel(float* y, const float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] /= *sum;
  }
}

__global__ void RopeNormalKernel(float* x, int n_heads, int head_dim, int pos,
                                 float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_head = head_dim / 2;
  int total_pairs = n_heads * pairs_per_head;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / pairs_per_head;
  int pair = pair_index % pairs_per_head;
  int dim = pair * 2;
  int base = head * head_dim + dim;
  float freq = 1.0f / powf(theta, static_cast<float>(dim) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base];
  float x1 = x[base + 1];
  x[base] = x0 * cos_val - x1 * sin_val;
  x[base + 1] = x0 * sin_val + x1 * cos_val;
}

__global__ void RopeNeoXKernel(float* x, int n_heads, int head_dim, int pos,
                               float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int total_pairs = n_heads * half_dim;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / half_dim;
  int pair = pair_index % half_dim;
  int base = head * head_dim;
  float freq = 1.0f / powf(theta, static_cast<float>(2 * pair) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base + pair];
  float x1 = x[base + half_dim + pair];
  x[base + pair] = x0 * cos_val - x1 * sin_val;
  x[base + half_dim + pair] = x0 * sin_val + x1 * cos_val;
}

int GridFor(int n) { return (n + kBlockSize - 1) / kBlockSize; }

void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
  if (a.shape != b.shape) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
}

void RequireSameShape(const CudaTensor& a, const CudaTensor& b,
                      const char* caller) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
  }
  if (a.device_id() != b.device_id()) {
    throw std::runtime_error(std::string(caller) +
                             ": tensors are on different CUDA devices");
  }
}

void Require1D(const Tensor& x, const char* caller) {
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void Require1D(const CudaTensor& x, const char* caller) {
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) +
                             ": expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void ValidateRopeInputs(const Tensor& q, const Tensor& k, int pos,
                        float theta) {
  if (q.num_dims() != 2 || k.num_dims() != 2) {
    throw std::runtime_error("CudaRope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("CudaRope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("CudaRope: theta must be finite and positive");
  }
  if (q.shape[1] != k.shape[1]) {
    throw std::runtime_error("CudaRope: q and k head_dim mismatch " +
                             q.ShapeStringShort() + " vs " +
                             k.ShapeStringShort());
  }
  if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
    throw std::runtime_error("CudaRope: head_dim must be positive and even");
  }
}

void ValidateRopeDeviceInputs(const CudaTensor& q, const CudaTensor& k,
                              int n_heads, int n_kv_heads, int head_dim,
                              int pos, float theta, int device_id) {
  if (q.device_id() != device_id || k.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRopeDeviceInput: input tensor is on a different CUDA device");
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("CudaRopeDeviceInput: invalid head shape");
  }
  if (q.size() !=
      static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("CudaRopeDeviceInput: q shape mismatch");
  }
  if (k.size() !=
      static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("CudaRopeDeviceInput: k shape mismatch");
  }
  if (pos < 0) {
    throw std::out_of_range(
        "CudaRopeDeviceInput: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error(
        "CudaRopeDeviceInput: theta must be finite and positive");
  }
}

void ValidateEmbeddingDeviceWeightInputs(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id) {
  if (embedding_data == nullptr) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: embedding data is null");
  }
  if (embedding_shape.size() != 2) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: expected 2D embedding weight");
  }
  int vocab_size = embedding_shape[0];
  int dim = embedding_shape[1];
  if (vocab_size <= 0 || dim <= 0) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: embedding shape must be positive");
  }
  if (token_id < 0 || token_id >= vocab_size) {
    throw std::out_of_range(
        "CudaEmbeddingLookupDeviceWeight: token id out of range");
  }
}

void ValidateRmsNormDeviceWeightInputs(const CudaTensor& x,
                                       const void* weight_data,
                                       const std::vector<int>& weight_shape,
                                       float eps, int device_id) {
  Require1D(x, "CudaRmsNormDeviceWeight");
  if (weight_data == nullptr) {
    throw std::runtime_error("CudaRmsNormDeviceWeight: weight data is null");
  }
  if (x.shape() != weight_shape) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: x and weight shape mismatch");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: eps must be finite and positive");
  }
}

void ValidateAddDeviceWeightInputs(const CudaTensor& a, const void* b_data,
                                   const std::vector<int>& b_shape,
                                   int device_id) {
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: input tensor is on a different CUDA "
        "device");
  }
  if (b_data == nullptr) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: weight data is null");
  }
  if (a.shape() != b_shape) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: input and weight shape mismatch");
  }
}

}  // namespace

Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id) {
  Require1D(x, "CudaRmsNorm");
  Require1D(weight, "CudaRmsNorm");
  if (x.shape != weight.shape) {
    throw std::runtime_error("CudaRmsNorm: x and weight shape mismatch");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error("CudaRmsNorm: eps must be finite and positive");
  }

  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));
  w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
  CheckCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(RmsNorm sum)");

  int n = static_cast<int>(x.size());
  SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(sum_dev.data()), n);
  CheckLastKernel("SumSquaresKernel");
  RmsNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const float*>(w_dev.data()),
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n, eps);
  CheckLastKernel("RmsNormKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor CudaSilu(const Tensor& x, int device_id) {
  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  SiluKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("SiluKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseMul");
  CudaSetDevice(device_id);
  Tensor y(a.shape, 0.0f);
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a.size() * sizeof(float));
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  ElementwiseMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("ElementwiseMulKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

CudaTensor CudaEmbeddingLookupDeviceWeight(
    const void* embedding_data, const std::vector<int>& embedding_shape,
    int token_id, int device_id) {
  ValidateEmbeddingDeviceWeightInputs(embedding_data, embedding_shape,
                                      token_id);
  CudaSetDevice(device_id);

  int dim = embedding_shape[1];
  CudaTensor y({dim}, device_id);
  EmbeddingLookupKernel<<<GridFor(dim), kBlockSize>>>(
      static_cast<const float*>(embedding_data), static_cast<float*>(y.data()),
      token_id, dim);
  CheckLastKernel("EmbeddingLookupKernel");
  return y;
}

CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id) {
  Require1D(x, "CudaRmsNormDeviceInput");
  Require1D(weight, "CudaRmsNormDeviceInput");
  if (x.shape() != weight.shape) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: x and weight shape mismatch");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps <= 0.0f) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: eps must be finite and positive");
  }

  CudaSetDevice(device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
  return CudaRmsNormDeviceWeight(x, w_dev.data(), weight.shape, eps, device_id);
}

CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape,
                                   float eps, int device_id) {
  ValidateRmsNormDeviceWeightInputs(x, weight_data, weight_shape, eps,
                                    device_id);
  CudaSetDevice(device_id);

  CudaTensor y(x.shape(), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  CheckCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(RmsNorm sum)");

  int n = static_cast<int>(x.size());
  SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<float*>(sum_dev.data()),
      n);
  CheckLastKernel("SumSquaresKernel");
  RmsNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x.data()),
      static_cast<const float*>(weight_data), static_cast<float*>(y.data()),
      static_cast<const float*>(sum_dev.data()), n, eps);
  CheckLastKernel("RmsNormKernel");

  return y;
}

CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaSiluDeviceInput: input tensor is on a different CUDA device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(x.shape(), device_id);

  int n = static_cast<int>(x.size());
  SiluKernel<<<GridFor(n), kBlockSize>>>(static_cast<const float*>(x.data()),
                                         static_cast<float*>(y.data()), n);
  CheckLastKernel("SiluKernel");

  return y;
}

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseMulDeviceInput");
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseMulDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseMulKernel");

  return y;
}

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a,
                                         const CudaTensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseAddDeviceInput");
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  return y;
}

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a,
                                          const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id) {
  ValidateAddDeviceWeightInputs(a, b_data, b_shape, device_id);
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b_data),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  return y;
}

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseAdd");
  CudaSetDevice(device_id);
  Tensor y(a.shape, 0.0f);
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a.size() * sizeof(float));
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()),
      static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads,
                         int n_kv_heads, int head_dim, int pos, float theta,
                         RopeType rope_type, int device_id) {
  ValidateRopeDeviceInputs(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                           device_id);
  CudaSetDevice(device_id);

  int q_pairs = n_heads * (head_dim / 2);
  int k_pairs = n_kv_heads * (head_dim / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNeoXKernel(q)");
    RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNeoXKernel(k)");
  } else {
    RopeNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNormalKernel(q)");
    RopeNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNormalKernel(k)");
  }
}

Tensor CudaSoftmax(const Tensor& x, int device_id) {
  Require1D(x, "CudaSoftmax");
  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer max_dev(sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  SoftmaxMaxKernel<<<1, kBlockSize>>>(static_cast<const float*>(x_dev.data()),
                                      static_cast<float*>(max_dev.data()), n);
  CheckLastKernel("SoftmaxMaxKernel");
  SoftmaxExpSumKernel<<<1, kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(max_dev.data()),
      static_cast<float*>(sum_dev.data()), n);
  CheckLastKernel("SoftmaxExpSumKernel");
  SoftmaxNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n);
  CheckLastKernel("SoftmaxNormKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type,
              int device_id) {
  ValidateRopeInputs(q, k, pos, theta);
  CudaSetDevice(device_id);
  CudaDeviceBuffer q_dev(q.size() * sizeof(float), device_id);
  CudaDeviceBuffer k_dev(k.size() * sizeof(float), device_id);
  q_dev.Upload(q.data.data(), q.size() * sizeof(float));
  k_dev.Upload(k.data.data(), k.size() * sizeof(float));

  int q_pairs = q.shape[0] * (q.shape[1] / 2);
  int k_pairs = k.shape[0] * (k.shape[1] / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    CheckLastKernel("RopeNeoXKernel(q)");
    RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    CheckLastKernel("RopeNeoXKernel(k)");
  } else {
    RopeNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    CheckLastKernel("RopeNormalKernel(q)");
    RopeNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    CheckLastKernel("RopeNormalKernel(k)");
  }

  q_dev.Download(q.data.data(), q.size() * sizeof(float));
  k_dev.Download(k.data.data(), k.size() * sizeof(float));
}

}  // namespace dqm
