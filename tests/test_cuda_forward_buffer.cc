// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "dqm_llama/batch.h"
#include "dqm_llama/context.h"
#include "dqm_llama/cuda_runtime.h"
#include "dqm_llama/forward.h"
#include "dqm_llama/loader.h"
#include "dqm_llama/model.h"
#include "dqm_llama/quant.h"

#include <gtest/gtest.h>

using namespace dqm;

namespace {

constexpr float kAbsTol = 2e-3f;
constexpr float kRelTol = 2e-3f;

void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool CloseEnough(float actual, float expected) {
  float abs_err = std::abs(actual - expected);
  float scale = std::max(1.0f, std::abs(expected));
  return abs_err <= kAbsTol || abs_err / scale <= kRelTol;
}

void RequireCloseTensor(const Tensor& actual, const Tensor& expected,
                        const std::string& label) {
  Require(actual.shape == expected.shape, label + ": shape mismatch");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!CloseEnough(actual.data[i], expected.data[i])) {
      throw std::runtime_error(
          label + ": value mismatch at " + std::to_string(i) +
          ", actual=" + std::to_string(actual.data[i]) +
          ", expected=" + std::to_string(expected.data[i]));
    }
  }
}

[[maybe_unused]] void RequireCpuBuildPath() {
  Require(!CudaRuntimeBuilt(), "CudaRuntimeBuilt should be false in CPU build");
  MiniLlamaModel model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(model.loaded, "tiny model should load");
  Require(ModelCudaHostToDeviceCopies(model) == 0,
          "CPU model should report zero host->device copies");
}

[[maybe_unused]] void RunTinyForwardBufferCase(bool quantize_q8_0) {
  Require(CudaRuntimeBuilt(), "CudaRuntimeBuilt should be true in CUDA build");
  Require(CudaDeviceCount() > 0, "CUDA build should see at least one device");

  MiniLlamaModel cpu_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  MiniLlamaModel cuda_model =
      LoadModel("models/tiny/model.json", "models/tiny/model.bin");
  Require(cpu_model.loaded, "CPU tiny model should load");
  Require(cuda_model.loaded, "CUDA tiny model should load");

  if (quantize_q8_0) {
    QuantizeModelToQ80(cpu_model);
    QuantizeModelToQ80(cuda_model);
  }

  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext cuda_ctx(&cuda_model);
  MiniBatch batch = MiniBatch::Single(1, 0);
  Tensor cpu_logits = ForwardBatch(cpu_ctx, cpu_model, batch);

  UploadModelWeightsToCuda(cuda_model, 0);
  ResetModelCudaRuntimeStats(cuda_model);
  Tensor cuda_logits = ForwardBatch(cuda_ctx, cuda_model, batch);

  RequireCloseTensor(cuda_logits, cpu_logits,
                     quantize_q8_0 ? "cuda_forward_buffer q8 logits"
                                   : "cuda_forward_buffer f32 logits");
  Require(ModelCudaLinearCalls(cuda_model) == 15,
          "one tiny token should still run 15 CUDA Linear calls");
  Require(ModelCudaActivationCalls(cuda_model) == 15,
          "one tiny token should run 15 CUDA activation calls");
  Require(ModelCudaAttentionCalls(cuda_model) == 2,
          "one tiny token should run 2 CUDA attention calls");
  Require(ModelCudaHostToDeviceCopies(cuda_model) == 0,
          "M14 path should keep host->device copies at zero");
  Require(ModelCudaDeviceToHostCopies(cuda_model) == 1,
          "M14 path should download only logits");
  Require(ModelCudaHostToDeviceBytes(cuda_model) == 0,
          "tiny M14 host->device bytes should stay zero");
  Require(ModelCudaDeviceToHostBytes(cuda_model) == 512,
          "tiny M14 device->host bytes should match logits");
}

}  // namespace

#ifdef DQM_LLAMA_USE_CUDA
TEST(CudaForwardBuffer, F32) { RunTinyForwardBufferCase(false); }
TEST(CudaForwardBuffer, DeviceResidentQ80) { RunTinyForwardBufferCase(true); }
#else
TEST(CudaForwardBuffer, CpuBuildPath) { RequireCpuBuildPath(); }
#endif
