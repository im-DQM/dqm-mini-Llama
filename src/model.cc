#include "dqm_llama/model.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "dqm_llama/quant.h"

namespace dqm {

namespace {

size_t TensorBytes(const Tensor& t) {
  return t.data.size() * sizeof(float);
}

size_t QuantizedTensorBytes(const QuantizedTensor& qt) {
  switch (qt.type) {
    case QuantType::kF32:
      return qt.f32_data.size() * sizeof(float);
    case QuantType::kQ80:
      return qt.q8_0_data.size() * sizeof(BlockQ80);
    case QuantType::kQ40:
      return qt.q4_0_data.size() * sizeof(BlockQ40);
    case QuantType::kQ41:
      return qt.q4_1_data.size() * sizeof(BlockQ41);
  }
  return 0;
}

size_t QuantizedTensorBytesF32(const QuantizedTensor& qt) {
  return qt.num_elements() * sizeof(float);
}

void UploadF32LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
  if (weight.type != QuantType::kF32) {
    return;
  }
  if (weight.f32_data.empty()) {
    throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
  }

  const size_t expected_numel = weight.num_elements();
  if (weight.f32_data.size() != expected_numel) {
    throw std::runtime_error("CUDA weight upload: data size mismatch for " +
                             name + ", expected " +
                             std::to_string(expected_numel) + ", got " +
                             std::to_string(weight.f32_data.size()));
  }

  const size_t bytes = weight.f32_data.size() * sizeof(float);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kF32;
  uploaded.linear = true;
  uploaded.shape = weight.shape;
  uploaded.block_count = 0;
  uploaded.buffer.Reset(bytes, device_id);
  uploaded.buffer.Upload(weight.f32_data.data(), bytes);

  storage.uploaded_weight_count += 1;
  storage.uploaded_bytes += bytes;
  storage.weights.push_back(std::move(uploaded));
}

void UploadF32TensorWeight(CudaModelWeights& storage, const Tensor& weight,
                           const std::string& name, int device_id) {
  if (weight.data.empty()) {
    return;
  }
  const size_t expected_numel = weight.size();
  if (weight.data.size() != expected_numel) {
    throw std::runtime_error(
        "CUDA weight upload: tensor data size mismatch for " + name +
        ", expected " + std::to_string(expected_numel) + ", got " +
        std::to_string(weight.data.size()));
  }

  const size_t bytes = weight.data.size() * sizeof(float);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kF32;
  uploaded.linear = false;
  uploaded.shape = weight.shape;
  uploaded.block_count = 0;
  uploaded.buffer.Reset(bytes, device_id);
  uploaded.buffer.Upload(weight.data.data(), bytes);

  storage.uploaded_weight_count += 1;
  storage.uploaded_bytes += bytes;
  storage.weights.push_back(std::move(uploaded));
}

template <typename Block, typename DataVector>
void UploadQuantizedLinearWeight(CudaModelWeights& storage,
                                 const DataVector& blocks,
                                 const std::vector<int>& shape,
                                 const std::string& name, QuantType type,
                                 int block_size, int device_id) {
  if (shape.size() != 2) {
    throw std::runtime_error(
        "CUDA weight upload: expected 2D quantized weight for " + name);
  }
  if (blocks.empty()) {
    throw std::runtime_error("CUDA weight upload: empty quantized data for " +
                             name);
  }

  const int out_features = shape[0];
  const int in_features = shape[1];
  const int blocks_per_row = (in_features + block_size - 1) / block_size;
  const size_t expected_blocks =
      static_cast<size_t>(out_features) * blocks_per_row;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(
        "CUDA weight upload: block count mismatch for " + name + ", expected " +
        std::to_string(expected_blocks) + ", got " +
        std::to_string(blocks.size()));
  }

  const size_t bytes = blocks.size() * sizeof(Block);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = type;
  uploaded.linear = true;
  uploaded.shape = shape;
  uploaded.block_count = blocks.size();
  uploaded.buffer.Reset(bytes, device_id);
  uploaded.buffer.Upload(blocks.data(), bytes);

  storage.uploaded_weight_count += 1;
  storage.uploaded_bytes += bytes;
  storage.weights.push_back(std::move(uploaded));
}

void UploadQ80LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
  if (weight.type != QuantType::kQ80) {
    return;
  }
  UploadQuantizedLinearWeight<BlockQ80>(storage, weight.q8_0_data, weight.shape,
                                        name, QuantType::kQ80, kQ80BlockSize,
                                        device_id);
}

void UploadQ40LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
  if (weight.type != QuantType::kQ40) {
    return;
  }
  UploadQuantizedLinearWeight<BlockQ40>(storage, weight.q4_0_data, weight.shape,
                                        name, QuantType::kQ40, kQ40BlockSize,
                                        device_id);
}

void UploadQ41LinearWeight(CudaModelWeights& storage,
                           const QuantizedTensor& weight,
                           const std::string& name, int device_id) {
  if (weight.type != QuantType::kQ41) {
    return;
  }
  UploadQuantizedLinearWeight<BlockQ41>(storage, weight.q4_1_data, weight.shape,
                                        name, QuantType::kQ41, kQ41BlockSize,
                                        device_id);
}

void UploadLinearWeight(CudaModelWeights& storage,
                        const QuantizedTensor& weight, const std::string& name,
                        int device_id) {
  UploadF32LinearWeight(storage, weight, name, device_id);
  UploadQ80LinearWeight(storage, weight, name, device_id);
  UploadQ40LinearWeight(storage, weight, name, device_id);
  UploadQ41LinearWeight(storage, weight, name, device_id);
}

}  // namespace

size_t ModelWeightBytes(const MiniLlamaModel& model) {
  size_t bytes = 0;
  bytes += TensorBytes(model.token_embedding);
  bytes += TensorBytes(model.final_norm);
  bytes += QuantizedTensorBytes(model.lm_head);
  for (const auto& lw : model.layers) {
    bytes += TensorBytes(lw.attention_norm);
    bytes += QuantizedTensorBytes(lw.wq);
    bytes += QuantizedTensorBytes(lw.wk);
    bytes += QuantizedTensorBytes(lw.wv);
    bytes += TensorBytes(lw.bq);
    bytes += TensorBytes(lw.bk);
    bytes += TensorBytes(lw.bv);
    bytes += QuantizedTensorBytes(lw.wo);
    bytes += TensorBytes(lw.ffn_norm);
    bytes += QuantizedTensorBytes(lw.w_gate);
    bytes += QuantizedTensorBytes(lw.w_up);
    bytes += QuantizedTensorBytes(lw.w_down);
  }
  return bytes;
}

size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
  size_t bytes = 0;
  bytes += TensorBytes(model.token_embedding);
  bytes += TensorBytes(model.final_norm);
  bytes += QuantizedTensorBytesF32(model.lm_head);
  for (const auto& lw : model.layers) {
    bytes += TensorBytes(lw.attention_norm);
    bytes += QuantizedTensorBytesF32(lw.wq);
    bytes += QuantizedTensorBytesF32(lw.wk);
    bytes += QuantizedTensorBytesF32(lw.wv);
    bytes += TensorBytes(lw.bq);
    bytes += TensorBytes(lw.bk);
    bytes += TensorBytes(lw.bv);
    bytes += QuantizedTensorBytesF32(lw.wo);
    bytes += TensorBytes(lw.ffn_norm);
    bytes += QuantizedTensorBytesF32(lw.w_gate);
    bytes += QuantizedTensorBytesF32(lw.w_up);
    bytes += QuantizedTensorBytesF32(lw.w_down);
  }
  return bytes;
}

void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id) {
  if (!model.loaded) {
    throw std::runtime_error("CUDA weight upload requires a loaded model");
  }

  CudaSetDevice(device_id);
  auto storage = std::make_shared<CudaModelWeights>();
  storage->device_id = device_id;

  UploadF32TensorWeight(*storage, model.token_embedding, "token_embedding",
                        device_id);
  UploadF32TensorWeight(*storage, model.final_norm, "final_norm", device_id);
  UploadLinearWeight(*storage, model.lm_head, "lm_head", device_id);
  for (size_t layer = 0; layer < model.layers.size(); ++layer) {
    const LayerWeights& lw = model.layers[layer];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    UploadF32TensorWeight(*storage, lw.attention_norm,
                          prefix + "attention_norm", device_id);
    UploadLinearWeight(*storage, lw.wq, prefix + "wq", device_id);
    UploadLinearWeight(*storage, lw.wk, prefix + "wk", device_id);
    UploadLinearWeight(*storage, lw.wv, prefix + "wv", device_id);
    UploadF32TensorWeight(*storage, lw.bq, prefix + "bq", device_id);
    UploadF32TensorWeight(*storage, lw.bk, prefix + "bk", device_id);
    UploadF32TensorWeight(*storage, lw.bv, prefix + "bv", device_id);
    UploadLinearWeight(*storage, lw.wo, prefix + "wo", device_id);
    UploadF32TensorWeight(*storage, lw.ffn_norm, prefix + "ffn_norm",
                          device_id);
    UploadLinearWeight(*storage, lw.w_gate, prefix + "w_gate", device_id);
    UploadLinearWeight(*storage, lw.w_up, prefix + "w_up", device_id);
    UploadLinearWeight(*storage, lw.w_down, prefix + "w_down", device_id);
  }

  model.cuda_weights = std::move(storage);
}

void ClearModelCudaWeights(MiniLlamaModel& model) {
  model.cuda_weights.reset();
}

bool ModelHasCudaWeights(const MiniLlamaModel& model) {
  return model.cuda_weights != nullptr;
}

size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return 0;
  }
  return model.cuda_weights->uploaded_weight_count;
}

namespace {

size_t ModelCudaUploadedWeightCountByType(const MiniLlamaModel& model,
                                          QuantType type) {
  if (!model.cuda_weights) {
    return 0;
  }
  size_t count = 0;
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.linear && weight.type == type) {
      count += 1;
    }
  }
  return count;
}

}  // namespace

size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kF32);
}

size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kQ80);
}

size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kQ40);
}

size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kQ41);
}

size_t ModelCudaMemoryBytes(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return 0;
  }
  return model.cuda_weights->uploaded_bytes;
}

void ResetModelCudaRuntimeStats(MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls = 0;
  model.cuda_weights->activation_calls = 0;
  model.cuda_weights->attention_calls = 0;
  model.cuda_weights->attention_cpu_fallbacks = 0;
  model.cuda_weights->kv_cache_write_bytes = 0;
  model.cuda_weights->kv_cache_read_bytes = 0;
  model.cuda_weights->host_to_device_copies = 0;
  model.cuda_weights->device_to_host_copies = 0;
  model.cuda_weights->host_to_device_bytes = 0;
  model.cuda_weights->device_to_host_bytes = 0;
}

size_t ModelCudaLinearCalls(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->linear_calls : 0;
}

size_t ModelCudaActivationCalls(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->activation_calls : 0;
}

size_t ModelCudaAttentionCalls(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->attention_calls : 0;
}

size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->attention_cpu_fallbacks : 0;
}

size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->kv_cache_write_bytes : 0;
}

size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->kv_cache_read_bytes : 0;
}

size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->host_to_device_copies : 0;
}

size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_to_host_copies : 0;
}

size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->host_to_device_bytes : 0;
}

size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_to_host_bytes : 0;
}

namespace {

void QuantizeQtToQ80(QuantizedTensor& qt) {
  if (qt.type == QuantType::kQ80) {
    return;
  }
  Tensor t = ToTensor(qt);
  qt.q8_0_data = QuantizeToQ80(t);
  qt.type = QuantType::kQ80;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q4_0_data.clear();
  qt.q4_0_data.shrink_to_fit();
  qt.q4_1_data.clear();
  qt.q4_1_data.shrink_to_fit();
}

void QuantizeQtToQ40(QuantizedTensor& qt) {
  if (qt.type == QuantType::kQ40) {
    return;
  }
  Tensor t = ToTensor(qt);
  qt.q4_0_data = QuantizeToQ40(t);
  qt.type = QuantType::kQ40;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q8_0_data.clear();
  qt.q8_0_data.shrink_to_fit();
  qt.q4_1_data.clear();
  qt.q4_1_data.shrink_to_fit();
}

}  // namespace

void QuantizeModelToQ80(MiniLlamaModel& model) {
  QuantizeQtToQ80(model.lm_head);
  for (auto& lw : model.layers) {
    QuantizeQtToQ80(lw.wq);
    QuantizeQtToQ80(lw.wk);
    QuantizeQtToQ80(lw.wv);
    QuantizeQtToQ80(lw.wo);
    QuantizeQtToQ80(lw.w_gate);
    QuantizeQtToQ80(lw.w_up);
    QuantizeQtToQ80(lw.w_down);
  }
}

void QuantizeModelToQ40(MiniLlamaModel& model) {
  QuantizeQtToQ40(model.lm_head);
  for (auto& lw : model.layers) {
    QuantizeQtToQ40(lw.wq);
    QuantizeQtToQ40(lw.wk);
    QuantizeQtToQ40(lw.wv);
    QuantizeQtToQ40(lw.wo);
    QuantizeQtToQ40(lw.w_gate);
    QuantizeQtToQ40(lw.w_up);
    QuantizeQtToQ40(lw.w_down);
  }
}

}  // namespace dqm
