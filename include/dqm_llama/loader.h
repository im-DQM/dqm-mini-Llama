#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "dqm_llama/model.h"

namespace dqm {

// 一个张量在 model.json 里的「身份证」。
// LoadTensorByName 按 name 查它，按 offset / byte_size 去 .bin 里抠字节。
struct TensorInfo {
  std::string name;        // 如 "layers.0.wq"
  std::vector<int> shape;  // 如 {32, 32}
  std::string dtype;       // 只认 "float32"
  size_t offset = 0;       // 在 .bin 里的字节偏移
  size_t byte_size = 0;    // 占多少字节
};

// 词表的「联系方式」。只装元数据，不装词表本身。
struct TokenizerInfo {
  std::string type = "ascii";  // 只认 "ascii" / "json_vocab"
  std::string path;            // 相对 model.json 所在目录
  int bos_id = 1;
  int eos_id = 2;
  int unk_id = 0;
};

// model.json 的完整投影：config + 词表元数据 + 全部张量的元数据。
struct ModelManifest {
  ModelConfig config;
  TokenizerInfo tokenizer;
  std::vector<TensorInfo> tensors;
};

// 读并校验 model.json。文件打不开 / 键缺失 / 值坏 / 校验不过 -> 抛 std::runtime_error。
ModelManifest ParseManifest(const std::string& config_path);

// 解析 + 读权重。失败**不抛** —— 返回 loaded=false 的模型，原因在 load_error 里。
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);

// 解析 + 把元数据打到 stdout。**不读权重**；失败返回 false。
bool InspectModel(const std::string& config_path);

}  // namespace dqm
