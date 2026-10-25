#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/loader.h"
#include "dqm_llama/model.h"
#include "dqm_llama/quantized_tensor.h"

// 源码根目录，由 CMake 编译期传入（见 CMakeLists.txt）。兜底：退回当前目录。
#ifndef DQM_TEST_DATA_DIR
#define DQM_TEST_DATA_DIR "."
#endif

using dqm::ModelManifest;
using dqm::MiniLlamaModel;
using dqm::ModelConfig;
using dqm::QuantType;
using dqm::Tensor;
using dqm::TensorInfo;

namespace {

// ===========================================================================
// 夹具：读 / 改 / 写「真的 model.json」
// ===========================================================================
//
// 参考的做法是：不手搓 JSON，而是**读真的 tiny manifest、在文本上改一处**，
// 再喂给 ParseManifest。这样每个用例只表达"我改了什么"。

std::string TinyManifestPath() {
  return std::string(DQM_TEST_DATA_DIR) + "/models/tiny/model.json";
}

std::string TinyWeightsPath() {
  return std::string(DQM_TEST_DATA_DIR) + "/models/tiny/model.bin";
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// 临时文件：构造时写入，析构时删掉（保证异常路径也不留垃圾）。
class TempFile {
 public:
  TempFile(const std::string& name, const std::string& content)
      : path_(std::filesystem::temp_directory_path() /
              ("dqm_loader_" + name)) {
    std::ofstream out(path_, std::ios::binary);
    out << content;
  }
  ~TempFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

std::string TinyJson() { return ReadTextFile(TinyManifestPath()); }

// 把 from 换一次成 to；找不到返回 false（让用例自己 ASSERT）。
bool ReplaceOnce(std::string& text, const std::string& from,
                 const std::string& to) {
  const size_t pos = text.find(from);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

// 在 marker 之后把 from 换一次成 to（用于"改第 N 个同名片段"）。
bool ReplaceAfter(std::string& text, const std::string& marker,
                  const std::string& from, const std::string& to) {
  const size_t marker_pos = text.find(marker);
  if (marker_pos == std::string::npos) {
    return false;
  }
  const size_t pos = text.find(from, marker_pos);
  if (pos == std::string::npos) {
    return false;
  }
  text.replace(pos, from.size(), to);
  return true;
}

// 写临时文件 -> 期望 ParseManifest 抛 std::runtime_error
bool ManifestFails(const std::string& json, const std::string& name) {
  TempFile tmp(name, json);
  try {
    dqm::ParseManifest(tmp.path());
    return false;
  } catch (const std::runtime_error&) {
    return true;
  }
}

// 把 tiny 权重复制成临时文件；prefix_bytes=0 表示整份，
// append_extra_byte 表示末尾多塞一个字节。返回路径，调用方负责删。
std::string CopyWeights(const std::string& name, size_t prefix_bytes = 0,
                        bool append_extra_byte = false) {
  const std::string src = ReadTextFile(TinyWeightsPath());
  std::string content = prefix_bytes == 0 ? src : src.substr(0, prefix_bytes);
  if (append_extra_byte) {
    content.push_back('\0');
  }
  const std::string path =
      (std::filesystem::temp_directory_path() / ("dqm_loader_" + name)).string();
  std::ofstream out(path, std::ios::binary);
  out << content;
  return path;
}

std::string RemoveTemp(const std::string& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
  return path;
}

}  // namespace

// ===========================================================================
// ParseManifest：成功路径
// ===========================================================================

TEST(Loader, ParsesTinyManifest) {
  const ModelManifest manifest = dqm::ParseManifest(TinyManifestPath());

  EXPECT_EQ(manifest.config.vocab_size, 128);
  EXPECT_EQ(manifest.config.dim, 32);
  EXPECT_EQ(manifest.config.hidden_dim, 86);
  EXPECT_EQ(manifest.config.n_layers, 2);
  EXPECT_EQ(manifest.config.n_heads, 4);
  EXPECT_EQ(manifest.config.n_kv_heads, 4);
  EXPECT_EQ(manifest.config.max_seq_len, 128);
  EXPECT_FLOAT_EQ(manifest.config.rope_theta, 10000.0f);
  EXPECT_FLOAT_EQ(manifest.config.rms_norm_eps, 1e-5f);

  // 3 + 9 * n_layers = 3 + 18
  EXPECT_EQ(manifest.tensors.size(), 21u);
}

// 契约：head_dim 不在 manifest 里，是 dim / n_heads 算出来的
TEST(Loader, DerivesHeadDim) {
  const ModelManifest manifest = dqm::ParseManifest(TinyManifestPath());

  EXPECT_EQ(manifest.config.head_dim, 32 / 4);
}

// 契约：tensors 数组按文件里的顺序保留（不排序）
TEST(Loader, KeepsTensorOrderFromFile) {
  const ModelManifest manifest = dqm::ParseManifest(TinyManifestPath());

  ASSERT_EQ(manifest.tensors.size(), 21u);
  EXPECT_EQ(manifest.tensors.front().name, "token_embedding");
  EXPECT_EQ(manifest.tensors[1].name, "layers.0.attention_norm");
  EXPECT_EQ(manifest.tensors.back().name, "lm_head");
}

TEST(Loader, ParsesTensorOffsetAndByteSize) {
  const ModelManifest manifest = dqm::ParseManifest(TinyManifestPath());

  ASSERT_FALSE(manifest.tensors.empty());
  const TensorInfo& first = manifest.tensors[0];
  EXPECT_EQ(first.name, "token_embedding");
  EXPECT_EQ(first.shape, (std::vector<int>{128, 32}));
  EXPECT_EQ(first.dtype, "float32");
  EXPECT_EQ(first.offset, 0u);
  EXPECT_EQ(first.byte_size, 16384u);
}

TEST(Loader, ParsesTokenizerMetadata) {
  const ModelManifest manifest = dqm::ParseManifest(TinyManifestPath());

  EXPECT_EQ(manifest.tokenizer.type, "json_vocab");
  EXPECT_EQ(manifest.tokenizer.path, "vocab.json");
  EXPECT_EQ(manifest.tokenizer.bos_id, 1);
  EXPECT_EQ(manifest.tokenizer.eos_id, 2);
  EXPECT_EQ(manifest.tokenizer.unk_id, 0);
}

// 契约：tokenizer 块整个缺失 -> 退回默认（ascii + 1/2/0），不报错
TEST(Loader, DefaultsToAsciiWhenTokenizerBlockMissing) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"tokenizer\"", "\"unused_tokenizer\""));

  TempFile tmp("no_tokenizer.json", json);
  const ModelManifest manifest = dqm::ParseManifest(tmp.path());

  EXPECT_EQ(manifest.tokenizer.type, "ascii");
  EXPECT_TRUE(manifest.tokenizer.path.empty());
  EXPECT_EQ(manifest.tokenizer.bos_id, 1);
  EXPECT_EQ(manifest.tokenizer.eos_id, 2);
  EXPECT_EQ(manifest.tokenizer.unk_id, 0);
}

// 契约：rope_type 不从 manifest 解析 —— 就算文件里写了也不认
TEST(Loader, IgnoresRopeTypeInManifest) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"rms_norm_eps\": 1e-05",
                          "\"rms_norm_eps\": 1e-05,\n    \"rope_type\": 99"));

  TempFile tmp("rope_type.json", json);
  const ModelManifest manifest = dqm::ParseManifest(tmp.path());

  EXPECT_EQ(manifest.config.rope_type, dqm::RopeType::kNormal);
}

// ===========================================================================
// ParseManifest：配置校验
// ===========================================================================

TEST(Loader, RejectsFractionalIntegerConfig) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"n_heads\": 4", "\"n_heads\": 4.5"));

  EXPECT_TRUE(ManifestFails(json, "fractional_int.json"));
}

TEST(Loader, RejectsNonfiniteFloatConfig) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"rope_theta\": 10000.0",
                          "\"rope_theta\": NaN"));

  EXPECT_TRUE(ManifestFails(json, "nonfinite_float.json"));
}

// 数字后面跟了非终止符：std::stoi 会"截断着吃"，不能被它骗过
TEST(Loader, RejectsGarbageAfterNumber) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"n_layers\": 2", "\"n_layers\": 2x"));

  EXPECT_TRUE(ManifestFails(json, "garbage_after_number.json"));
}

TEST(Loader, RejectsZeroDimension) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"n_layers\": 2", "\"n_layers\": 0"));

  EXPECT_TRUE(ManifestFails(json, "zero_dimension.json"));
}

TEST(Loader, RejectsMissingRequiredConfigKey) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"vocab_size\": 128",
                          "\"vocab_size_unused\": 128"));

  EXPECT_TRUE(ManifestFails(json, "missing_config_key.json"));
}

TEST(Loader, RejectsDimNotDivisibleByHeads) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"dim\": 32", "\"dim\": 33"));

  EXPECT_TRUE(ManifestFails(json, "dim_divisibility.json"));
}

TEST(Loader, RejectsHeadsNotDivisibleByKvHeads) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"n_kv_heads\": 4", "\"n_kv_heads\": 3"));

  EXPECT_TRUE(ManifestFails(json, "kv_head_divisibility.json"));
}

// dim=36 / n_heads=4 -> head_dim=9，奇数 —— RoPE 要成对，必须是偶数
TEST(Loader, RejectsOddHeadDim) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"dim\": 32", "\"dim\": 36"));

  EXPECT_TRUE(ManifestFails(json, "odd_head_dim.json"));
}

TEST(Loader, RejectsNonPositiveRopeTheta) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"rope_theta\": 10000.0",
                          "\"rope_theta\": 0.0"));

  EXPECT_TRUE(ManifestFails(json, "nonpositive_rope_theta.json"));
}

TEST(Loader, RejectsNonPositiveRmsNormEps) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"rms_norm_eps\": 1e-05",
                          "\"rms_norm_eps\": 0"));

  EXPECT_TRUE(ManifestFails(json, "nonpositive_eps.json"));
}

// ===========================================================================
// ParseManifest：tokenizer 校验
// ===========================================================================

TEST(Loader, RejectsUnsupportedTokenizerType) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"type\": \"json_vocab\"",
                          "\"type\": \"bpe\""));

  EXPECT_TRUE(ManifestFails(json, "bad_tokenizer_type.json"));
}

TEST(Loader, RejectsJsonVocabWithoutPath) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"path\": \"vocab.json\"", "\"path\": \"\""));

  EXPECT_TRUE(ManifestFails(json, "json_vocab_no_path.json"));
}

TEST(Loader, RejectsDuplicateSpecialIds) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"eos_id\": 2", "\"eos_id\": 1"));

  EXPECT_TRUE(ManifestFails(json, "duplicate_special_ids.json"));
}

TEST(Loader, RejectsNegativeSpecialId) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"bos_id\": 1", "\"bos_id\": -1"));

  EXPECT_TRUE(ManifestFails(json, "negative_special_id.json"));
}

// ===========================================================================
// ParseManifest：张量清单校验
// ===========================================================================

TEST(Loader, RejectsDuplicateTensorName) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"name\": \"final_norm\"",
                          "\"name\": \"token_embedding\""));

  EXPECT_TRUE(ManifestFails(json, "duplicate_tensor.json"));
}

TEST(Loader, RejectsMissingDtype) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "      \"dtype\": \"float32\",\n", ""));

  EXPECT_TRUE(ManifestFails(json, "missing_dtype.json"));
}

TEST(Loader, RejectsUnsupportedDtype) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"dtype\": \"float32\"",
                          "\"dtype\": \"float16\""));

  EXPECT_TRUE(ManifestFails(json, "unsupported_dtype.json"));
}

TEST(Loader, RejectsByteSizeMismatch) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceAfter(json, "\"name\": \"token_embedding\"",
                           "\"byte_size\": 16384", "\"byte_size\": 16383"));

  EXPECT_TRUE(ManifestFails(json, "byte_size_mismatch.json"));
}

TEST(Loader, RejectsEmptyShape) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"shape\": [\n        128,\n        32\n      ]",
                          "\"shape\": []"));

  EXPECT_TRUE(ManifestFails(json, "empty_shape.json"));
}

TEST(Loader, RejectsOverlappingTensorRanges) {
  std::string json = TinyJson();
  // lm_head 的 offset 改回 final_norm 那一格，两个区间就叠上了
  ASSERT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "\"offset\": 115840", "\"offset\": 115712"));

  EXPECT_TRUE(ManifestFails(json, "overlapping_ranges.json"));
}

TEST(Loader, RejectsRequiredShapeMismatch) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceAfter(json, "\"name\": \"lm_head\"",
                           "\"shape\": [\n        128,\n        32\n      ]",
                           "\"shape\": [\n        64,\n        64\n      ]"));

  EXPECT_TRUE(ManifestFails(json, "shape_mismatch.json"));
}

TEST(Loader, RejectsMissingRequiredTensor) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"name\": \"final_norm\"",
                          "\"name\": \"final_norm_typo\""));

  EXPECT_TRUE(ManifestFails(json, "missing_required_tensor.json"));
}

// ===========================================================================
// LoadModel
// ===========================================================================

TEST(Loader, LoadsTinyModel) {
  const MiniLlamaModel model =
      dqm::LoadModel(TinyManifestPath(), TinyWeightsPath());

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  EXPECT_EQ(model.config.n_layers, 2);
  EXPECT_EQ(model.layers.size(), 2u);
}

// 端到端：权重总字节 == manifest 最后一格的 offset + byte_size（§11.5 的黄金值）
TEST(Loader, LoadsWeightsWithExpectedTotalBytes) {
  const MiniLlamaModel model =
      dqm::LoadModel(TinyManifestPath(), TinyWeightsPath());
  ASSERT_TRUE(model.loaded) << model.load_error;

  EXPECT_EQ(dqm::ModelWeightBytesF32(model), 132224u);
}

// 契约：attention_norm / ffn_norm 是 Tensor；会做矩阵乘的走 QuantizedTensor
TEST(Loader, LoadsWeightsWithExpectedContainerTypes) {
  const MiniLlamaModel model =
      dqm::LoadModel(TinyManifestPath(), TinyWeightsPath());
  ASSERT_TRUE(model.loaded) << model.load_error;

  EXPECT_EQ(model.token_embedding.data.size(), 128u * 32u);
  EXPECT_EQ(model.final_norm.data.size(), 32u);

  ASSERT_EQ(model.layers.size(), 2u);
  const dqm::LayerWeights& layer0 = model.layers[0];
  EXPECT_EQ(layer0.attention_norm.data.size(), 32u);
  EXPECT_EQ(layer0.wq.shape, (std::vector<int>{32, 32}));
  EXPECT_EQ(layer0.wq.type, QuantType::kF32);
  EXPECT_EQ(layer0.w_gate.shape, (std::vector<int>{86, 32}));
  EXPECT_EQ(model.lm_head.type, QuantType::kF32);
  EXPECT_EQ(model.lm_head.shape, (std::vector<int>{128, 32}));
}

// 契约：LoadModel 不抛 —— 权重文件不存在 -> loaded=false + 原因
TEST(Loader, FailsOnMissingWeightsFileWithoutThrowing) {
  const MiniLlamaModel model = dqm::LoadModel(
      TinyManifestPath(), "/definitely/not/here/model.bin");

  EXPECT_FALSE(model.loaded);
  EXPECT_NE(model.load_error.find("failed to open weights"), std::string::npos);
}

// 契约：LoadModel 不抛 —— manifest 坏掉也走同一条路
TEST(Loader, FailsOnBadManifestWithoutThrowing) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"n_layers\": 2", "\"n_layers\": 0"));
  TempFile bad("bad_manifest.json", json);

  const MiniLlamaModel model = dqm::LoadModel(bad.path(), TinyWeightsPath());

  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

TEST(Loader, FailsOnTrailingWeightBytes) {
  const std::string path =
      CopyWeights("trailing.bin", 0, /*append_extra_byte=*/true);
  const MiniLlamaModel model = dqm::LoadModel(TinyManifestPath(), path);
  RemoveTemp(path);

  EXPECT_FALSE(model.loaded);
  EXPECT_NE(model.load_error.find("trailing bytes"), std::string::npos);
}

TEST(Loader, FailsOnShortWeightsFile) {
  const std::string path = CopyWeights("short.bin", 100000);
  const MiniLlamaModel model = dqm::LoadModel(TinyManifestPath(), path);
  RemoveTemp(path);

  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

// ===========================================================================
// InspectModel
// ===========================================================================

TEST(Loader, InspectSucceedsOnTinyManifest) {
  EXPECT_TRUE(dqm::InspectModel(TinyManifestPath()));
}

TEST(Loader, InspectFailsOnBadManifest) {
  std::string json = TinyJson();
  ASSERT_TRUE(ReplaceOnce(json, "\"n_layers\": 2", "\"n_layers\": 0"));
  TempFile bad("inspect_bad.json", json);

  EXPECT_FALSE(dqm::InspectModel(bad.path()));
}
