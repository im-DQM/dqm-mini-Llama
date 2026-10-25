#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dqm_llama/tokenizer.h"

namespace dqm {

// ---------------------------------------------------------------------------
// GgufTokenizer：直接从 GGUF metadata 读词表，实现 GPT-2 风格 byte-level BPE。
//
// 读的键：
//   tokenizer.ggml.tokens       （词表 pieces）
//   tokenizer.ggml.merges       （BPE 合并规则）
//   tokenizer.ggml.token_type   （识别 control/special token）
//   tokenizer.ggml.bos_token_id / eos_token_id / unknown_token_id
//
// 只依赖 gguf.h 的 GgufGetMetadata* 三个函数，不建 GgufReader、不读张量字节。
// ---------------------------------------------------------------------------
class GgufTokenizer : public ITokenizer {
 public:
  GgufTokenizer() = default;
  ~GgufTokenizer() override = default;

  // 从 GGUF 文件加载词表。缺 tokenizer.ggml.tokens 时返回 false。
  bool Load(const std::string& gguf_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override {
    return static_cast<int>(id_to_token_.size());
  }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  std::unordered_map<std::string, int> vocab_;                      // piece -> id
  std::vector<std::string> id_to_token_;                            // id -> piece
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;  // 相邻对 -> rank
  std::vector<std::pair<std::string, int>> special_tokens_;         // 按长度降序

  std::vector<std::string> b2u_;                  // byte -> unicode 替身串
  std::unordered_map<std::string, uint8_t> u2b_;  // unicode 替身串 -> byte

  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;

  void BuildByteMappings();
};

// 从 GGUF 文件建 tokenizer；无 tokenizer metadata 时返回 nullptr。
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}  // namespace dqm
