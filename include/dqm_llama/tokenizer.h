#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dqm {

// ---------------------------------------------------------------------------
// Tokenizer 接口：四个 tokenizer 的公共契约
// ---------------------------------------------------------------------------
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // 文本 -> token ids（是否含 BOS 由各实现自定）
  virtual std::vector<int> Encode(const std::string& text) const = 0;

  // 单个 token id -> 字符串
  virtual std::string DecodeToken(int token) const = 0;

  // id 序列 -> 字符串
  virtual std::string Decode(const std::vector<int>& tokens) const = 0;

  virtual int vocab_size() const = 0;
  virtual int bos_id() const = 0;
  virtual int eos_id() const = 0;
  virtual int unk_id() const = 0;
};

// ---------------------------------------------------------------------------
// AsciiTokenizer：0..127 字符级映射（玩具实现）
// ---------------------------------------------------------------------------
class AsciiTokenizer : public ITokenizer {
 public:
  static constexpr int kVocabSize = 128;

  AsciiTokenizer() = default;

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return kVocabSize; }
  int bos_id() const override { return 1; }
  int eos_id() const override { return 2; }
  int unk_id() const override { return 0; }
};

// ---------------------------------------------------------------------------
// JsonVocabTokenizer：从 JSON 词表文件加载
//   格式：[{"id": 0, "content": "<unk>", "special": true}, ...]
//   构造失败抛 std::runtime_error。
// ---------------------------------------------------------------------------
class JsonVocabTokenizer : public ITokenizer {
 public:
  explicit JsonVocabTokenizer(const std::string& vocab_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return vocab_size_; }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  struct VocabEntry {
    int id = 0;
    std::string content;
    bool special = false;
  };

  int vocab_size_ = 0;
  int bos_id_ = 1;
  int eos_id_ = 2;
  int unk_id_ = 0;

  // id -> 条目
  std::vector<VocabEntry> id_to_entry_;

  // content -> id（含长度，Encode 时按长度降序试最长匹配）
  std::vector<std::pair<std::string, int>> content_to_id_;
};

// ---------------------------------------------------------------------------
// BpeTokenizer：GPT-2 风格 BPE
//   需要 vocab.json + merges.txt + special_tokens.json，由 Load() 加载。
// ---------------------------------------------------------------------------
class BpeTokenizer : public ITokenizer {
 public:
  BpeTokenizer() = default;
  ~BpeTokenizer() override = default;

  bool Load(const std::string& vocab_path, const std::string& merges_path,
            const std::string& special_path);

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
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;

  // special token（如 <|im_start|>），按长度降序
  std::vector<std::pair<std::string, int>> special_tokens_;

  std::vector<std::string> b2u_;                  // byte -> unicode 串
  std::unordered_map<std::string, uint8_t> u2b_;  // unicode 串 -> byte

  void BuildByteMappings();
};

// ---------------------------------------------------------------------------
// 工厂
// ---------------------------------------------------------------------------
// vocab_path 非空且文件可开 -> JsonVocabTokenizer；否则/失败回落 AsciiTokenizer。
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

// 从三份文件建 BPE；Load 失败返回 nullptr。
std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path);

}  // namespace dqm
