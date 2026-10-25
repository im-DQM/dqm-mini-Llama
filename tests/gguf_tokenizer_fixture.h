#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dqm_llama/gguf.h"

namespace dqm::test {

// 临时文件路径（写盘后由调用方 std::filesystem::remove）。
inline std::filesystem::path GgufTokenizerTempPath(const std::string& name) {
  return std::filesystem::temp_directory_path() /
         ("dqm_gguf_tokenizer_" + name);
}

// --- 手搓 GGUF 字节用的小工具（小端）----------------------------------------

inline void TokAppendU32Le(std::vector<char>& data, uint32_t value) {
  data.push_back(static_cast<char>(value & 0xff));
  data.push_back(static_cast<char>((value >> 8) & 0xff));
  data.push_back(static_cast<char>((value >> 16) & 0xff));
  data.push_back(static_cast<char>((value >> 24) & 0xff));
}

inline void TokAppendU64Le(std::vector<char>& data, uint64_t value) {
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
  }
}

inline void TokAppendString(std::vector<char>& data, const std::string& value) {
  TokAppendU64Le(data, static_cast<uint64_t>(value.size()));
  data.insert(data.end(), value.begin(), value.end());
}

inline void TokAppendMetadataString(std::vector<char>& data,
                                    const std::string& key,
                                    const std::string& value) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeString);
  TokAppendString(data, value);
}

inline void TokAppendMetadataU32(std::vector<char>& data,
                                 const std::string& key, uint32_t value) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeUint32);
  TokAppendU32Le(data, value);
}

inline void TokAppendStringArray(std::vector<char>& data, const std::string& key,
                                 const std::vector<std::string>& values) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeArray);
  TokAppendU32Le(data, kGgufTypeString);
  TokAppendU64Le(data, static_cast<uint64_t>(values.size()));
  for (const std::string& value : values) {
    TokAppendString(data, value);
  }
}

inline void TokAppendI32Array(std::vector<char>& data, const std::string& key,
                              const std::vector<int32_t>& values) {
  TokAppendString(data, key);
  TokAppendU32Le(data, kGgufTypeArray);
  TokAppendU32Le(data, kGgufTypeInt32);
  TokAppendU64Le(data, static_cast<uint64_t>(values.size()));
  for (int32_t value : values) {
    TokAppendU32Le(data, static_cast<uint32_t>(value));
  }
}

// 标准 Qwen2 官方聊天模板（`\n` 是真正的换行符）。
inline std::string TinyQwen2Template() {
  return "{% for message in messages %}{% if loop.first and "
         "messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a "
         "helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + "
         "message['role'] + '\n' + message['content'] + '<|im_end|>' + "
         "'\n'}}{% endfor %}{% if add_generation_prompt %}{{ "
         "'<|im_start|>assistant\n' }}{% endif %}";
}

// 合成一个只带 tokenizer metadata 的 GGUF v3（0 个张量），写到临时路径并返回。
// 词表 14 条：a b ab <s> </s> <|im_start|> <|im_end|> u s e r system assistant \n
// architecture / include_chat_template 可调，用于测 LoadChatTemplateFromGguf。
inline std::filesystem::path WriteTinyGgufTokenizerFixture(
    const std::string& name, const std::string& architecture = "qwen2",
    bool include_chat_template = true) {
  const std::vector<std::string> tokens = {
      "a", "b", "ab", "<s>", "</s>", "<|im_start|>", "<|im_end|>",
      "u", "s", "e", "r", "system", "assistant", "\n",
  };
  const std::vector<int32_t> token_types = {
      1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1,
  };
  const std::vector<std::string> merges = {"a b"};

  constexpr uint64_t kAlignment = 32;
  const uint64_t metadata_count = include_chat_template ? 9 : 8;

  std::vector<char> data;
  data.insert(data.end(), {'G', 'G', 'U', 'F'});
  TokAppendU32Le(data, 3);
  TokAppendU64Le(data, 0);  // n_tensors
  TokAppendU64Le(data, metadata_count);

  TokAppendMetadataString(data, "general.architecture", architecture);
  TokAppendMetadataString(data, "general.name", "dqm-tiny-tokenizer-test");
  TokAppendMetadataU32(data, "general.alignment",
                       static_cast<uint32_t>(kAlignment));
  TokAppendStringArray(data, "tokenizer.ggml.tokens", tokens);
  TokAppendI32Array(data, "tokenizer.ggml.token_type", token_types);
  TokAppendStringArray(data, "tokenizer.ggml.merges", merges);
  TokAppendMetadataU32(data, "tokenizer.ggml.bos_token_id", 3);
  TokAppendMetadataU32(data, "tokenizer.ggml.eos_token_id", 4);
  if (include_chat_template) {
    TokAppendMetadataString(data, "tokenizer.chat_template",
                            TinyQwen2Template());
  }

  size_t padding = static_cast<size_t>(
      (kAlignment - (data.size() % kAlignment)) % kAlignment);
  data.insert(data.end(), padding, '\0');

  std::filesystem::path path = GgufTokenizerTempPath(name);
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to create tokenizer fixture: " +
                             path.string());
  }
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  return path;
}

}  // namespace dqm::test
