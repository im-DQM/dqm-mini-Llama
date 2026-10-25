// GgufTokenizer：从 GGUF metadata 读词表并复用 BPE；以及 CreateGgufTokenizer 工厂。

#include "dqm_llama/gguf_tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dqm_llama/gguf.h"

namespace dqm {

namespace {

// 一个码点编码成 UTF-8 字节串。
std::string CodepointToUtf8(char32_t cp) {
  std::string result;
  if (cp <= 0x7F) {
    result += static_cast<char>(cp);
  } else if (cp <= 0x7FF) {
    result += static_cast<char>(0xC0 | (cp >> 6));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (cp >> 12));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (cp >> 18));
    result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return result;
}

// 从 s 的 pos 处取一个 UTF-8 码点（整段字节），并原地推进 pos。
// 非法首字节或越界：只 ++pos 并返回空串（丢一个字节前进，不会卡死）。
std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
  if (pos >= s.size()) {
    return "";
  }
  unsigned char first = static_cast<unsigned char>(s[pos]);
  size_t len = 1;
  if ((first & 0x80) == 0) {
    len = 1;
  } else if ((first & 0xE0) == 0xC0) {
    len = 2;
  } else if ((first & 0xF0) == 0xE0) {
    len = 3;
  } else if ((first & 0xF8) == 0xF0) {
    len = 4;
  } else {
    ++pos;
    return "";
  }
  if (pos + len > s.size()) {
    ++pos;
    return "";
  }
  std::string result = s.substr(pos, len);
  pos += len;
  return result;
}

// GPT-2 bytes_to_unicode：256 个字节各配一个可打印码点替身（双射）。
std::vector<std::string> BuildBytesToUnicode() {
  std::vector<std::string> map(256);

  std::vector<int> bs;
  for (int c = '!'; c <= '~'; ++c) {
    bs.push_back(c);
  }
  for (int c = 0xA1; c <= 0xAC; ++c) {
    bs.push_back(c);
  }
  for (int c = 0xAE; c <= 0xFF; ++c) {
    bs.push_back(c);
  }

  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      ++n;
    }
  }

  for (size_t i = 0; i < bs.size(); ++i) {
    map[static_cast<size_t>(bs[i])] =
        CodepointToUtf8(static_cast<char32_t>(cs[i]));
  }
  return map;
}

}  // namespace

void GgufTokenizer::BuildByteMappings() {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[static_cast<size_t>(b)]] = static_cast<uint8_t>(b);
  }
}

bool GgufTokenizer::Load(const std::string& gguf_path) {
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;

  BuildByteMappings();

  std::vector<std::string> tokens;
  if (!GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.tokens", tokens)) {
    return false;
  }

  std::vector<int32_t> token_types;
  bool has_token_types = GgufGetMetadataIntArray(
      gguf_path, "tokenizer.ggml.token_type", token_types);

  id_to_token_.resize(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    vocab_[tokens[i]] = static_cast<int>(i);
    id_to_token_[i] = tokens[i];
  }

  // token_type == 3 视为 control/special；按长度降序，长匹配优先。
  if (has_token_types && token_types.size() == tokens.size()) {
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (token_types[i] == 3) {
        special_tokens_.push_back({tokens[i], static_cast<int>(i)});
      }
    }
  }
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const std::pair<std::string, int>& a,
               const std::pair<std::string, int>& b) {
              return a.first.size() > b.first.size();
            });

  // merges.txt 每行 "a b"，行号即 rank。
  std::vector<std::string> merges;
  if (GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.merges", merges)) {
    for (size_t rank = 0; rank < merges.size(); ++rank) {
      const std::string& m = merges[rank];
      size_t space_pos = m.find(' ');
      if (space_pos == std::string::npos) {
        continue;
      }
      std::string left = m.substr(0, space_pos);
      std::string right = m.substr(space_pos + 1);
      merge_ranks_[{left, right}] = static_cast<int>(rank);
    }
  }

  int64_t bos = -1;
  if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.bos_token_id", bos)) {
    bos_id_ = static_cast<int>(bos);
  }
  int64_t eos = -1;
  if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.eos_token_id", eos)) {
    eos_id_ = static_cast<int>(eos);
  }
  int64_t unk = -1;
  if (GgufGetMetadataInt(gguf_path, "tokenizer.ggml.unknown_token_id", unk)) {
    unk_id_ = static_cast<int>(unk);
  }

  return true;
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  if (text.empty()) {
    return {};
  }

  std::vector<int> result;
  size_t pos = 0;

  while (pos < text.size()) {
    // 1. 先试 special（长匹配优先，不经 b2u / BPE）。
    bool matched = false;
    for (const auto& st : special_tokens_) {
      const std::string& token_str = st.first;
      if (pos + token_str.size() <= text.size() &&
          std::memcmp(text.data() + pos, token_str.data(), token_str.size()) ==
              0) {
        result.push_back(st.second);
        pos += token_str.size();
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    // 2. 找下一个 special（或串尾）作普通段边界。
    size_t end = text.size();
    for (const auto& st : special_tokens_) {
      size_t p = text.find(st.first, pos);
      if (p != std::string::npos && p < end) {
        end = p;
      }
    }

    std::string segment = text.substr(pos, end - pos);
    pos = end;

    // 3. 逐字节 b2u -> 初始 piece。
    std::vector<std::string> word;
    for (unsigned char c : segment) {
      if (static_cast<size_t>(c) >= b2u_.size()) {
        return {};
      }
      word.push_back(b2u_[static_cast<size_t>(c)]);
    }

    // 4. 每轮合并全局 rank 最小的相邻对。
    while (word.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      size_t best_idx = word.size();
      for (size_t i = 0; i + 1 < word.size(); ++i) {
        auto it = merge_ranks_.find({word[i], word[i + 1]});
        if (it != merge_ranks_.end() && it->second < best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }
      if (best_idx >= word.size()) {
        break;
      }
      word[best_idx] += word[best_idx + 1];
      word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_idx) + 1);
    }

    // 5. piece -> id（查不到压 unk）。
    for (const auto& w : word) {
      auto it = vocab_.find(w);
      if (it != vocab_.end()) {
        result.push_back(it->second);
      } else {
        result.push_back(unk_id_);
      }
    }
  }

  return result;
}

std::string GgufTokenizer::DecodeToken(int token) const {
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[static_cast<size_t>(token)];
  }
  return "";
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string text;
  for (int id : tokens) {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
      text += id_to_token_[static_cast<size_t>(id)];
    }
  }

  // 逐码点把 unicode 替身还原回原始字节；非替身码点原样输出。
  std::string result;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t start = pos;
    std::string cp = NextUtf8Codepoint(text, pos);
    if (cp.empty()) {
      if (pos == start) {
        ++pos;
      }
      continue;
    }
    auto it = u2b_.find(cp);
    if (it != u2b_.end()) {
      result += static_cast<char>(it->second);
    } else {
      result += cp;
    }
  }
  return result;
}

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
  auto tok = std::make_unique<GgufTokenizer>();
  if (tok->Load(gguf_path)) {
    return tok;
  }
  return nullptr;
}

}  // namespace dqm
