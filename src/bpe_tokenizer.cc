// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

// Primary header for the BPE tokenizer implementation.
// clang-format off
#include "dqm_llama/tokenizer.h"
// clang-format on

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace dqm {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

static std::string CodepointToUtf8(char32_t cp) {
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

// Extract next UTF-8 codepoint as a string; return empty if invalid
static std::string NextUtf8Codepoint(const std::string& s, size_t& pos) {
  if (pos >= s.size()) {
    return "";
  }
  unsigned char first = s[pos];
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
    // Invalid UTF-8, skip one byte
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

// ---------------------------------------------------------------------------
// GPT-2 bytes_to_unicode mapping
// ---------------------------------------------------------------------------

static std::vector<std::string> BuildBytesToUnicode() {
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
    map[bs[i]] = CodepointToUtf8(static_cast<char32_t>(cs[i]));
  }
  return map;
}

// ---------------------------------------------------------------------------
// Simple JSON number parser helper
// ---------------------------------------------------------------------------

static bool ParseJsonInt(const std::string& s, size_t& pos, int& out) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos >= s.size()) {
    return false;
  }
  size_t start = pos;
  if (s[pos] == '-') {
    ++pos;
  }
  while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (start == pos) {
    return false;
  }
  try {
    out = std::stoi(s.substr(start, pos - start));
    return true;
  } catch (...) {
    return false;
  }
}

static bool ParseJsonString(const std::string& s, size_t& pos,
                            std::string& out) {
  while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) {
    ++pos;
  }
  if (pos >= s.size() || s[pos] != '"') {
    return false;
  }
  ++pos;  // skip opening quote
  out.clear();
  while (pos < s.size() && s[pos] != '"') {
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      char esc = s[pos + 1];
      switch (esc) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          if (pos + 5 < s.size()) {
            std::string hex = s.substr(pos + 2, 4);
            try {
              char32_t cp = static_cast<char32_t>(std::stoul(hex, nullptr, 16));
              out += CodepointToUtf8(cp);
            } catch (...) {
              out += '?';
            }
            pos += 4;
          }
          break;
        }
        default:
          out += esc;
          break;
      }
      pos += 2;
    } else {
      out += s[pos];
      ++pos;
    }
  }
  if (pos < s.size() && s[pos] == '"') {
    ++pos;
  }
  return true;
}

// ---------------------------------------------------------------------------
// BpeTokenizer
// ---------------------------------------------------------------------------

void BpeTokenizer::BuildByteMappings() {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  for (int b = 0; b < 256; ++b) {
    u2b_[b2u_[b]] = static_cast<uint8_t>(b);
  }
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
  BuildByteMappings();

  // Load vocab.json: { "token": id, ... }
  {
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
      std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
      return false;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    std::string content = buf.str();

    size_t pos = 0;
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos]))) {
      ++pos;
    }
    if (pos < content.size() && content[pos] == '{') {
      ++pos;
    }
    while (pos < content.size()) {
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }
      if (pos < content.size() && content[pos] == '}') {
        break;
      }
      std::string key;
      if (!ParseJsonString(content, pos, key)) {
        break;
      }
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }
      if (pos >= content.size() || content[pos] != ':') {
        break;
      }
      ++pos;
      int val = 0;
      if (!ParseJsonInt(content, pos, val)) {
        break;
      }
      vocab_[key] = val;
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }
      if (pos < content.size() && content[pos] == ',') {
        ++pos;
      }
    }

    if (vocab_.empty()) {
      std::cerr << "Vocab is empty" << std::endl;
      return false;
    }

    int max_id = 0;
    for (const auto& kv : vocab_) {
      if (kv.second > max_id) {
        max_id = kv.second;
      }
    }
    id_to_token_.resize(max_id + 1);
    for (const auto& kv : vocab_) {
      id_to_token_[kv.second] = kv.first;
    }

    // Extract special tokens (e.g. <|im_start|>) from vocab for longest-match
    // Encode.
    for (const auto& kv : vocab_) {
      if (kv.first.size() >= 4 && kv.first.front() == '<' &&
          kv.first[1] == '|' && kv.first[kv.first.size() - 2] == '|' &&
          kv.first.back() == '>') {
        special_tokens_.push_back({kv.first, kv.second});
      }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto& a, const auto& b) {
                return a.first.size() > b.first.size();
              });
  }

  // Load merges.txt: "token1 token2" per line
  {
    std::ifstream f(merges_path);
    if (!f.is_open()) {
      std::cerr << "Failed to open merges: " << merges_path << std::endl;
      return false;
    }
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string a, b;
      if (ss >> a >> b) {
        merge_ranks_[{a, b}] = rank++;
      }
    }
  }

  // Load special_tokens.json
  {
    std::ifstream f(special_path);
    if (f.is_open()) {
      std::stringstream buf;
      buf << f.rdbuf();
      std::string content = buf.str();
      auto find_int = [&](const std::string& key) -> int {
        size_t p = content.find('"' + key + '"');
        if (p == std::string::npos) {
          return -1;
        }
        p = content.find(':', p);
        if (p == std::string::npos) {
          return -1;
        }
        size_t start = p + 1;
        while (start < content.size() &&
               std::isspace(static_cast<unsigned char>(content[start]))) {
          ++start;
        }
        size_t end = start;
        while (end < content.size() &&
               (std::isdigit(static_cast<unsigned char>(content[end])) ||
                content[end] == '-')) {
          ++end;
        }
        if (start == end) {
          return -1;
        }
        try {
          return std::stoi(content.substr(start, end - start));
        } catch (...) {
          return -1;
        }
      };
      bos_id_ = find_int("bos_id");
      eos_id_ = find_int("eos_id");
      unk_id_ = find_int("unk_id");
    }
  }

  std::cout << "BPE tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
  return true;
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
  if (text.empty()) {
    return {};
  }

  std::vector<int> result;
  size_t pos = 0;

  while (pos < text.size()) {
    // Try to match a special token first (longest match)
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

    // Find the next special token or end of string
    size_t end = text.size();
    for (const auto& st : special_tokens_) {
      size_t p = text.find(st.first, pos);
      if (p != std::string::npos && p < end) {
        end = p;
      }
    }

    // Encode the plain text segment [pos, end) with BPE
    std::string segment = text.substr(pos, end - pos);
    pos = end;

    // 1. Convert text to initial word pieces (byte -> unicode mapping)
    std::vector<std::string> word;
    for (unsigned char c : segment) {
      if (c >= b2u_.size()) {
        std::cerr << "ERROR: byte " << static_cast<int>(c) << " >= b2u_ size "
                  << b2u_.size() << "\n";
        return {};
      }
      word.push_back(b2u_[c]);
    }

    // 2. Apply BPE merges
    while (word.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      size_t best_idx = word.size();  // invalid

      for (size_t i = 0; i + 1 < word.size(); ++i) {
        auto it = merge_ranks_.find({word[i], word[i + 1]});
        if (it != merge_ranks_.end() && it->second < best_rank) {
          best_rank = it->second;
          best_idx = i;
        }
      }

      if (best_idx >= word.size()) {
        break;  // no more merges possible
      }

      word[best_idx] += word[best_idx + 1];
      word.erase(word.begin() + best_idx + 1);
    }

    // 3. Map to token IDs
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

std::string BpeTokenizer::DecodeToken(int token) const {
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[token];
  }
  return "";
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
  // 1. Concatenate token strings
  std::string text;
  for (int id : tokens) {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
      text += id_to_token_[id];
    }
  }

  // 2. Convert unicode chars back to bytes
  std::string result;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t start = pos;
    std::string cp = NextUtf8Codepoint(text, pos);
    if (cp.empty()) {
      // Invalid UTF-8, skip
      if (pos == start) {
        ++pos;
      }
      continue;
    }
    auto it = u2b_.find(cp);
    if (it != u2b_.end()) {
      result += static_cast<char>(it->second);
    } else {
      // Not a byte token; pass through as-is (shouldn't happen for standard
      // BPE)
      result += cp;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<ITokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path) {
  auto tok = std::make_unique<BpeTokenizer>();
  if (tok->Load(vocab_path, merges_path, special_path)) {
    return tok;
  }
  return nullptr;
}

}  // namespace dqm
