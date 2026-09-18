// JsonVocabTokenizer：从 JSON 词表文件加载；以及 CreateTokenizer 工厂。
//
// 词表格式：[{"id": 0, "content": "<unk>", "special": true}, ...]

#include "dqm_llama/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dqm {

namespace {

// --- 手写 JSON 解析辅助（只认 [{"id":,"content":,"special":}, ...] 这种数组）---
// 约定：content 是整份文件内容，pos 是游标（传引用），函数边解析边推进它。

// 解析一个 JSON 字符串字面量；返回解转义后的内容，pos 停在闭引号之后。
std::string ParseJsonString(const std::string& content, size_t& pos) {
  while (pos < content.size() && content[pos] != '"') {
    ++pos;
  }
  if (pos >= content.size()) {
    throw std::runtime_error("expected opening quote for string");
  }
  ++pos;  // 跳过开引号
  std::string result;
  while (pos < content.size() && content[pos] != '"') {
    if (content[pos] == '\\' && pos + 1 < content.size()) {
      char next = content[pos + 1];
      if (next == 'n') {
        result += '\n';
      } else if (next == 't') {
        result += '\t';
      } else if (next == 'r') {
        result += '\r';
      } else if (next == '\\') {
        result += '\\';
      } else if (next == '"') {
        result += '"';
      } else if (next == 'u' && pos + 5 < content.size()) {
        // \uXXXX：四位十六进制（只支持 ASCII 码位，其余降级成 '?'）
        std::string hex = content.substr(pos + 2, 4);
        unsigned int codepoint =
            static_cast<unsigned int>(std::stoul(hex, nullptr, 16));
        result += (codepoint <= 0x7F) ? static_cast<char>(codepoint) : '?';
        pos += 6;
        continue;
      } else {
        result += next;
      }
      pos += 2;
    } else {
      result += content[pos];
      ++pos;
    }
  }
  if (pos >= content.size()) {
    throw std::runtime_error("unterminated string");
  }
  ++pos;  // 跳过闭引号
  return result;
}

// 解析一个整数（可带负号）；pos 停在数字之后。
int ParseJsonInt(const std::string& content, size_t& pos) {
  while (pos < content.size() &&
         !std::isdigit(static_cast<unsigned char>(content[pos])) &&
         content[pos] != '-') {
    ++pos;
  }
  size_t start = pos;
  if (pos < content.size() && content[pos] == '-') {
    ++pos;
  }
  while (pos < content.size() &&
         std::isdigit(static_cast<unsigned char>(content[pos]))) {
    ++pos;
  }
  return std::stoi(content.substr(start, pos - start));
}

// 解析 true / false；pos 各前进 4 / 5。
bool ParseJsonBool(const std::string& content, size_t& pos) {
  if (content.compare(pos, 4, "true") == 0) {
    pos += 4;
    return true;
  }
  if (content.compare(pos, 5, "false") == 0) {
    pos += 5;
    return false;
  }
  throw std::runtime_error("expected true or false");
}

}  // namespace

JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f.is_open()) {
    throw std::runtime_error("failed to open vocab: " + vocab_path);
  }
  std::stringstream buffer;
  buffer << f.rdbuf();
  if (f.bad()) {
    throw std::runtime_error("failed to read vocab: " + vocab_path);
  }

  std::string content = buffer.str();

  size_t pos = content.find('[');
  if (pos == std::string::npos) {
    throw std::runtime_error("vocab JSON must be an array");
  }
  ++pos;  // 跳过 '['

  int max_id = -1;

  while (pos < content.size()) {
    // 找下一个对象的 '{'
    while (pos < content.size() && content[pos] != '{') {
      if (content[pos] == ']') {
        break;
      }
      ++pos;
    }
    if (pos >= content.size() || content[pos] != '{') {
      break;
    }
    ++pos;  // 跳过 '{'

    VocabEntry entry;
    bool has_id = false;
    bool has_content = false;

    // 逐个字段
    while (pos < content.size() && content[pos] != '}') {
      // 跳过空白与逗号
      while (pos < content.size() &&
             (std::isspace(static_cast<unsigned char>(content[pos])) ||
              content[pos] == ',')) {
        ++pos;
      }
      if (pos >= content.size() || content[pos] == '}') {
        break;
      }

      std::string key = ParseJsonString(content, pos);

      // 跳到冒号
      while (pos < content.size() && content[pos] != ':') {
        ++pos;
      }
      if (pos < content.size()) {
        ++pos;
      }
      // 跳过冒号后的空白
      while (pos < content.size() &&
             std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
      }

      if (key == "id") {
        entry.id = ParseJsonInt(content, pos);
        has_id = true;
      } else if (key == "content") {
        entry.content = ParseJsonString(content, pos);
        has_content = true;
      } else if (key == "special") {
        entry.special = ParseJsonBool(content, pos);
      } else {
        // 未知键：跳过它的值
        if (pos < content.size() && content[pos] == '"') {
          ParseJsonString(content, pos);
        } else {
          while (pos < content.size() && content[pos] != ',' &&
                 content[pos] != '}') {
            ++pos;
          }
        }
      }
    }

    if (pos < content.size() && content[pos] == '}') {
      ++pos;  // 跳过 '}'
    }

    if (!has_id || !has_content) {
      throw std::runtime_error("vocab entry missing id or content");
    }
    if (entry.id < 0) {
      throw std::runtime_error("vocab entry id must be non-negative");
    }
    if (entry.id > max_id) {
      max_id = entry.id;
    }

    // 特殊 token：按 content 字符串覆盖默认 id
    if (entry.content == "<bos>") {
      bos_id_ = entry.id;
    } else if (entry.content == "<eos>") {
      eos_id_ = entry.id;
    } else if (entry.content == "<unk>") {
      unk_id_ = entry.id;
    }

    // 解码表：id -> 条目（空洞位置留默认构造的条目）
    if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
      id_to_entry_.resize(entry.id + 1);
    }
    id_to_entry_[entry.id] = entry;
    // 编码表：content -> id（顺序待会儿排）
    content_to_id_.push_back({entry.content, entry.id});
  }

  if (max_id < 0) {
    throw std::runtime_error("vocab file contains no entries");
  }

  vocab_size_ = max_id + 1;

  // 按 content 长度降序 —— Encode 的最长匹配依赖此顺序
  std::sort(content_to_id_.begin(), content_to_id_.end(),
            [](const auto& a, const auto& b) {
              return a.first.size() > b.first.size();
            });
}

std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens;
  tokens.push_back(bos_id_);

  size_t pos = 0;
  while (pos < text.size()) {
    bool matched = false;
    // 表已按长度降序 -> 第一个命中者即最长匹配
    for (const auto& pair : content_to_id_) {
      const std::string& content = pair.first;
      if (content.empty()) {
        continue;  // 空 content 会恒匹配 -> 死循环，必须跳过
      }
      if (text.compare(pos, content.size(), content) == 0) {
        tokens.push_back(pair.second);
        pos += content.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      // 没词表条目能匹配 -> 吃 1 字节当 unk，保证前进、保证终止
      tokens.push_back(unk_id_);
      ++pos;
    }
  }

  return tokens;
}

std::string JsonVocabTokenizer::DecodeToken(int token) const {
  if (token < 0 || static_cast<size_t>(token) >= id_to_entry_.size()) {
    return "<unk>";
  }
  return id_to_entry_[token].content;
}

std::string JsonVocabTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
  if (!vocab_path.empty()) {
    std::ifstream f(vocab_path);
    if (f.good()) {
      try {
        return std::make_unique<JsonVocabTokenizer>(vocab_path);
      } catch (const std::exception& e) {
        std::cerr << "Warning: failed to load vocab '" << vocab_path
                  << "', falling back to ASCII tokenizer: " << e.what()
                  << std::endl;
      }
    }
  }
  return std::make_unique<AsciiTokenizer>();
}

}  // namespace dqm
