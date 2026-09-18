// AsciiTokenizer：0..127 字符级映射（玩具实现）。

#include "dqm_llama/tokenizer.h"

#include <stdexcept>

namespace dqm {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  std::vector<int> tokens(1, bos_id());
  for (char c : text) {
    int id = static_cast<int>(c);
    if (id >= kVocabSize) {
      tokens.push_back(unk_id());
    } else {
      tokens.push_back(id);
    }
  }
  return tokens;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
  if (token == bos_id()) {
    return "<bos>";
  }
  if (token == eos_id()) {
    return "<eos>";
  }
  if (token == unk_id() || token >= vocab_size()) {
    return "<unk>";
  }
  if (token < 32 || token == 127) {
    // Control characters: represent as empty
    return "";
  }
  return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  std::string res;
  for (int token : tokens) {
    res += DecodeToken(token);
  }
  return res;
}

}  // namespace dqm 
