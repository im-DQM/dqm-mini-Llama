// BpeTokenizer：GPT-2 风格 BPE；以及 CreateBpeTokenizer 工厂。

#include "dqm_llama/tokenizer.h"

#include <stdexcept>

namespace dqm {

void BpeTokenizer::BuildByteMappings() {
  throw std::logic_error("TODO: BpeTokenizer::BuildByteMappings");
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
  throw std::logic_error("TODO: BpeTokenizer::Load");
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
  throw std::logic_error("TODO: BpeTokenizer::Encode");
}

std::string BpeTokenizer::DecodeToken(int token) const {
  throw std::logic_error("TODO: BpeTokenizer::DecodeToken");
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
  throw std::logic_error("TODO: BpeTokenizer::Decode");
}

std::unique_ptr<ITokenizer> CreateBpeTokenizer(
    const std::string& vocab_path, const std::string& merges_path,
    const std::string& special_path) {
  throw std::logic_error("TODO: CreateBpeTokenizer");
}

}  // namespace dqm
