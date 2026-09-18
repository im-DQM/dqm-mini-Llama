#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "dqm_llama/tokenizer.h"

// 源码根目录，由 CMake 编译期传入（见 CMakeLists.txt）。
// 兜底：万一没定义，退回当前目录。
#ifndef DQM_TEST_DATA_DIR
#define DQM_TEST_DATA_DIR "."
#endif

using dqm::AsciiTokenizer;
using dqm::CreateTokenizer;
using dqm::JsonVocabTokenizer;

// ===========================================================================
// AsciiTokenizer
// ===========================================================================

TEST(AsciiTokenizer, EncodeBasic) {
  AsciiTokenizer tok;
  std::vector<int> tokens = tok.Encode("hi");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], tok.bos_id());  // 1
  EXPECT_EQ(tokens[1], static_cast<int>('h'));
  EXPECT_EQ(tokens[2], static_cast<int>('i'));
}

TEST(AsciiTokenizer, EncodeEmpty) {
  AsciiTokenizer tok;
  std::vector<int> tokens = tok.Encode("");
  ASSERT_EQ(tokens.size(), 1u);
  EXPECT_EQ(tokens[0], tok.bos_id());
}

TEST(AsciiTokenizer, DecodeToken) {
  AsciiTokenizer tok;
  EXPECT_EQ(tok.DecodeToken(tok.bos_id()), "<bos>");
  EXPECT_EQ(tok.DecodeToken(tok.eos_id()), "<eos>");
  EXPECT_EQ(tok.DecodeToken(tok.unk_id()), "<unk>");
  EXPECT_EQ(tok.DecodeToken(static_cast<int>('a')), "a");
  EXPECT_EQ(tok.DecodeToken(32), " ");        // 空格：可打印
  EXPECT_EQ(tok.DecodeToken(10), "");         // 控制字符：空
  EXPECT_EQ(tok.DecodeToken(127), "");        // DEL：空
  EXPECT_EQ(tok.DecodeToken(200), "<unk>");   // 越界：<unk>
}

TEST(AsciiTokenizer, DecodeSequence) {
  AsciiTokenizer tok;
  std::vector<int> tokens = {tok.bos_id(), static_cast<int>('h'),
                             static_cast<int>('i')};
  EXPECT_EQ(tok.Decode(tokens), "<bos>hi");
}

TEST(AsciiTokenizer, VocabSizeAndSpecialIds) {
  AsciiTokenizer tok;
  EXPECT_EQ(tok.vocab_size(), 128);
  EXPECT_EQ(tok.bos_id(), 1);
  EXPECT_EQ(tok.eos_id(), 2);
  EXPECT_EQ(tok.unk_id(), 0);
}

// ===========================================================================
// JsonVocabTokenizer
// ===========================================================================

namespace {

// 项目自带的 tiny 词表（= 参考 models/tiny/vocab.json）。
// 用编译期传入的源码根目录拼绝对路径，所以从哪个工作目录跑都能找到。
std::string TinyVocabPath() {
  return std::string(DQM_TEST_DATA_DIR) + "/models/tiny/vocab.json";
}

}  // namespace

TEST(JsonVocabTokenizer, LoadsAndVocabSize) {
  JsonVocabTokenizer tok(TinyVocabPath());
  EXPECT_EQ(tok.vocab_size(), 128);
  EXPECT_EQ(tok.bos_id(), 1);
  EXPECT_EQ(tok.eos_id(), 2);
  EXPECT_EQ(tok.unk_id(), 0);
}

TEST(JsonVocabTokenizer, EncodeBasic) {
  JsonVocabTokenizer tok(TinyVocabPath());
  std::vector<int> tokens = tok.Encode("hi");
  ASSERT_EQ(tokens.size(), 3u);
  EXPECT_EQ(tokens[0], tok.bos_id());
  EXPECT_EQ(tokens[1], static_cast<int>('h'));
  EXPECT_EQ(tokens[2], static_cast<int>('i'));
}

TEST(JsonVocabTokenizer, DecodeToken) {
  JsonVocabTokenizer tok(TinyVocabPath());
  EXPECT_EQ(tok.DecodeToken(0), "<unk>");
  EXPECT_EQ(tok.DecodeToken(1), "<bos>");
  EXPECT_EQ(tok.DecodeToken(2), "<eos>");
  EXPECT_EQ(tok.DecodeToken(static_cast<int>('a')), "a");
  EXPECT_EQ(tok.DecodeToken(32), " ");
  EXPECT_EQ(tok.DecodeToken(10), "");        // 词表无此 id -> 空洞 -> 空串
  EXPECT_EQ(tok.DecodeToken(127), "");       // 有条目但 content 为空
  EXPECT_EQ(tok.DecodeToken(200), "<unk>");  // 越界
}

TEST(JsonVocabTokenizer, DecodeSequence) {
  JsonVocabTokenizer tok(TinyVocabPath());
  std::vector<int> tokens = {tok.bos_id(), static_cast<int>('h'),
                             static_cast<int>('i')};
  EXPECT_EQ(tok.Decode(tokens), "<bos>hi");
}

// 词表在可打印范围内与 ASCII 等价 -> 同一段文本，两个 tokenizer 结果应相同。
TEST(JsonVocabTokenizer, MatchesAscii) {
  AsciiTokenizer ascii;
  JsonVocabTokenizer json(TinyVocabPath());
  const std::string prompt = "hello!";
  EXPECT_EQ(json.Encode(prompt), ascii.Encode(prompt));
  EXPECT_EQ(json.Decode(json.Encode(prompt)),
            ascii.Decode(ascii.Encode(prompt)));
}

TEST(JsonVocabTokenizer, MissingFileThrows) {
  EXPECT_THROW(JsonVocabTokenizer("/no/such/vocab.json"), std::runtime_error);
}

// ===========================================================================
// CreateTokenizer（工厂）
// ===========================================================================

TEST(CreateTokenizer, UsesJsonWhenFileExists) {
  auto tok = CreateTokenizer(TinyVocabPath());
  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->vocab_size(), 128);
  EXPECT_EQ(dynamic_cast<AsciiTokenizer*>(tok.get()), nullptr);  // 不是 Ascii
}

TEST(CreateTokenizer, FallsBackToAsciiWhenEmptyPath) {
  auto tok = CreateTokenizer("");
  ASSERT_NE(tok, nullptr);
  EXPECT_NE(dynamic_cast<AsciiTokenizer*>(tok.get()), nullptr);
}

TEST(CreateTokenizer, FallsBackWhenFileMissing) {
  auto tok = CreateTokenizer("/no/such/vocab.json");
  ASSERT_NE(tok, nullptr);
  EXPECT_NE(dynamic_cast<AsciiTokenizer*>(tok.get()), nullptr);
}
