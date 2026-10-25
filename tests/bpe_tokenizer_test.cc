// BpeTokenizer 的行为契约（参考 tests/test_tokenizer.cc 的 BPE 用例）。

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "dqm_llama/tokenizer.h"

namespace dqm {
namespace {

struct TempBpeFiles {
  std::filesystem::path dir;
  std::filesystem::path vocab;
  std::filesystem::path merges;
  std::filesystem::path special;

  ~TempBpeFiles() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

TempBpeFiles WriteTempBpeFiles() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  TempBpeFiles files;
  files.dir = std::filesystem::temp_directory_path() /
              ("dqm_bpe_tokenizer_" + std::to_string(stamp));
  std::filesystem::create_directories(files.dir);
  files.vocab = files.dir / "vocab.json";
  files.merges = files.dir / "merges.txt";
  files.special = files.dir / "special_tokens.json";

  {
    std::ofstream out(files.vocab);
    out << "{\n"
        << "  \"h\": 0,\n"
        << "  \"e\": 1,\n"
        << "  \"l\": 2,\n"
        << "  \"o\": 3,\n"
        << "  \"he\": 4,\n"
        << "  \"hel\": 5,\n"
        << "  \"hell\": 6,\n"
        << "  \"hello\": 7,\n"
        << "  \"<|im_start|>\": 8,\n"
        << "  \"<unk>\": 9,\n"
        << "  \"<bos>\": 10,\n"
        << "  \"<eos>\": 11\n"
        << "}\n";
  }
  {
    std::ofstream out(files.merges);
    out << "h e\n"
        << "he l\n"
        << "hel l\n"
        << "hell o\n";
  }
  {
    std::ofstream out(files.special);
    out << "{ \"bos_id\": 10, \"eos_id\": 11, \"unk_id\": 9 }\n";
  }
  return files;
}

TEST(BpeTokenizer, EncodeDecodeAndSpecialToken) {
  TempBpeFiles files = WriteTempBpeFiles();
  auto tok = CreateBpeTokenizer(files.vocab.string(), files.merges.string(),
                                files.special.string());
  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->bos_id(), 10);
  EXPECT_EQ(tok->eos_id(), 11);
  EXPECT_EQ(tok->unk_id(), 9);

  std::vector<int> hello = tok->Encode("hello");
  ASSERT_EQ(hello.size(), 1u);
  EXPECT_EQ(hello[0], 7);
  EXPECT_EQ(tok->Decode(hello), "hello");

  std::vector<int> special = tok->Encode("<|im_start|>");
  ASSERT_EQ(special.size(), 1u);
  EXPECT_EQ(special[0], 8);
  EXPECT_EQ(tok->Decode(special), "<|im_start|>");
}

TEST(BpeTokenizer, EncodeEmptyIsEmpty) {
  TempBpeFiles files = WriteTempBpeFiles();
  auto tok = CreateBpeTokenizer(files.vocab.string(), files.merges.string(),
                                files.special.string());
  ASSERT_NE(tok, nullptr);
  EXPECT_TRUE(tok->Encode("").empty());
}

TEST(BpeTokenizer, DecodeUnknownIdIsEmpty) {
  TempBpeFiles files = WriteTempBpeFiles();
  auto tok = CreateBpeTokenizer(files.vocab.string(), files.merges.string(),
                                files.special.string());
  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->DecodeToken(9999), "");
}

}  // namespace
}  // namespace dqm
