#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "dqm_llama/gguf_tokenizer.h"
#include "gguf_tokenizer_fixture.h"

// 源码根目录，由 CMake 编译期传入（见 CMakeLists.txt）。兜底：退回当前目录。
#ifndef DQM_TEST_DATA_DIR
#define DQM_TEST_DATA_DIR "."
#endif

namespace {

std::string TestDataPath(const std::string& rel) {
  return std::string(DQM_TEST_DATA_DIR) + "/" + rel;
}

}  // namespace

// ---------------------------------------------------------------------------
// 从合成 GGUF 加载：词表规模 + 三个特殊 id
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, LoadsTinyFixture) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tiny_load.gguf");
  auto tok = dqm::CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->vocab_size(), 14);
  EXPECT_EQ(tok->bos_id(), 3);
  EXPECT_EQ(tok->eos_id(), 4);
}

// ---------------------------------------------------------------------------
// 编码 + 解码：merge 规则让 "ab" 合成一个 token（id=2）
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, EncodeDecodeRoundtrip) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tiny_roundtrip.gguf");
  auto tok = dqm::CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_NE(tok, nullptr);
  std::vector<int> ids = tok->Encode("ab");
  EXPECT_EQ(ids, (std::vector<int>{2}));
  EXPECT_EQ(tok->Decode(ids), "ab");
}

// ---------------------------------------------------------------------------
// special token 整体成 id（不走 BPE），且长匹配优先
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, SpecialTokens) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tiny_special.gguf");
  auto tok = dqm::CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->Encode("<|im_start|>"), (std::vector<int>{5}));
  EXPECT_EQ(tok->Encode("<|im_end|>"), (std::vector<int>{6}));
  EXPECT_EQ(tok->eos_id(), 4);
  EXPECT_EQ(tok->DecodeToken(4), "</s>");
}

// ---------------------------------------------------------------------------
// special 作硬边界：前后普通文本各自走 BPE
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, SpecialSplitsSegments) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tiny_segments.gguf");
  auto tok = dqm::CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_NE(tok, nullptr);
  // "ab" + special + "ab" -> [2, 5, 2]
  EXPECT_EQ(tok->Encode("ab<|im_start|>ab"),
            (std::vector<int>{2, 5, 2}));
}

// ---------------------------------------------------------------------------
// 词表外的字节 -> unk_id（本 fixture 无 unknown_token_id -> -1）
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, UnknownByteMapsToUnk) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tiny_unk.gguf");
  auto tok = dqm::CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->Encode("z"), (std::vector<int>{-1}));
}

TEST(GgufTokenizer, DecodeTokenOutOfRangeIsEmpty) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tiny_oob.gguf");
  auto tok = dqm::CreateGgufTokenizer(path.string());
  std::filesystem::remove(path);

  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->DecodeToken(-1), "");
  EXPECT_EQ(tok->DecodeToken(999), "");
}

// ---------------------------------------------------------------------------
// 错误路径：缺文件 / 不是 GGUF —— 都回 nullptr，不抛
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, RejectsMissingFile) {
  auto tok = dqm::CreateGgufTokenizer(TestDataPath("models/chat/nonexistent.gguf"));
  EXPECT_EQ(tok, nullptr);
}

TEST(GgufTokenizer, RejectsNonGgufFile) {
  auto tok = dqm::CreateGgufTokenizer(TestDataPath("models/tiny/model.json"));
  EXPECT_EQ(tok, nullptr);
}

// ---------------------------------------------------------------------------
// 有真模型时顺带跑一遍（dqm 里通常没有 -> 跳过）
// ---------------------------------------------------------------------------
TEST(GgufTokenizer, LoadsRealModelWhenAvailable) {
  const std::string path =
      TestDataPath("models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "real model not present in dqm";
  }

  auto tok = dqm::CreateGgufTokenizer(path);
  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->vocab_size(), 151936);
  EXPECT_EQ(tok->bos_id(), 151643);
  EXPECT_EQ(tok->eos_id(), 151645);
  EXPECT_EQ(tok->Encode("<|im_start|>")[0], 151644);
}
