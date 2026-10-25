#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "dqm_llama/prompt_builder.h"
#include "gguf_tokenizer_fixture.h"

// ---------------------------------------------------------------------------
// 从 GGUF 读模板：有 tokenizer.chat_template -> 返回原文
// ---------------------------------------------------------------------------
TEST(GgufChatTemplate, LoadsTemplateFromGguf) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tpl_qwen2.gguf");
  std::string tmpl = dqm::LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  EXPECT_FALSE(tmpl.empty());
  EXPECT_NE(tmpl.find("{% for message in messages %}"), std::string::npos);
}

// ---------------------------------------------------------------------------
// 没有模板、但 architecture == qwen2 -> 返回速记名 "qwen2"
// ---------------------------------------------------------------------------
TEST(GgufChatTemplate, MissingTemplateFallsBackToQwen2ByArch) {
  std::filesystem::path path = dqm::test::WriteTinyGgufTokenizerFixture(
      "tpl_no_qwen2.gguf", "qwen2", /*include_chat_template=*/false);
  std::string tmpl = dqm::LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  EXPECT_EQ(tmpl, "qwen2");
}

// ---------------------------------------------------------------------------
// 没有模板、architecture 也不是 qwen2 -> 空串
// ---------------------------------------------------------------------------
TEST(GgufChatTemplate, MissingTemplateAndUnknownArchIsEmpty) {
  std::filesystem::path path = dqm::test::WriteTinyGgufTokenizerFixture(
      "tpl_no_llama.gguf", "llama", /*include_chat_template=*/false);
  std::string tmpl = dqm::LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  EXPECT_TRUE(tmpl.empty());
}

// ---------------------------------------------------------------------------
// 端到端：读出的模板喂给 PromptBuilder，经 Jinja 引擎拼出规范 prompt
// ---------------------------------------------------------------------------
TEST(GgufChatTemplate, BuildsPromptThroughEngine) {
  std::filesystem::path path =
      dqm::test::WriteTinyGgufTokenizerFixture("tpl_build.gguf");
  std::string tmpl = dqm::LoadChatTemplateFromGguf(path.string());
  std::filesystem::remove(path);

  dqm::PromptBuilder builder;
  builder.SetChatTemplate(tmpl);
  std::string prompt = builder.Build({{"user", "Hello"}});

  // 无 system -> 模板自动注入默认 system；user 段；末尾 assistant 钩子
  EXPECT_NE(prompt.find("<|im_start|>system"), std::string::npos);
  EXPECT_NE(prompt.find("You are a helpful assistant."), std::string::npos);
  EXPECT_NE(prompt.find("<|im_start|>user"), std::string::npos);
  EXPECT_NE(prompt.find("Hello"), std::string::npos);
  EXPECT_NE(prompt.find("<|im_start|>assistant"), std::string::npos);
}
