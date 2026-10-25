#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "dqm_llama/context.h"
#include "dqm_llama/forward.h"
#include "dqm_llama/gguf_loader.h"
#include "gguf_loader_fixture.h"

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
// 用 tiny 合成的 GGUF 加载，验 config + 张量形状
// ---------------------------------------------------------------------------
TEST(GgufLoader, LoadsTinyFixture) {
  std::filesystem::path path =
      dqm::test::WriteTinyLlamaGgufFixture("tiny_loader_fixture.gguf");
  dqm::MiniLlamaModel model = dqm::LoadGgufModel(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());

  EXPECT_EQ(model.config.vocab_size, 128);
  EXPECT_EQ(model.config.dim, 32);
  EXPECT_EQ(model.config.n_layers, 2);
  EXPECT_EQ(model.config.n_heads, 4);
  EXPECT_EQ(model.config.n_kv_heads, 4);
  EXPECT_EQ(model.config.head_dim, 8);
  EXPECT_EQ(model.config.hidden_dim, 86);
  EXPECT_EQ(model.config.max_seq_len, 128);
  EXPECT_NEAR(model.config.rms_norm_eps, 1e-5f, 1e-7f);

  ASSERT_GT(model.token_embedding.data.size(), 0u);
  ASSERT_EQ(model.token_embedding.shape.size(), 2u);
  EXPECT_EQ(model.token_embedding.shape[0], 128);
  EXPECT_EQ(model.token_embedding.shape[1], 32);

  ASSERT_EQ(static_cast<int>(model.layers.size()), 2);
  for (int i = 0; i < 2; ++i) {
    const dqm::LayerWeights& lw = model.layers[static_cast<size_t>(i)];

    EXPECT_GT(lw.attention_norm.data.size(), 0u);
    EXPECT_EQ(lw.attention_norm.shape[0], 32);

    EXPECT_GT(lw.wq.f32_data.size(), 0u);
    EXPECT_EQ(lw.wq.shape, (std::vector<int>{32, 32}));
    EXPECT_GT(lw.wk.f32_data.size(), 0u);
    EXPECT_EQ(lw.wk.shape, (std::vector<int>{32, 32}));
    EXPECT_GT(lw.wv.f32_data.size(), 0u);
    EXPECT_EQ(lw.wv.shape, (std::vector<int>{32, 32}));
    EXPECT_GT(lw.wo.f32_data.size(), 0u);
    EXPECT_EQ(lw.wo.shape, (std::vector<int>{32, 32}));

    EXPECT_GT(lw.ffn_norm.data.size(), 0u);
    EXPECT_EQ(lw.ffn_norm.shape[0], 32);
    EXPECT_GT(lw.w_gate.f32_data.size(), 0u);
    EXPECT_EQ(lw.w_gate.shape, (std::vector<int>{86, 32}));
    EXPECT_GT(lw.w_up.f32_data.size(), 0u);
    EXPECT_EQ(lw.w_up.shape, (std::vector<int>{86, 32}));
    EXPECT_GT(lw.w_down.f32_data.size(), 0u);
    EXPECT_EQ(lw.w_down.shape, (std::vector<int>{32, 86}));
  }

  EXPECT_GT(model.final_norm.data.size(), 0u);
  EXPECT_EQ(model.final_norm.shape[0], 32);

  EXPECT_GT(model.lm_head.f32_data.size(), 0u);
  ASSERT_EQ(model.lm_head.shape.size(), 2u);
  EXPECT_EQ(model.lm_head.shape[0], 128);
  EXPECT_EQ(model.lm_head.shape[1], 32);
}

// ---------------------------------------------------------------------------
// 合成的 GGUF 模型必须能进正常的前向路径
// ---------------------------------------------------------------------------
TEST(GgufLoader, FixtureEntersForwardPath) {
  std::filesystem::path path =
      dqm::test::WriteTinyLlamaGgufFixture("tiny_loader_forward.gguf");
  dqm::MiniLlamaModel model = dqm::LoadGgufModel(path.string());
  std::filesystem::remove(path);

  ASSERT_TRUE(model.loaded) << model.load_error;
  dqm::MiniLlamaContext ctx(&model);
  ctx.pos = 0;
  dqm::Tensor logits = dqm::ForwardToken(ctx, model, 1);
  EXPECT_EQ(logits.shape, (std::vector<int>{128}));
}

// ---------------------------------------------------------------------------
// 有真模型时顺带跑一遍大文件（dqm 里通常没有 -> 跳过）
// ---------------------------------------------------------------------------
TEST(GgufLoader, LoadsRealModelWhenAvailable) {
  const std::string path =
      TestDataPath("models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf");
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "real model not present in dqm";
  }

  dqm::MiniLlamaModel model = dqm::LoadGgufModel(path);
  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  EXPECT_EQ(model.config.vocab_size, 151936);
  EXPECT_EQ(model.config.dim, 896);
  EXPECT_EQ(model.config.n_layers, 24);
  EXPECT_EQ(model.config.n_heads, 14);
  EXPECT_EQ(model.config.n_kv_heads, 2);
  EXPECT_EQ(model.config.head_dim, 64);
  EXPECT_EQ(model.config.hidden_dim, 4864);
  EXPECT_EQ(model.config.max_seq_len, 32768);
  EXPECT_NEAR(model.config.rms_norm_eps, 1e-6f, 1e-7f);
}

// ---------------------------------------------------------------------------
// 错误路径
// ---------------------------------------------------------------------------
TEST(GgufLoader, MissingFile) {
  dqm::MiniLlamaModel model =
      dqm::LoadGgufModel(TestDataPath("models/chat/nonexistent.gguf"));
  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

TEST(GgufLoader, InvalidFile) {
  dqm::MiniLlamaModel model =
      dqm::LoadGgufModel(TestDataPath("models/tiny/model.json"));
  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

TEST(GgufLoader, RejectsIncompleteConfig) {
  // test.gguf 没有 config 类 metadata -> BuildConfig 校验应失败。
  dqm::MiniLlamaModel model =
      dqm::LoadGgufModel(TestDataPath("models/tiny/test.gguf"));
  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}
