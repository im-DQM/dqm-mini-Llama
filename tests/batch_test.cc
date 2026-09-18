#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "dqm_llama/batch.h"
#include "dqm_llama/context.h"
#include "dqm_llama/forward.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"
#include "test_tiny_model.h"

using dqm::ForwardBatch;
using dqm::ForwardToken;
using dqm::MiniBatch;
using dqm::MiniLlamaContext;
using dqm::MiniLlamaModel;
using dqm::Tensor;
using dqm::test::MakeTinyModel;
using dqm::test::NearlyEqual;

// ===========================================================================
// 默认构造
// ===========================================================================

TEST(MiniBatch, DefaultConstructedIsEmpty) {
  const MiniBatch batch;
  EXPECT_EQ(batch.num_tokens(), 0);
  EXPECT_TRUE(batch.tokens.empty());
  EXPECT_TRUE(batch.positions.empty());
}

// ===========================================================================
// num_tokens
// ===========================================================================

TEST(MiniBatch, NumTokensFollowsTokensSize) {
  MiniBatch batch;
  batch.tokens = {1, 2, 3, 4};
  batch.positions = {0, 1, 2, 3};
  EXPECT_EQ(batch.num_tokens(), 4);
}

// ===========================================================================
// Single —— decode 用
// ===========================================================================

TEST(MiniBatch, SingleHasExactlyOneToken) {
  const MiniBatch batch = MiniBatch::Single(7, 100);
  EXPECT_EQ(batch.num_tokens(), 1);
  ASSERT_EQ(batch.tokens.size(), 1u);
  ASSERT_EQ(batch.positions.size(), 1u);
  EXPECT_EQ(batch.tokens[0], 7);
  EXPECT_EQ(batch.positions[0], 100);
}

// token 和 pos 取差别很大的值：参数顺序写反就会露馅。
TEST(MiniBatch, SingleKeepsTokenAndPositionDistinct) {
  const MiniBatch a = MiniBatch::Single(7, 100);
  const MiniBatch b = MiniBatch::Single(100, 7);

  EXPECT_EQ(a.tokens[0], 7);
  EXPECT_EQ(a.positions[0], 100);
  EXPECT_EQ(b.tokens[0], 100);
  EXPECT_EQ(b.positions[0], 7);
}

TEST(MiniBatch, SingleAcceptsZeroForBoth) {
  const MiniBatch batch = MiniBatch::Single(0, 0);
  EXPECT_EQ(batch.num_tokens(), 1);
  EXPECT_EQ(batch.tokens[0], 0);
  EXPECT_EQ(batch.positions[0], 0);
}

// ===========================================================================
// FromTokens —— prefill 用
// ===========================================================================

TEST(MiniBatch, FromTokensAssignsConsecutivePositionsFromZero) {
  const MiniBatch batch = MiniBatch::FromTokens({10, 20, 30});
  EXPECT_EQ(batch.num_tokens(), 3);
  EXPECT_EQ(batch.tokens, (std::vector<int>{10, 20, 30}));
  EXPECT_EQ(batch.positions, (std::vector<int>{0, 1, 2}));
}

// 前缀复用的场景：前面 prefix_len 个 token 的 KV 已经算过，位置不从 0 起。
TEST(MiniBatch, FromTokensHonorsStartPos) {
  const MiniBatch batch = MiniBatch::FromTokens({10, 20, 30}, 5);
  EXPECT_EQ(batch.tokens, (std::vector<int>{10, 20, 30}));
  EXPECT_EQ(batch.positions, (std::vector<int>{5, 6, 7}));
}

// 空输入是合法的 —— 不抛，返回空 batch。（ForwardBatch 才会拒绝空 batch。）
TEST(MiniBatch, FromTokensOnEmptyVectorIsEmpty) {
  const MiniBatch batch = MiniBatch::FromTokens({}, 7);
  EXPECT_EQ(batch.num_tokens(), 0);
  EXPECT_TRUE(batch.tokens.empty());
  EXPECT_TRUE(batch.positions.empty());
}

// tokens 是拷贝，不是引用调用方的 vector。
TEST(MiniBatch, FromTokensCopiesTokensInsteadOfAliasing) {
  std::vector<int> source = {1, 2, 3};
  const MiniBatch batch = MiniBatch::FromTokens(source, 0);

  source[0] = 99;

  EXPECT_EQ(batch.tokens[0], 1);
}

// ===========================================================================
// 不变量：两个数组必须等长
// 结构体自己不保证，但两个工厂都必须维持住。
// ===========================================================================

TEST(MiniBatch, FactoriesKeepTokensAndPositionsSameLength) {
  const MiniBatch single = MiniBatch::Single(3, 4);
  const MiniBatch multi = MiniBatch::FromTokens({1, 2, 3, 4, 5}, 2);

  EXPECT_EQ(single.tokens.size(), single.positions.size());
  EXPECT_EQ(multi.tokens.size(), multi.positions.size());
}

// ===========================================================================
// ForwardBatch 的两条入参校验
// —— 它们跑在 ForwardToken 之前，所以不需要模型真的能推理。
// ===========================================================================

TEST(ForwardBatch, RejectsEmptyBatch) {
  const MiniLlamaModel model;
  MiniLlamaContext ctx(nullptr);
  const MiniBatch empty;

  EXPECT_THROW(ForwardBatch(ctx, model, empty), std::runtime_error);
}

TEST(ForwardBatch, RejectsMismatchedTokenAndPositionCount) {
  const MiniLlamaModel model;
  MiniLlamaContext ctx(nullptr);

  MiniBatch bad;
  bad.tokens = {1, 2, 3};
  bad.positions = {0};

  EXPECT_THROW(ForwardBatch(ctx, model, bad), std::runtime_error);
}

// ===========================================================================
// ForwardBatch 主路径
//
// 用共享夹具手工造模型（不经过 loader）—— 这四条不依赖 models/tiny/ 下的文件。
// ===========================================================================

// 「统一 prefill 和 decode」这个契约：整批跑一遍，必须和逐个 ForwardToken 跑逐位相同。
// 位置推进写错（比如提到循环外）立刻在这里现形。
TEST(ForwardBatch, PrefillMatchesIndividualForward) {
  MiniLlamaModel model = MakeTinyModel();
  const std::vector<int> tokens = {1, 104, 101, 108, 108, 111};

  // 逐个来：位置自己推进
  MiniLlamaContext manual(&model);
  Tensor manual_logits;
  for (size_t i = 0; i < tokens.size(); ++i) {
    manual.pos = static_cast<int>(i);
    manual_logits = ForwardToken(manual, model, tokens[i]);
  }

  // 整批来：位置由 MiniBatch 带进去
  MiniLlamaContext batched(&model);
  const Tensor batch_logits =
      ForwardBatch(batched, model, MiniBatch::FromTokens(tokens, 0));

  ASSERT_EQ(batch_logits.shape, manual_logits.shape);
  EXPECT_TRUE(NearlyEqual(batch_logits, manual_logits));
}

// decode 路径：单 token 的 batch。位置不从 0 起也要对（decode 每步都是这样）。
TEST(ForwardBatch, SingleMatchesIndividualForward) {
  MiniLlamaModel model = MakeTinyModel();

  MiniLlamaContext manual(&model);
  manual.pos = 3;
  const Tensor manual_logits = ForwardToken(manual, model, 42);

  MiniLlamaContext batched(&model);
  const Tensor batch_logits =
      ForwardBatch(batched, model, MiniBatch::Single(42, 3));

  ASSERT_EQ(batch_logits.shape, manual_logits.shape);
  EXPECT_TRUE(NearlyEqual(batch_logits, manual_logits));
}

// ForwardBatch 对 ctx 的两处写入：token_history 追加、pos 停在最后一个位置。
// 注意 ForwardToken 自己两样都不碰 —— 所以这里是它的唯一归宿。
TEST(ForwardBatch, UpdatesContextPositionAndHistory) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);

  const std::vector<int> tokens = {1, 104, 101};
  ForwardBatch(ctx, model, MiniBatch::FromTokens(tokens, 0));

  EXPECT_EQ(ctx.token_history, tokens);
  EXPECT_EQ(ctx.pos, 2);  // 最后一个 token 的位置（0 起）
}

// prefill 之后再 decode：历史累加，不是被清空或覆盖。多轮对话就靠这条。
TEST(ForwardBatch, AppendsDecodeHistoryAcrossCalls) {
  MiniLlamaModel model = MakeTinyModel();
  MiniLlamaContext ctx(&model);

  ForwardBatch(ctx, model, MiniBatch::FromTokens({1, 104}, 0));
  ForwardBatch(ctx, model, MiniBatch::Single(101, 2));

  EXPECT_EQ(ctx.token_history, (std::vector<int>{1, 104, 101}));
  EXPECT_EQ(ctx.pos, 2);
}
