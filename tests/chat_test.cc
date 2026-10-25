#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "dqm_llama/chat.h"

using dqm::ChatMessage;
using dqm::ChatSession;

// ===========================================================================
// 默认状态
// ===========================================================================

TEST(ChatSession, DefaultConstructedIsEmpty) {
  const ChatSession session;

  EXPECT_TRUE(session.messages.empty());
  EXPECT_TRUE(session.token_history.empty());
  EXPECT_TRUE(session.prefix_cache.empty());

  EXPECT_FLOAT_EQ(session.sampling_params.temperature, 0.0f);
  EXPECT_EQ(session.sampling_params.top_k, 0);
  EXPECT_EQ(session.sampling_params.seed, 0u);

  EXPECT_EQ(session.total_prompt_tokens, 0);
  EXPECT_EQ(session.total_generated_tokens, 0);
  EXPECT_DOUBLE_EQ(session.total_time_ms, 0.0);
}

// ===========================================================================
// ChatMessage
// ===========================================================================

// 契约：聚合类型 —— 没有构造函数，直接花括号初始化
TEST(ChatMessage, AggregateInitialization) {
  const ChatMessage msg{"user", "你好"};

  EXPECT_EQ(msg.role, "user");
  EXPECT_EQ(msg.content, "你好");
}

// ===========================================================================
// AddMessage
// ===========================================================================

TEST(ChatSession, AddMessageAppendsInOrder) {
  ChatSession session;

  session.AddMessage("system", "You are helpful.");
  session.AddMessage("user", "hi");

  ASSERT_EQ(session.messages.size(), 2u);
  EXPECT_EQ(session.messages[0].role, "system");
  EXPECT_EQ(session.messages[0].content, "You are helpful.");
  EXPECT_EQ(session.messages[1].role, "user");
  EXPECT_EQ(session.messages[1].content, "hi");
}

// 契约：不做校验 —— role 传什么都原样收下
TEST(ChatSession, AddMessageDoesNotValidateRole) {
  ChatSession session;

  session.AddMessage("User", "capitalized");
  session.AddMessage("", "empty role");

  ASSERT_EQ(session.messages.size(), 2u);
  EXPECT_EQ(session.messages[0].role, "User");
  EXPECT_EQ(session.messages[1].role, "");
}

// ===========================================================================
// RecordTurn
// ===========================================================================

// 契约：三个统计都是「累加」，不是赋值
TEST(ChatSession, RecordTurnAccumulatesAcrossCalls) {
  ChatSession session;

  session.RecordTurn(10, 4, 12.5);
  session.RecordTurn(3, 7, 0.5);
  session.RecordTurn(0, 0, 1.0);

  EXPECT_EQ(session.total_prompt_tokens, 13);
  EXPECT_EQ(session.total_generated_tokens, 11);
  EXPECT_DOUBLE_EQ(session.total_time_ms, 14.0);
}

// ===========================================================================
// SetTokenHistory / AppendToken
// ===========================================================================

// 契约：整体替换（不是追加）
TEST(ChatSession, SetTokenHistoryReplacesEntireHistory) {
  ChatSession session;

  session.SetTokenHistory({1, 2, 3});
  session.SetTokenHistory({7, 8});

  EXPECT_EQ(session.token_history, (std::vector<int>{7, 8}));
}

TEST(ChatSession, SetTokenHistoryWithEmptyClearsHistory) {
  ChatSession session;

  session.SetTokenHistory({1, 2, 3});
  session.SetTokenHistory({});

  EXPECT_TRUE(session.token_history.empty());
}

TEST(ChatSession, AppendTokenAddsOneAtATime) {
  ChatSession session;

  session.AppendToken(5);
  session.AppendToken(6);
  session.AppendToken(7);

  EXPECT_EQ(session.token_history, (std::vector<int>{5, 6, 7}));
}

// prefill 用 SetTokenHistory 重置整条，decode 用 AppendToken 往后接 —— 两者配合
TEST(ChatSession, SetThenAppendBuildsFullSequence) {
  ChatSession session;

  session.SetTokenHistory({100, 101});
  session.AppendToken(102);
  session.AppendToken(103);

  EXPECT_EQ(session.token_history, (std::vector<int>{100, 101, 102, 103}));
}

// ===========================================================================
// RecordPrefix / LongestCachedPrefix
// ===========================================================================

TEST(ChatSession, LongestCachedPrefixOnEmptyCacheIsZero) {
  const ChatSession session;

  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0u);
}

TEST(ChatSession, LongestCachedPrefixOfEmptyQueryIsZero) {
  ChatSession session;
  session.RecordPrefix({1, 2, 3});

  EXPECT_EQ(session.LongestCachedPrefix({}), 0u);
}

// 登记过的序列，查它自己 -> 全长
TEST(ChatSession, RecordPrefixThenQueryItReturnsFullLength) {
  ChatSession session;

  session.RecordPrefix({1, 2, 3, 4});

  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3, 4}), 4u);
}

// 查询比登记的长 -> 只数到公共前缀
TEST(ChatSession, QueryLongerThanRecordedReturnsCommonPrefixLength) {
  ChatSession session;

  session.RecordPrefix({1, 2, 3, 4});

  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3, 4, 5, 6}), 4u);
}

// 中途分叉 -> 数到分叉前一格
TEST(ChatSession, DivergentQueryReturnsLengthBeforeDivergence) {
  ChatSession session;

  session.RecordPrefix({1, 2, 3, 4});

  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 9}), 2u);
}

// 第一个 token 就不同 -> 0
TEST(ChatSession, QuerySharingNoPrefixReturnsZero) {
  ChatSession session;

  session.RecordPrefix({1, 2, 3});

  EXPECT_EQ(session.LongestCachedPrefix({9, 2, 3}), 0u);
}

// RunChat 每轮登记的是「整条 token_history」，所以树里是一条比一条长的序列
TEST(ChatSession, RecordingLongerSequenceExtendsReuse) {
  ChatSession session;

  session.RecordPrefix({1, 2});
  session.RecordPrefix({1, 2, 3, 4});

  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3, 4, 5}), 4u);
}

// RunChat 每轮都问一次「token_history 我算过没」，问完还要接着用这个 cache
TEST(ChatSession, LongestCachedPrefixIsNonMutating) {
  ChatSession session;
  session.RecordPrefix({1, 2, 3});

  const size_t first = session.LongestCachedPrefix({1, 2, 3, 4});
  const size_t second = session.LongestCachedPrefix({1, 2, 3, 4});

  EXPECT_EQ(first, 3u);
  EXPECT_EQ(second, 3u);
  EXPECT_FALSE(session.prefix_cache.empty());
}

// ===========================================================================
// Clear
// ===========================================================================

TEST(ChatSession, ClearResetsMessagesHistoryCacheAndStats) {
  ChatSession session;

  session.AddMessage("user", "hi");
  session.SetTokenHistory({1, 2, 3});
  session.RecordPrefix({1, 2, 3});
  session.RecordTurn(3, 5, 8.0);

  session.Clear();

  EXPECT_TRUE(session.messages.empty());
  EXPECT_TRUE(session.token_history.empty());
  EXPECT_TRUE(session.prefix_cache.empty());
  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0u);
  EXPECT_EQ(session.total_prompt_tokens, 0);
  EXPECT_EQ(session.total_generated_tokens, 0);
  EXPECT_DOUBLE_EQ(session.total_time_ms, 0.0);
}

// 契约：Clear() 有意保留 sampling_params —— 「忘掉对话」不等于「重置参数」
TEST(ChatSession, ClearKeepsSamplingParams) {
  ChatSession session;
  session.sampling_params.temperature = 0.7f;
  session.sampling_params.top_k = 5;
  session.sampling_params.seed = 42u;

  session.Clear();

  EXPECT_FLOAT_EQ(session.sampling_params.temperature, 0.7f);
  EXPECT_EQ(session.sampling_params.top_k, 5);
  EXPECT_EQ(session.sampling_params.seed, 42u);
}

// 推论：Clear() 之后的对象 != 默认构造态（参数还在）
TEST(ChatSession, ClearedSessionDiffersFromDefaultConstructed) {
  ChatSession session;
  session.sampling_params.temperature = 0.7f;

  session.Clear();

  const ChatSession fresh;
  EXPECT_NE(session.sampling_params.temperature,
            fresh.sampling_params.temperature);
}

// ===========================================================================
// 类型契约
// ===========================================================================

// 拷贝删掉（传染自 RadixTree）、移动保留 —— 编译期钉住
TEST(ChatSession, CopyDeletedMoveNoexcept) {
  static_assert(!std::is_copy_constructible_v<ChatSession>,
                "拷贝构造必须被删除");
  static_assert(!std::is_copy_assignable_v<ChatSession>, "拷贝赋值必须被删除");
  static_assert(std::is_move_constructible_v<ChatSession>, "必须可移动构造");
  static_assert(std::is_move_assignable_v<ChatSession>, "必须可移动赋值");
  static_assert(std::is_nothrow_move_constructible_v<ChatSession>,
                "移动构造必须 noexcept");
  static_assert(std::is_nothrow_move_assignable_v<ChatSession>,
                "移动赋值必须 noexcept");
  SUCCEED();
}
