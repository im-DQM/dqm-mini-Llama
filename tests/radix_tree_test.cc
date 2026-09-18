#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "dqm_llama/radix_tree.h"

using dqm::RadixTree;

// ===========================================================================
// 类型契约（编译期）
// ===========================================================================

// 节点被 unique_ptr 独占持有，拷贝没有意义，必须禁掉。
static_assert(!std::is_copy_constructible_v<RadixTree>, "RadixTree 必须禁拷贝");
static_assert(!std::is_copy_assignable_v<RadixTree>, "RadixTree 必须禁拷贝赋值");

// 移动要可用，而且要 noexcept —— 否则 std::vector<RadixTree> 扩容时会退化成拷贝
// （而拷贝又被删了，直接编译不过）。
static_assert(std::is_move_constructible_v<RadixTree>, "RadixTree 必须可移动");
static_assert(std::is_move_assignable_v<RadixTree>, "RadixTree 必须可移动赋值");
static_assert(std::is_nothrow_move_constructible_v<RadixTree>,
              "RadixTree 移动构造必须 noexcept");
static_assert(std::is_nothrow_move_assignable_v<RadixTree>,
              "RadixTree 移动赋值必须 noexcept");

// ===========================================================================
// 空树
// ===========================================================================

TEST(RadixTree, NewTreeIsEmpty) {
  RadixTree tree;
  EXPECT_TRUE(tree.empty());
  EXPECT_EQ(tree.size(), 0u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 0u);
}

// ===========================================================================
// Insert / size —— size() 数的是「序列条数」，不是节点数
// ===========================================================================

TEST(RadixTree, InsertSingleSequenceCountsOne) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_FALSE(tree.empty());
  EXPECT_EQ(tree.size(), 1u);
}

TEST(RadixTree, InsertSameSequenceTwiceCountsOnce) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.size(), 1u);
}

TEST(RadixTree, InsertDistinctSequencesCountsEach) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({4, 5});
  tree.Insert({9});
  EXPECT_EQ(tree.size(), 3u);
}

TEST(RadixTree, InsertSharedPrefixCountsEach) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 4});
  EXPECT_EQ(tree.size(), 2u);
}

// 中间节点也可以是 terminal：{1,2,3} 是 {1,2,3,4,5} 的前缀，两条各算一条。
TEST(RadixTree, InsertExtensionOfExistingSequenceCountsSeparately) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 3, 4, 5});
  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4, 5}), 5u);
}

TEST(RadixTree, InsertPrefixOfExistingSequenceCountsSeparately) {
  RadixTree tree;
  tree.Insert({1, 2, 3, 4, 5});
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4, 5}), 5u);
}

// 空序列也是一条被存过的序列。
TEST(RadixTree, InsertEmptySequenceIsCounted) {
  RadixTree tree;
  tree.Insert({});
  EXPECT_FALSE(tree.empty());
  EXPECT_EQ(tree.size(), 1u);
}

TEST(RadixTree, InsertEmptySequenceTwiceCountsOnce) {
  RadixTree tree;
  tree.Insert({});
  tree.Insert({});
  EXPECT_EQ(tree.size(), 1u);
}

// ===========================================================================
// Clear
// ===========================================================================

TEST(RadixTree, ClearResetsToEmpty) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({4, 5});
  ASSERT_EQ(tree.size(), 2u);

  tree.Clear();

  EXPECT_TRUE(tree.empty());
  EXPECT_EQ(tree.size(), 0u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 0u);
}

// Clear 也要把根重置 —— 否则 Insert({}) 留下的 terminal 标记会跟着活下来。
TEST(RadixTree, ClearResetsRootTerminalFlag) {
  RadixTree tree;
  tree.Insert({});
  ASSERT_EQ(tree.size(), 1u);

  tree.Clear();
  ASSERT_TRUE(tree.empty());

  // 清空之后再插空序列，应该重新算一条（而不是被残留的 terminal 吞掉）
  tree.Insert({});
  EXPECT_EQ(tree.size(), 1u);
}

TEST(RadixTree, ClearThenReuse) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Clear();
  tree.Insert({7, 8});

  EXPECT_EQ(tree.size(), 1u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 0u);
  EXPECT_EQ(tree.LongestPrefix({7, 8, 9}), 2u);
}

// ===========================================================================
// LongestPrefix
// ===========================================================================

TEST(RadixTree, LongestPrefixEmptyQueryIsZero) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.LongestPrefix({}), 0u);
}

TEST(RadixTree, LongestPrefixExactMatch) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
}

TEST(RadixTree, LongestPrefixStopsAtMismatch) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.LongestPrefix({1, 2, 9}), 2u);
}

TEST(RadixTree, LongestPrefixQueryLongerThanStored) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 7, 7}), 3u);
}

TEST(RadixTree, LongestPrefixNoSharedFirstTokenIsZero) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.LongestPrefix({9, 9}), 0u);
}

// 匹配到的前缀「是某条已插入序列的前缀」即可 —— 不要求它自己也是 terminal。
TEST(RadixTree, LongestPrefixNeedNotBeTerminal) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  EXPECT_EQ(tree.LongestPrefix({1, 2}), 2u);
}

TEST(RadixTree, LongestPrefixPicksLongestAcrossBranches) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 3, 4});
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4, 9}), 4u);
}

// ===========================================================================
// 节点分裂（长边被从中间劈开）
// ===========================================================================

TEST(RadixTree, SplitsEdgeWhenInsertingDivergentSequence) {
  RadixTree tree;
  tree.Insert({1, 2, 3, 4, 5});  // 一条 5-token 的边
  tree.Insert({1, 2, 9});        // 在第 3 个 token 处岔开 → 必须劈开

  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4, 5}), 5u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 9}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2}), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 7}), 2u);
}

TEST(RadixTree, SplitsEdgeAndMarksIntermediateTerminal) {
  RadixTree tree;
  tree.Insert({1, 2, 3, 4, 5});
  tree.Insert({1, 2});  // 劈开后，劈出来的那个节点自己就是终点

  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 2}), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4, 5}), 5u);
}

TEST(RadixTree, SplitsTwiceOnSameBranch) {
  RadixTree tree;
  tree.Insert({1, 2, 3, 4});
  tree.Insert({1, 2, 5});
  tree.Insert({1, 6});

  EXPECT_EQ(tree.size(), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 4}), 4u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 5}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 6}), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 9}), 1u);
}

TEST(RadixTree, ManyBranchesUnderCommonRoot) {
  RadixTree tree;
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 4});
  tree.Insert({1, 5});
  tree.Insert({1, 5, 6});
  tree.Insert({2, 2});

  EXPECT_EQ(tree.size(), 5u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 4}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 5}), 2u);
  EXPECT_EQ(tree.LongestPrefix({1, 5, 6, 7}), 3u);
  EXPECT_EQ(tree.LongestPrefix({2, 2}), 2u);
  EXPECT_EQ(tree.LongestPrefix({2}), 1u);
  EXPECT_EQ(tree.LongestPrefix({3}), 0u);
}

// 长序列 / 重复 token 也应当只是普通数据，没有特殊含义。
TEST(RadixTree, HandlesRepeatedTokens) {
  RadixTree tree;
  tree.Insert({7, 7, 7});
  tree.Insert({7, 7, 8});

  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.LongestPrefix({7, 7, 7}), 3u);
  EXPECT_EQ(tree.LongestPrefix({7, 7, 9}), 2u);
}

// ===========================================================================
// 移动语义
// ===========================================================================

TEST(RadixTree, MoveConstructPreservesQueries) {
  RadixTree src;
  src.Insert({1, 2, 3});
  src.Insert({4, 5});

  RadixTree dst(std::move(src));  // 移动后不要断言 src 的内容（状态未指定）

  EXPECT_EQ(dst.size(), 2u);
  EXPECT_EQ(dst.LongestPrefix({1, 2, 3}), 3u);
  EXPECT_EQ(dst.LongestPrefix({4, 5, 6}), 2u);
}

TEST(RadixTree, MoveAssignPreservesQueries) {
  RadixTree src;
  src.Insert({1, 2, 3});

  RadixTree dst;
  dst.Insert({9, 9, 9, 9});
  dst = std::move(src);

  EXPECT_EQ(dst.size(), 1u);
  EXPECT_EQ(dst.LongestPrefix({1, 2, 3}), 3u);
  EXPECT_EQ(dst.LongestPrefix({9, 9, 9, 9}), 0u);
}
