#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "dqm_llama/context.h"
#include "dqm_llama/kv_cache.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"

using dqm::MiniLlamaContext;
using dqm::MiniLlamaModel;

namespace {

// 构造函数只读 config、不碰权重，所以模型只要有 config 就能拿去建 context。
MiniLlamaModel MakeModelWithConfig(int n_layers, int max_seq_len, int n_kv_heads,
                                   int head_dim) {
  MiniLlamaModel m;
  m.config.n_layers = n_layers;
  m.config.max_seq_len = max_seq_len;
  m.config.n_kv_heads = n_kv_heads;
  m.config.head_dim = head_dim;
  return m;
}

}  // namespace

// ===========================================================================
// 默认状态
// ===========================================================================

TEST(MiniLlamaContext, DefaultConstructedIsEmpty) {
  const MiniLlamaContext ctx;

  EXPECT_EQ(ctx.model, nullptr);
  EXPECT_EQ(ctx.pos, 0);
  EXPECT_TRUE(ctx.token_history.empty());
  EXPECT_EQ(ctx.n_prefill_tokens, 0);
  EXPECT_EQ(ctx.n_decode_tokens, 0);
  EXPECT_TRUE(ctx.kv_cache.keys.data.empty());
  EXPECT_TRUE(ctx.kv_cache.values.data.empty());
}

// ===========================================================================
// 构造函数：nullptr 模型
// ===========================================================================

// 契约：model 为空 -> 不建 cache、不抛
TEST(MiniLlamaContext, NullModelBuildsNoCache) {
  const MiniLlamaContext ctx(nullptr);

  EXPECT_EQ(ctx.model, nullptr);
  EXPECT_EQ(ctx.pos, 0);
  EXPECT_TRUE(ctx.kv_cache.keys.shape.empty());
  EXPECT_TRUE(ctx.kv_cache.keys.data.empty());
  EXPECT_TRUE(ctx.kv_cache.values.data.empty());
}

// ===========================================================================
// 构造函数：cache 尺寸来自 ModelConfig
// ===========================================================================

// 4 个字段 -> KvCache 的 4 个维度，顺序是
// [n_layers, max_seq_len, n_kv_heads, head_dim]
TEST(MiniLlamaContext, CacheShapeComesFromConfig) {
  MiniLlamaModel small = MakeModelWithConfig(1, 4, 1, 2);
  MiniLlamaModel other = MakeModelWithConfig(3, 16, 2, 5);

  const MiniLlamaContext a(&small);
  const MiniLlamaContext b(&other);

  EXPECT_EQ(a.kv_cache.keys.shape, (std::vector<int>{1, 4, 1, 2}));
  EXPECT_EQ(a.kv_cache.values.shape, (std::vector<int>{1, 4, 1, 2}));
  EXPECT_EQ(a.kv_cache.keys.size(), 8u);

  EXPECT_EQ(b.kv_cache.keys.shape, (std::vector<int>{3, 16, 2, 5}));
  EXPECT_EQ(b.kv_cache.keys.size(), 480u);
}

TEST(MiniLlamaContext, FreshCacheIsZeroFilled) {
  MiniLlamaModel model = MakeModelWithConfig(1, 3, 1, 2);
  const MiniLlamaContext ctx(&model);

  ASSERT_EQ(ctx.kv_cache.keys.size(), 6u);
  ASSERT_EQ(ctx.kv_cache.values.size(), 6u);
  for (size_t i = 0; i < ctx.kv_cache.keys.size(); ++i) {
    EXPECT_NEAR(ctx.kv_cache.keys.data[i], 0.0f, 1e-6f);
    EXPECT_NEAR(ctx.kv_cache.values.data[i], 0.0f, 1e-6f);
  }
}

// ===========================================================================
// 拷贝 / 移动
// ===========================================================================

// 拷贝删掉、移动保留 —— 编译期钉住（写错的话这段根本不编译）
TEST(MiniLlamaContext, CopyDeletedMoveNoexcept) {
  static_assert(!std::is_copy_constructible_v<MiniLlamaContext>,
                "拷贝构造必须被删除");
  static_assert(!std::is_copy_assignable_v<MiniLlamaContext>,
                "拷贝赋值必须被删除");
  static_assert(std::is_move_constructible_v<MiniLlamaContext>,
                "必须可移动构造");
  static_assert(std::is_move_assignable_v<MiniLlamaContext>,
                "必须可移动赋值");
  static_assert(std::is_nothrow_move_constructible_v<MiniLlamaContext>,
                "移动构造必须 noexcept");
  static_assert(std::is_nothrow_move_assignable_v<MiniLlamaContext>,
                "移动赋值必须 noexcept");
  SUCCEED();
}

// main.cc 的前缀复用路径就是这句：扔一个旧 context、换一个新的进来
TEST(MiniLlamaContext, MoveAssignRebuildsCache) {
  MiniLlamaModel model = MakeModelWithConfig(2, 8, 3, 4);

  MiniLlamaContext ctx;
  ASSERT_TRUE(ctx.kv_cache.keys.shape.empty());

  ctx = MiniLlamaContext(&model);

  EXPECT_EQ(ctx.model, &model);
  EXPECT_EQ(ctx.pos, 0);
  EXPECT_EQ(ctx.kv_cache.keys.shape, (std::vector<int>{2, 8, 3, 4}));
  EXPECT_EQ(ctx.kv_cache.keys.size(), 192u);
}

TEST(MiniLlamaContext, MoveConstructTransfersState) {
  MiniLlamaModel model = MakeModelWithConfig(2, 8, 3, 4);
  MiniLlamaContext a(&model);
  a.token_history = {1, 2, 3};

  MiniLlamaContext b(std::move(a));

  EXPECT_EQ(b.model, &model);
  EXPECT_EQ(b.kv_cache.keys.shape, (std::vector<int>{2, 8, 3, 4}));
  EXPECT_EQ(b.kv_cache.keys.size(), 192u);
  EXPECT_EQ(b.token_history, (std::vector<int>{1, 2, 3}));
}
