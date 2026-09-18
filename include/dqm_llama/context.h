#pragma once

#include <vector>

#include "dqm_llama/kv_cache.h"
#include "dqm_llama/model.h"

namespace dqm {

// 一次推理会话的全部「可变状态」。
// 模型（MiniLlamaModel）是不变、可共享的部分；跑到哪了 / 缓存了哪些 K/V
// 这些每次会话独有的东西装在这里。
//
// 参考的 cuda_kv_cache 成员已砍（dqm 无 CUDA）。
struct MiniLlamaContext {
  const MiniLlamaModel* model = nullptr;  // 裸指针、非所有者
  KvCache kv_cache;
  int pos = 0;  // 当前写到第几个位置
  std::vector<int> token_history;  // 该 context 见过的全部 token（含 prefill）

  // 统计
  int n_prefill_tokens = 0;
  int n_decode_tokens = 0;

  MiniLlamaContext() = default;
  MiniLlamaContext(const MiniLlamaContext&) = delete;
  MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
  MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
  MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;
  explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}  // namespace dqm
