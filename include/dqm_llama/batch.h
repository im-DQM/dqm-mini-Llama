#pragma once

#include <vector>

namespace dqm {

// 一组「token + 各自的位置」。
// 教学版简化：每个 token 恰好对应一个位置（不支持下述那种复杂情形）。
//
// 名字虽然叫 batch，但这里的 "batch" 是「一起提交的一组 token」，
// 不是「批量并行计算」—— ForwardBatch 内部是逐个顺序处理的。
struct MiniBatch {
  std::vector<int> tokens;     // token id
  std::vector<int> positions;  // 每个 token 的位置

  int num_tokens() const { return static_cast<int>(tokens.size()); }

  // decode 用：单 token 的 batch
  static MiniBatch Single(int token, int pos);

  // prefill 用：整段序列，位置从 start_pos 开始逐个 +1
  static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);
};

}  // namespace dqm
