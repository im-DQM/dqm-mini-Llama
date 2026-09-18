#pragma once

#include "dqm_llama/batch.h"
#include "dqm_llama/context.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tensor.h"

namespace dqm {

// 单个 token 的前向：token id 进，logits 出。
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    int token);

// 一批 token 的前向：内部顺序处理，只返回最后一个 token 的 logits。
// 用来把 prefill（多 token）和 decode（单 token）统一成一条路径。
// 注意：实现在 batch.cc，不在 forward.cc。
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace dqm
