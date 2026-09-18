#include "dqm_llama/batch.h"

#include <stdexcept>

#include "dqm_llama/forward.h"

namespace dqm {

MiniBatch MiniBatch::Single(int token, int pos) {
  MiniBatch mb;
  mb.tokens.push_back(token);
  mb.positions.push_back(pos);
  return mb;
}

MiniBatch MiniBatch::FromTokens(const std::vector<int>& toks, int start_pos) {
  MiniBatch mb;
  mb.tokens = toks;
  for (size_t i = 0; i < toks.size(); i ++) {
    mb.positions.push_back(start_pos);
    start_pos ++;
  }
  return mb;
}

Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch) {
  if (batch.tokens.empty()) {
    throw std::runtime_error("ForwardBatch: empty token");
  }
  if (batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error("ForwardBatch: tokens and positions size mismatch");
  }
  Tensor logits;
  for (int i = 0; i < batch.num_tokens(); i ++) {
    ctx.pos = batch.positions[i];
    int token = batch.tokens[i];
    logits = ForwardToken(ctx, model, token);
    ctx.token_history.push_back(token);
  }
  return logits;
}

}  // namespace dqm
