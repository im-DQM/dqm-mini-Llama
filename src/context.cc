#include "dqm_llama/context.h"

#include <stdexcept>

namespace dqm {

MiniLlamaContext::MiniLlamaContext(const MiniLlamaModel* model)
    : model(model), pos(0) {
  if (model != nullptr) {
    const auto& c = model->config;
    kv_cache = KvCache(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim);
  }
}

}  // namespace dqm
