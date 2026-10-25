#pragma once

#include <string>

#include "dqm_llama/model.h"

namespace dqm {

// 从 GGUF 文件加载模型。
// 失败返回 loaded=false 的模型，原因在 load_error 里（不抛异常）。
MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}  // namespace dqm
