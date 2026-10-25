#pragma once

#include <string>
#include <vector>

#include "dqm_llama/chat.h"

namespace dqm {

// 把 ChatMessage 列表拼成一段纯文本 prompt，交给 tokenizer 编码后喂给模型。
// 它是「文本层」的最后一公里 —— 只做「消息 → 文本」，反方向是 tokenizer 的活。
//
// 三条路（按 chat_template_ 的内容分发）：
//   plain   —— 默认；逐条 "Role: content\n"，末尾补 "Assistant:"
//   Jinja2  —— 把模板当程序编译并执行（各家模型官方模板走这条）
//   qwen2   —— 内建 Qwen2 拼法（<|im_start|> / <|im_end|>）
class PromptBuilder {
 public:
  PromptBuilder() = default;

  // 设置模板。纯赋值 —— 不校验、不解析（解析推迟到 Build）。
  void SetChatTemplate(const std::string& template_str);

  // 主入口：按 chat_template_ 的内容四路分发。
  std::string Build(const std::vector<ChatMessage>& messages) const;

 private:
  // 模板原文：Jinja2 字符串，或速记名 "qwen2"，或空（= plain）
  std::string chat_template_;

  std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
  std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

// 从 GGUF 读聊天模板：有 tokenizer.chat_template 返回原文；否则若
// general.architecture == "qwen2" 返回速记名 "qwen2"；都没有返回 ""。
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);

}  // namespace dqm
