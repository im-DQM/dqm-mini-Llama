#pragma once

#include <string>

#include "dqm_llama/chat.h"
#include "dqm_llama/model.h"
#include "dqm_llama/tokenizer.h"

namespace dqm {

// 交互式聊天的 I/O 薄壳：只会「打印」和「读一行」，不含任何流程控制。
// 无状态（没有成员变量），10 个方法全是 const。
class Terminal {
 public:
  Terminal() = default;

  // 用户输入提示符 "mini-llama > "（含 flush）。
  void PrintUserPrompt() const;

  // 从 stdin 读一行。EOF 时返回空串（与「用户敲空行」无法区分）。
  std::string ReadLine() const;

  // 助手回复前缀 "assistant  > "（中间两个空格，含 flush）。
  void PrintAssistantPrefix() const;

  // 打印一段文本。不换行、不 flush —— 供流式输出用。
  void PrintTokenText(const std::string& text) const;

  // 刷新 stdout。
  void Flush() const;

  // 输出一个换行。
  void NewLine() const;

  // 打印 /help 文本（硬编码）。
  void PrintHelp() const;

  // 打印会话统计（/stats）。
  void PrintStats(const ChatSession& session) const;

  // 打印当前采样参数（/params）。
  void PrintParams(const SamplingParams& params) const;

  // 通用打印 —— 自己补一个 '\n'。
  void PrintMessage(const std::string& msg) const;
};

}  // namespace dqm
