#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "dqm_llama/radix_tree.h"
#include "dqm_llama/sampler.h"

namespace dqm {

// 一条对话消息。role 取 "system" / "user" / "assistant"。
struct ChatMessage {
  std::string role;
  std::string content;
};

// 一次交互式聊天会话的全部状态。
//
// 推理本身不读这个对象 —— 把它整个删掉，forward / sampler 一步都不会少算。
// 它的用途是给 RunChat 循环一个「记住上一轮发生了什么」的抽屉。
class ChatSession {
 public:
  std::vector<ChatMessage> messages;
  std::vector<int> token_history;
  RadixTree prefix_cache;
  SamplingParams sampling_params;

  // 统计
  int total_prompt_tokens = 0;
  int total_generated_tokens = 0;
  double total_time_ms = 0.0;

  ChatSession() = default;

  // 重置消息与统计。注意：不动 sampling_params（「忘掉对话」不等于「重置参数」）。
  void Clear();

  // 往历史里追加一条消息。
  void AddMessage(const std::string& role, const std::string& content);

  // 一轮结束后累加统计。
  void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);

  // 把「当前已算过 KV 的 token 序列」整体替换成 tokens。
  void SetTokenHistory(const std::vector<int>& tokens);

  // 往 token_history 末尾追加一个 token（每步 decode 调一次）。
  void AppendToken(int token);

  // 登记「这条序列的 KV 已经算过」。
  void RecordPrefix(const std::vector<int>& tokens);

  // 返回 tokens 在已登记的序列里能匹配到的最长前缀长度。
  size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};

}  // namespace dqm
