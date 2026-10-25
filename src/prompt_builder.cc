#include "dqm_llama/prompt_builder.h"

#include <cctype>
#include <stack>
#include <string>
#include <vector>

#include "dqm_llama/gguf.h"

namespace dqm {

// ===========================================================================
// 极简 Jinja2 模板引擎（本模块的主体）
// ===========================================================================

namespace {

// ── 一条指令 ────────────────────────────────────────────────────────────
struct Instr {
  enum Type { kText, kOutput, kIf, kEndIf, kFor, kEndFor } type;
  std::string text;  // kText 装原文；kOutput / kIf / kFor 装表达式
  size_t jump = 0;   // kIf / kFor 的跳转目标（指令下标）
};

// ── 执行期「这一层循环走到第几个了」 ────────────────────────────────────
struct ForFrame {
  size_t pc = 0;     // 这层 FOR 指令的下标（用来识别「哪一层」）
  size_t index = 0;  // 当前迭代到 messages 的第几个
};

// ── 表达式求值时的「环境」 ──────────────────────────────────────────────
struct EvalContext {
  const std::vector<ChatMessage>* messages = nullptr;
  const ChatMessage* current_message = nullptr;
  size_t message_index = 0;
  bool add_generation_prompt = false;
};

// ── 辅助 ────────────────────────────────────────────────────────────────

// 两端去空白。isspace 的参数必须转成 unsigned char —— char 有符号，
// 字节 >= 128 时直接传会变负，是 UB。
std::string Trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  size_t b = s.size();
  // 注意是 b > a 而不是 b > 0 —— 全空白时不能让 b 退到 a 左边
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) {
    --b;
  }
  return s.substr(a, b - a);
}

// 剥掉 Jinja 空白控制标记 {%- -%} 首尾各一个 '-'
// （参考不做真正的空白控制，只剥标记；不剥的话下面 rfind("if ") 匹配不上）
std::string StripJinjaTrimMarkers(const std::string& raw) {
  std::string value = Trim(raw);
  if (!value.empty() && value.front() == '-') {
    value = Trim(value.substr(1));
  }
  if (!value.empty() && value.back() == '-') {
    value.pop_back();
    value = Trim(value);
  }
  return value;
}

// ── 表达式 / 条件求值 ───────────────────────────────────────────────────

// 只支持：字符串字面量 / 变量与访问链 / '+' 拼接。其余一律静默求值为空串。
std::string EvalExpr(const std::string& expr_raw, const EvalContext& ctx) {
  std::string expr = Trim(expr_raw);
  if (expr.empty()) {
    return "";
  }

  std::vector<std::string> parts;
  size_t pos = 0;
  while (pos < expr.size()) {
    while (pos < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[pos]))) {
      ++pos;
    }
    if (pos >= expr.size()) {
      break;
    }

    if (expr[pos] == '\'') {
      // 字符串字面量：读到下一个 ' 为止，逐字符原样取（不处理转义）
      ++pos;
      std::string lit;
      while (pos < expr.size() && expr[pos] != '\'') {
        lit += expr[pos];
        ++pos;
      }
      if (pos < expr.size() && expr[pos] == '\'') {
        ++pos;
      }
      parts.push_back(lit);
    } else {
      // 变量 / 访问链：读到下一个 '+' 或 '\'' 之前
      size_t start = pos;
      while (pos < expr.size() && expr[pos] != '+' && expr[pos] != '\'') {
        if (expr[pos] == '[') {
          ++pos;
          while (pos < expr.size() && expr[pos] != ']') {
            ++pos;
          }
          if (pos < expr.size()) {
            ++pos;
          }
        } else if (expr[pos] == '.') {
          ++pos;
          while (pos < expr.size() &&
                 !std::isspace(static_cast<unsigned char>(expr[pos])) &&
                 expr[pos] != '+' && expr[pos] != '\'' && expr[pos] != '[' &&
                 expr[pos] != '.') {
            ++pos;
          }
        } else {
          ++pos;
        }
      }
      std::string token = Trim(expr.substr(start, pos - start));
      if (!token.empty()) {
        if (token == "add_generation_prompt") {
          parts.push_back(ctx.add_generation_prompt ? "true" : "false");
        } else if (token == "message['role']" || token == "message[\"role\"]") {
          parts.push_back(ctx.current_message ? ctx.current_message->role : "");
        } else if (token == "message['content']" ||
                   token == "message[\"content\"]") {
          parts.push_back(ctx.current_message ? ctx.current_message->content
                                              : "");
        } else if (token.find("messages[") == 0) {
          // messages[<下标>]['role' | 'content']
          size_t close = token.find(']');
          if (close != std::string::npos && close + 1 < token.size() &&
              token[close + 1] == '[') {
            // substr(9, ...) 里的 9 是 "messages[" 的长度
            std::string idx_str = Trim(token.substr(9, close - 9));
            std::string key;
            size_t key_start = close + 2;
            size_t key_end = token.find(']', key_start);
            if (key_end != std::string::npos) {
              key = token.substr(key_start, key_end - key_start);
              if (key.size() >= 2 &&
                  ((key.front() == '\'' && key.back() == '\'') ||
                   (key.front() == '"' && key.back() == '"'))) {
                key = key.substr(1, key.size() - 2);
              }
            }
            try {
              int idx = std::stoi(idx_str);
              if (ctx.messages && idx >= 0 &&
                  idx < static_cast<int>(ctx.messages->size())) {
                const ChatMessage& msg = (*ctx.messages)[idx];
                if (key == "role") {
                  parts.push_back(msg.role);
                } else if (key == "content") {
                  parts.push_back(msg.content);
                } else {
                  parts.push_back("");
                }
              } else {
                parts.push_back("");
              }
            } catch (...) {
              parts.push_back("");
            }
          } else {
            parts.push_back("");
          }
        } else if (token == "loop.first") {
          parts.push_back(ctx.message_index == 0 ? "true" : "false");
        } else {
          parts.push_back("");
        }
      }
    }

    while (pos < expr.size() &&
           std::isspace(static_cast<unsigned char>(expr[pos]))) {
      ++pos;
    }
    if (pos < expr.size() && expr[pos] == '+') {
      ++pos;
    }
  }

  std::string result;
  for (const std::string& p : parts) {
    result += p;
  }
  return result;
}

// 只支持 and / or / == / != / 单变量。注意查 " and " 排在 " or " 之前，
// 于是 a or b and c 被求成 (a or b) and c —— 与 Jinja2 的优先级相反。
bool EvalCondition(const std::string& cond_raw, const EvalContext& ctx) {
  std::string cond = Trim(cond_raw);
  if (cond.empty()) {
    return false;
  }

  size_t and_pos = cond.find(" and ");
  if (and_pos != std::string::npos) {
    return EvalCondition(cond.substr(0, and_pos), ctx) &&
           EvalCondition(cond.substr(and_pos + 5), ctx);
  }

  size_t or_pos = cond.find(" or ");
  if (or_pos != std::string::npos) {
    return EvalCondition(cond.substr(0, or_pos), ctx) ||
           EvalCondition(cond.substr(or_pos + 4), ctx);
  }

  size_t neq_pos = cond.find(" != ");
  if (neq_pos != std::string::npos) {
    return EvalExpr(cond.substr(0, neq_pos), ctx) !=
           EvalExpr(cond.substr(neq_pos + 4), ctx);
  }
  size_t eq_pos = cond.find(" == ");
  if (eq_pos != std::string::npos) {
    return EvalExpr(cond.substr(0, eq_pos), ctx) ==
           EvalExpr(cond.substr(eq_pos + 4), ctx);
  }

  // 单变量：true/True 真，false/False 假，其他非空真、空假
  std::string val = EvalExpr(cond, ctx);
  if (val == "true" || val == "True") {
    return true;
  }
  if (val == "false" || val == "False") {
    return false;
  }
  return !val.empty();
}

// ── 编译期：模板字符串 → 指令数组 ───────────────────────────────────────

// 单遍扫描 + jump 回填：遇到 {% for %} / {% if %} 时还不知道配对标签在哪，
// 先把指令压进去、下标入栈；等遇到 endfor / endif 才回头把 jump 填上。
// if_stack / for_stack 只在编译期存在。
std::vector<Instr> CompileTemplate(const std::string& tmpl) {
  std::vector<Instr> instrs;
  std::stack<size_t> if_stack;
  std::stack<size_t> for_stack;

  size_t pos = 0;
  while (pos < tmpl.size()) {
    if (pos + 1 < tmpl.size() && tmpl[pos] == '{' && tmpl[pos + 1] == '%') {
      // 语句 {% ... %}
      pos += 2;
      size_t end = tmpl.find("%}", pos);
      if (end == std::string::npos) {
        break;  // 残缺模板：静默截断
      }
      std::string stmt = StripJinjaTrimMarkers(tmpl.substr(pos, end - pos));
      pos = end + 2;

      if (stmt.rfind("for ", 0) == 0) {
        instrs.push_back({Instr::kFor, stmt});
        for_stack.push(instrs.size() - 1);
      } else if (stmt == "endfor") {
        if (for_stack.empty()) {
          continue;
        }
        size_t for_pc = for_stack.top();
        for_stack.pop();
        instrs.push_back({Instr::kEndFor, "", for_pc});
        // 先 push 再取 size ⇒ 指向 ENDFOR 的**下一条**
        instrs[for_pc].jump = instrs.size();
      } else if (stmt.rfind("if ", 0) == 0) {
        instrs.push_back({Instr::kIf, stmt.substr(3)});
        if_stack.push(instrs.size() - 1);
      } else if (stmt == "endif") {
        if (if_stack.empty()) {
          continue;
        }
        size_t if_pc = if_stack.top();
        if_stack.pop();
        instrs.push_back({Instr::kEndIf, ""});
        instrs[if_pc].jump = instrs.size();
      }
      // 其余语句（set / macro…）静默跳过
    } else if (pos + 1 < tmpl.size() && tmpl[pos] == '{' &&
               tmpl[pos + 1] == '{') {
      // 表达式 {{ ... }}
      pos += 2;
      size_t end = tmpl.find("}}", pos);
      if (end == std::string::npos) {
        break;
      }
      instrs.push_back(
          {Instr::kOutput, StripJinjaTrimMarkers(tmpl.substr(pos, end - pos))});
      pos = end + 2;
    } else {
      // 纯文本：扫到下一个 {{ 或 {% 为止
      size_t end = pos;
      while (end < tmpl.size() &&
             !(tmpl[end] == '{' && end + 1 < tmpl.size() &&
               (tmpl[end + 1] == '%' || tmpl[end + 1] == '{'))) {
        ++end;
      }
      if (end > pos) {
        instrs.push_back({Instr::kText, tmpl.substr(pos, end - pos)});
      }
      pos = end;
    }
  }

  // 收尾：模板里没闭合的 if / for，把 jump 全指向数组末尾（= 直接结束）
  while (!if_stack.empty()) {
    instrs[if_stack.top()].jump = instrs.size();
    if_stack.pop();
  }
  while (!for_stack.empty()) {
    instrs[for_stack.top()].jump = instrs.size();
    for_stack.pop();
  }

  return instrs;
}

// ── 执行期：指令数组 + messages → 结果字符串 ────────────────────────────

// pc 驱动的栈式虚拟机：指令自己负责决定 pc 下一步去哪。
std::string ExecuteTemplate(const std::vector<Instr>& instrs,
                            const std::vector<ChatMessage>& messages,
                            bool add_gen) {
  std::string output;
  EvalContext ctx;
  ctx.messages = &messages;
  ctx.add_generation_prompt = add_gen;

  std::stack<ForFrame> for_stack;
  size_t pc = 0;

  while (pc < instrs.size()) {
    const Instr& instr = instrs[pc];
    switch (instr.type) {
      case Instr::kText:
        output += instr.text;
        ++pc;
        break;
      case Instr::kOutput:
        output += EvalExpr(instr.text, ctx);
        ++pc;
        break;
      case Instr::kIf:
        // 假 → 一步跳到 endif 之后（编译期已经回填好了）
        pc = EvalCondition(instr.text, ctx) ? pc + 1 : instr.jump;
        break;
      case Instr::kEndIf:
        // 纯锚点：什么都不做，它存在只是为了让 kIf 的 jump 有地方指
        ++pc;
        break;
      case Instr::kFor: {
        // 同一条 kFor 会在两种时刻被执行：刚进入循环、以及被 kEndFor 跳回来。
        // 靠 for_stack.top().pc == pc 区分 —— 比栈顶才能认出自已是哪一层。
        if (for_stack.empty() || for_stack.top().pc != pc) {
          // 新循环
          ForFrame frame;
          frame.pc = pc;
          frame.index = 0;
          for_stack.push(frame);
          if (messages.empty()) {
            for_stack.pop();
            pc = instr.jump;  // 空列表 ⇒ 循环体一次都不跑
          } else {
            ctx.current_message = &messages[0];
            ctx.message_index = 0;
            ++pc;
          }
        } else {
          // 续下一轮
          ForFrame& frame = for_stack.top();
          ++frame.index;
          if (frame.index >= messages.size()) {
            for_stack.pop();
            pc = instr.jump;  // 转完了 ⇒ 退出循环
          } else {
            ctx.current_message = &messages[frame.index];
            ctx.message_index = frame.index;
            ++pc;
          }
        }
        break;
      }
      case Instr::kEndFor:
        pc = instr.jump;  // 跳回配对的 kFor，由它判断"续轮还是退出"
        break;
    }
  }

  return output;
}

}  // namespace

// ===========================================================================
// PromptBuilder
// ===========================================================================

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
  chat_template_ = template_str;
}

std::string PromptBuilder::Build(
    const std::vector<ChatMessage>& messages) const {
  if (chat_template_.empty()) {
    return BuildPlain(messages);
  }

  if (chat_template_.size() >= 2 && chat_template_[0] == '{' &&
      (chat_template_[1] == '%' || chat_template_[1] == '{')) {
    std::vector<Instr> instrs = CompileTemplate(chat_template_);
    return ExecuteTemplate(instrs, messages, true);
  }

  if (chat_template_ == "qwen2") {
    return BuildQwen2(messages);
  }

  return BuildPlain(messages);
}

std::string PromptBuilder::BuildPlain(
    const std::vector<ChatMessage>& messages) const {
  std::string prompt;
  for (const ChatMessage& msg : messages) {
    if (msg.role == "system") {
      prompt += "System: " + msg.content + "\n";
    } else if (msg.role == "user") {
      prompt += "User: " + msg.content + "\n";
    } else if (msg.role == "assistant") {
      prompt += "Assistant: " + msg.content + "\n";
    }
  }

  prompt += "Assistant:";

  return prompt;
}

std::string PromptBuilder::BuildQwen2(
    const std::vector<ChatMessage>& messages) const {
  // 先扫一遍：有没有 system
  bool has_system = false;
  for (const ChatMessage& msg : messages) {
    if (msg.role == "system") {
      has_system = true;
      break;
    }
  }

  std::string prompt;
  if (!has_system) {
    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  }

  for (const ChatMessage& msg : messages) {
    if (msg.role != "system" && msg.role != "user" && msg.role != "assistant") {
      continue;
    }
    prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
  }

  prompt += "<|im_start|>assistant\n";

  return prompt;
}

std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  std::string tmpl;
  if (GgufGetMetadataString(gguf_path, "tokenizer.chat_template", tmpl)) {
    return tmpl;
  }
  std::string architecture;
  if (GgufGetMetadataString(gguf_path, "general.architecture", architecture) &&
      architecture == "qwen2") {
    return "qwen2";
  }
  return "";
}

}  // namespace dqm
