#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "dqm_llama/prompt_builder.h"

using dqm::ChatMessage;
using dqm::PromptBuilder;

namespace {

// 一把顺手的构造器：Build({...}) 的入参都是消息列表
std::string BuildWith(const PromptBuilder& b,
                      const std::vector<ChatMessage>& messages) {
  return b.Build(messages);
}

}  // namespace

// ===========================================================================
// 默认状态与四路分发
// ===========================================================================

// 默认构造 => chat_template_ 为空 => plain
TEST(PromptBuilder, DefaultConstructedUsesPlain) {
  const PromptBuilder builder;

  EXPECT_EQ(BuildWith(builder, {{"user", "hi"}}), "User: hi\nAssistant:");
}

TEST(PromptBuilder, EmptyTemplateUsesPlain) {
  PromptBuilder builder;
  builder.SetChatTemplate("");

  EXPECT_EQ(BuildWith(builder, {{"user", "hi"}}), "User: hi\nAssistant:");
}

// 识别模板只看前两个字符；长度不足 2 时不能越界读 => 落回 plain
TEST(PromptBuilder, TooShortTemplateFallsBackToPlain) {
  PromptBuilder builder;
  builder.SetChatTemplate("{");

  EXPECT_EQ(BuildWith(builder, {{"user", "hi"}}), "User: hi\nAssistant:");
}

// 第二个字符既不是 % 也不是 { => 不是模板 => plain
TEST(PromptBuilder, SingleBraceLooksLikePlainText) {
  PromptBuilder builder;
  builder.SetChatTemplate("{hello}");

  EXPECT_EQ(BuildWith(builder, {{"user", "hi"}}), "User: hi\nAssistant:");
}

// 不认识的名字 => 兜底 plain（SetChatTemplate 不校验）
TEST(PromptBuilder, UnknownTemplateNameFallsBackToPlain) {
  PromptBuilder builder;
  builder.SetChatTemplate("llama3");

  EXPECT_EQ(BuildWith(builder, {{"user", "hi"}}), "User: hi\nAssistant:");
}

// 以 {% 开头 => 交给 Jinja2 引擎
TEST(PromptBuilder, TemplateStartingWithStatementGoesToEngine) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% if add_generation_prompt %}X{% endif %}");

  EXPECT_EQ(BuildWith(builder, {}), "X");
}

// 以 {{ 开头 => 交给 Jinja2 引擎
TEST(PromptBuilder, TemplateStartingWithExpressionGoesToEngine) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ 'hi' }}");

  EXPECT_EQ(BuildWith(builder, {}), "hi");
}

// ===========================================================================
// plain
// ===========================================================================

TEST(PromptBuilder, PlainFormatsKnownRoles) {
  PromptBuilder builder;

  EXPECT_EQ(BuildWith(builder, {{"system", "S"},
                                {"user", "U"},
                                {"assistant", "A"}}),
            "System: S\nUser: U\nAssistant: A\nAssistant:");
}

// 契约：不认识的 role 整条静默跳过（不是报错）
TEST(PromptBuilder, PlainSkipsUnknownRole) {
  PromptBuilder builder;

  EXPECT_EQ(BuildWith(builder, {{"tool", "ignored"}, {"user", "hello"}}),
            "User: hello\nAssistant:");
}

// 契约：role 是精确字符串比较，大小写敏感
TEST(PromptBuilder, PlainRoleIsCaseSensitive) {
  PromptBuilder builder;

  EXPECT_EQ(BuildWith(builder, {{"User", "ignored"}, {"user", "kept"}}),
            "User: kept\nAssistant:");
}

// 空 messages => 输出就只剩那个"该你说了"的钩子
TEST(PromptBuilder, PlainEmptyMessagesIsJustAssistantHook) {
  PromptBuilder builder;

  EXPECT_EQ(BuildWith(builder, {}), "Assistant:");
}

// 契约：末尾的 "Assistant:" 没有换行 —— 模型要接着它续写
TEST(PromptBuilder, PlainAssistantHookHasNoTrailingNewline) {
  PromptBuilder builder;

  const std::string prompt = BuildWith(builder, {{"user", "hi"}});

  ASSERT_FALSE(prompt.empty());
  EXPECT_EQ(prompt.back(), ':');
}

TEST(PromptBuilder, PlainMultiTurnKeepsOrder) {
  PromptBuilder builder;
  const std::vector<ChatMessage> messages = {{"system", "You are helpful."},
                                             {"user", "hi"},
                                             {"assistant", "Hello!"},
                                             {"user", "bye"}};

  EXPECT_EQ(BuildWith(builder, messages),
            "System: You are helpful.\n"
            "User: hi\n"
            "Assistant: Hello!\n"
            "User: bye\n"
            "Assistant:");
}

// ===========================================================================
// 内建 qwen2
// ===========================================================================

// 契约：没有 system 时自动注入一条默认的
TEST(PromptBuilder, Qwen2InjectsDefaultSystemWhenAbsent) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");

  EXPECT_EQ(BuildWith(builder, {{"user", "你好"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "你好<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 契约：已经有 system 就原样用，不重复注入
TEST(PromptBuilder, Qwen2KeepsExistingSystem) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");

  EXPECT_EQ(BuildWith(builder, {{"system", "Be concise."}, {"user", "hi"}}),
            "<|im_start|>system\n"
            "Be concise.<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST(PromptBuilder, Qwen2SkipsUnknownRole) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");

  EXPECT_EQ(BuildWith(builder, {{"system", "S"}, {"tool", "ignored"}}),
            "<|im_start|>system\n"
            "S<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 空 messages 仍然有默认 system + 生成钩子
TEST(PromptBuilder, Qwen2EmptyMessagesStillHasSystemAndHook) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");

  EXPECT_EQ(BuildWith(builder, {}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// ===========================================================================
// Jinja2 引擎：文本 / 表达式
// ===========================================================================

TEST(PromptBuilder, EngineEmitsTextAfterOutput) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ 'b' }}c");

  EXPECT_EQ(BuildWith(builder, {}), "bc");
}

TEST(PromptBuilder, EngineEvaluatesStringLiteral) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ 'abc' }}");

  EXPECT_EQ(BuildWith(builder, {}), "abc");
}

// 表达式支持的运算只有字符串拼接 +
TEST(PromptBuilder, EngineConcatenatesLiteralsWithPlus) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ 'a' + 'b' + 'c' }}");

  EXPECT_EQ(BuildWith(builder, {}), "abc");
}

TEST(PromptBuilder, EngineEmitsRawTextInsideIf) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% if add_generation_prompt %}raw{% endif %}");

  EXPECT_EQ(BuildWith(builder, {}), "raw");
}

// ===========================================================================
// Jinja2 引擎：for
// ===========================================================================

// 循环变量必须**字面叫 message** —— 引擎认的是写死的 "message[...]"（见下面的坑用例）
TEST(PromptBuilder, EngineForLoopsOverMessages) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}{{ message['role'] }};{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "a"}, {"assistant", "b"}}),
            "user;assistant;");
}

TEST(PromptBuilder, EngineForReadsContent) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}{{ message['content'] }}{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "你好"}, {"assistant", "hi"}}),
            "你好hi");
}

// 契约：循环空列表 => 循环体一次都不执行（不抛）
TEST(PromptBuilder, EngineEmptyMessagesRunsLoopBodyZeroTimes) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% for m in messages %}.{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {}), "");
}

TEST(PromptBuilder, EngineLoopFirst) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}"
      "{% if loop.first %}F{% endif %}.{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "a"}, {"user", "b"}, {"user", "c"}}),
            "F...");
}

// ⚠️ 坑：引擎认的是**写死的字面量** "message['role']"。循环变量换个名字
// 不会报错，只会静默求值成空串 —— 循环变量名和表达式之间没有绑定。
TEST(PromptBuilder, EngineRequiresLiteralMessageVariableName) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for m in messages %}{{ m['role'] }};{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "a"}, {"assistant", "b"}}), ";;");
}

// ⚠️ 坑：不支持 {% else %} —— 它被当"不认识的语句"静默跳过，
// 于是 else 后面的内容落进 if 体，条件为假时被一并跳过。
TEST(PromptBuilder, EngineDoesNotSupportElse) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}"
      "{% if loop.first %}F{% else %}N{% endif %}"
      "{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "a"}, {"user", "b"}}), "FN");
}

// 契约：循环嵌套支持（运行期的 for_stack 是栈）
TEST(PromptBuilder, EngineNestedFor) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for m in messages %}{% for n in messages %}.{% endfor %}{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "a"}, {"user", "b"}}), "....");
}

// for 里嵌 if，靠当前这条消息的 role 决定输出
TEST(PromptBuilder, EngineForWithIfOnCurrentMessage) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}"
      "{% if message['role'] == 'user' %}U{% endif %}"
      "{% endfor %}");

  EXPECT_EQ(BuildWith(builder, {{"user", "a"}, {"assistant", "b"}, {"user", "c"}}),
            "UU");
}

// ===========================================================================
// Jinja2 引擎：条件
// ===========================================================================

// Build 传给引擎的 add_gen 硬编码 true
TEST(PromptBuilder, EngineIfTrueOnAddGenerationPrompt) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% if add_generation_prompt %}Y{% endif %}");

  EXPECT_EQ(BuildWith(builder, {}), "Y");
}

// 契约：未知变量静默求值为空串 => 条件恒假 => 整段被略过，且不报错
// （模板必须以 {% 或 {{ 开头才会进引擎，所以把尾随的 "A" 放在 if 之后）
TEST(PromptBuilder, EngineIfFalseOnUnknownVariable) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% if typo_variable %}B{% endif %}A");

  EXPECT_EQ(BuildWith(builder, {}), "A");
}

// 契约：不认识的语句（set / macro…）静默跳过
TEST(PromptBuilder, EngineUnknownStatementIsIgnored) {
  PromptBuilder builder;
  builder.SetChatTemplate("{% set x = 1 %}ok");

  EXPECT_EQ(BuildWith(builder, {}), "ok");
}

// Jinja2 的空白控制标记 {%- -%}：参考只把首尾的 '-' 剥掉
TEST(PromptBuilder, EngineStripsTrimMarkers) {
  PromptBuilder builder;
  builder.SetChatTemplate("{%- if add_generation_prompt -%}X{%- endif -%}");

  EXPECT_EQ(BuildWith(builder, {}), "X");
}

// 契约：找不到 }} => 编译直接 break，后半段丢弃，不抛
TEST(PromptBuilder, EngineTruncatesOnMissingCloseDelimiter) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ 'abc'");

  EXPECT_EQ(BuildWith(builder, {}), "");
}

// ===========================================================================
// Jinja2 引擎：端到端
// ===========================================================================

// Qwen2 官方模板的形状（GGUF 里存的就是这种）。
// 注意 '\n' 在模板里是**真换行符** —— GGUF 存 JSON 时已被解码；
// 引擎的字面量扫描不做转义处理，读到 ' 才停。
TEST(PromptBuilder, EngineBuildsQwen2StyleTemplateEndToEnd) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}"
      "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
      "'<|im_end|>' + '\n' }}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");

  EXPECT_EQ(BuildWith(builder, {{"system", "Be concise."}, {"user", "hi"}}),
            "<|im_start|>system\n"
            "Be concise.<|im_end|>\n"
            "<|im_start|>user\n"
            "hi<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 同一条 messages 走「模板引擎」和走「内建 qwen2」，应当拼出一样的串
TEST(PromptBuilder, EngineTemplateAgreesWithBuiltinQwen2) {
  const std::vector<ChatMessage> messages = {{"system", "S"}, {"user", "hi"}};

  PromptBuilder jinja;
  jinja.SetChatTemplate(
      "{% for message in messages %}"
      "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
      "'<|im_end|>' + '\n' }}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}");

  PromptBuilder builtin;
  builtin.SetChatTemplate("qwen2");

  EXPECT_EQ(jinja.Build(messages), builtin.Build(messages));
}

// messages[n][...] 走的是下标访问链
TEST(PromptBuilder, EngineIndexedMessageAccess) {
  PromptBuilder builder;
  builder.SetChatTemplate("{{ messages[1]['content'] }}");

  EXPECT_EQ(BuildWith(builder, {{"user", "first"}, {"assistant", "second"}}),
            "second");
}
