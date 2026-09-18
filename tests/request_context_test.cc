#include <gtest/gtest.h>

#include <cctype>
#include <chrono>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#include "dqm_llama/request_context.h"

using dqm::ElapsedMs;
using dqm::FormatRequestTraceEvents;
using dqm::FormatRequestTraceSummary;
using dqm::NewTraceId;
using dqm::RequestClock;
using dqm::RequestContext;
using dqm::StartRequest;

// 测耗时必须用单调时钟：system_clock 会被 NTP / 手动改时间拨动，
// 拨回去就会算出负的耗时。
static_assert(std::is_same_v<RequestClock, std::chrono::steady_clock>,
              "RequestClock 必须是 steady_clock");

namespace {

// 子串断言 —— 比咬死整行字符串脆性低，也更能表达「这里必须有这一段」。
void ExpectContains(const std::string& haystack, const std::string& needle) {
  EXPECT_NE(haystack.find(needle), std::string::npos)
      << "在输出里找不到 [" << needle << "]\n实际输出: " << haystack;
}

bool IsAllDigits(const std::string& s, size_t from, size_t to) {
  for (size_t i = from; i < to; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  return true;
}

}  // namespace

// ===========================================================================
// 默认状态
// ===========================================================================

TEST(RequestContext, DefaultsToOkWithNothingRecorded) {
  const RequestContext request;

  EXPECT_TRUE(request.ok());
  EXPECT_TRUE(request.error.empty());
  EXPECT_TRUE(request.events.empty());
  EXPECT_TRUE(request.trace_id.empty());

  EXPECT_DOUBLE_EQ(request.total_ms, 0.0);
  EXPECT_DOUBLE_EQ(request.model_load_ms, 0.0);
  EXPECT_DOUBLE_EQ(request.tokenize_ms, 0.0);
  EXPECT_DOUBLE_EQ(request.prefill_ms, 0.0);
  EXPECT_DOUBLE_EQ(request.decode_ms, 0.0);
  EXPECT_DOUBLE_EQ(request.sample_ms, 0.0);

  EXPECT_EQ(request.prompt_tokens, 0);
  EXPECT_EQ(request.prefill_tokens, 0);
  EXPECT_EQ(request.decode_tokens, 0);
  EXPECT_EQ(request.generated_tokens, 0);
}

// ===========================================================================
// 错误状态：ok() 等价于 error.empty()
// ===========================================================================

TEST(RequestContext, SetErrorFlipsOkToFalse) {
  RequestContext request;
  request.SetError("Inference error: bad shape");

  EXPECT_FALSE(request.ok());
  EXPECT_EQ(request.error, "Inference error: bad shape");
}

TEST(RequestContext, SetErrorOverwritesPreviousMessage) {
  RequestContext request;
  request.SetError("first");
  request.SetError("second");

  EXPECT_EQ(request.error, "second");
}

// ===========================================================================
// RecordEvent
// ===========================================================================

TEST(RequestContext, RecordEventAppendsInOrder) {
  RequestContext request;
  request.RecordEvent("prefill", 1.5, 4, "first");
  request.RecordEvent("decode", 2.25, 1, "second");

  ASSERT_EQ(request.events.size(), 2u);

  EXPECT_EQ(request.events[0].stage, "prefill");
  EXPECT_DOUBLE_EQ(request.events[0].elapsed_ms, 1.5);
  EXPECT_EQ(request.events[0].tokens, 4);
  EXPECT_EQ(request.events[0].detail, "first");

  EXPECT_EQ(request.events[1].stage, "decode");
  EXPECT_DOUBLE_EQ(request.events[1].elapsed_ms, 2.25);
  EXPECT_EQ(request.events[1].tokens, 1);
  EXPECT_EQ(request.events[1].detail, "second");
}

TEST(RequestContext, RecordEventDefaultsTokensAndDetail) {
  RequestContext request;
  request.RecordEvent("sample", 0.25);

  ASSERT_EQ(request.events.size(), 1u);
  EXPECT_EQ(request.events[0].stage, "sample");
  EXPECT_DOUBLE_EQ(request.events[0].elapsed_ms, 0.25);
  EXPECT_EQ(request.events[0].tokens, 0);
  EXPECT_TRUE(request.events[0].detail.empty());
}

// RecordEvent 不测时 —— elapsed_ms 是调用方算好传进来的。
TEST(RequestContext, RecordEventStoresCallerSuppliedElapsed) {
  RequestContext request;
  request.RecordEvent("prefill", 123.456, 7);

  ASSERT_EQ(request.events.size(), 1u);
  EXPECT_DOUBLE_EQ(request.events[0].elapsed_ms, 123.456);
}

// ===========================================================================
// 计时
// ===========================================================================

TEST(RequestContext, ElapsedMsIsNonNegative) {
  const RequestClock::time_point start = RequestClock::now();
  EXPECT_GE(ElapsedMs(start), 0.0);
}

TEST(RequestContext, FinishMeasuresSinceStart) {
  RequestContext request;
  request.Start();
  request.Finish();

  EXPECT_GE(request.total_ms, 0.0);
}

// Start() 只打起点，total_ms 归 Finish() 填。
TEST(RequestContext, StartAloneLeavesTotalMsUntouched) {
  RequestContext request;
  request.Start();

  EXPECT_DOUBLE_EQ(request.total_ms, 0.0);
}

// ===========================================================================
// NewTraceId
// ===========================================================================

TEST(RequestContext, NewTraceIdHasExpectedShape) {
  const std::string id = NewTraceId();

  // req_ + YYYYMMDDHHMMSS(14) + _ + 6 位零填充序号
  ASSERT_EQ(id.size(), 25u);
  EXPECT_EQ(id.substr(0, 4), "req_");
  EXPECT_EQ(id[18], '_');
  EXPECT_TRUE(IsAllDigits(id, 4, 18)) << "时间戳不是 14 位数字: " << id;
  EXPECT_TRUE(IsAllDigits(id, 19, 25)) << "序号不是 6 位数字: " << id;
}

TEST(RequestContext, NewTraceIdIsUniqueAcrossCalls) {
  EXPECT_NE(NewTraceId(), NewTraceId());
  EXPECT_NE(NewTraceId(), NewTraceId());
}

// 序号是 setw(6) + setfill('0') 补零的。
// 少了 setfill('0') 就会用空格补齐 —— 长度对、但字符集和首位都不对。
TEST(RequestContext, NewTraceIdSequenceIsZeroPadded) {
  const std::string id = NewTraceId();
  const std::string seq = id.substr(19, 6);

  ASSERT_EQ(seq.size(), 6u);
  EXPECT_TRUE(IsAllDigits(seq, 0, 6)) << "序号里混进了非数字（补成空格了？）: " << seq;
  EXPECT_EQ(seq[0], '0') << "进程内序号远小于 100000，首位必须是补出来的 0: " << seq;
}

// ===========================================================================
// StartRequest（工厂）
// ===========================================================================

TEST(RequestContext, StartRequestFillsIdentityFields) {
  const RequestContext request =
      StartRequest("run", "cpu", "models/tiny/model.bin");

  EXPECT_EQ(request.mode, "run");
  EXPECT_EQ(request.backend, "cpu");
  EXPECT_EQ(request.model_path, "models/tiny/model.bin");
  EXPECT_FALSE(request.trace_id.empty());
  EXPECT_TRUE(request.ok());
  EXPECT_TRUE(request.events.empty());
}

TEST(RequestContext, StartRequestStartsTheClock) {
  RequestContext request = StartRequest("generate", "cpu", "m.bin");
  request.Finish();

  EXPECT_GE(request.total_ms, 0.0);
}

// ===========================================================================
// FormatRequestTraceSummary
// ===========================================================================

TEST(RequestContext, SummaryReportsOkStatus) {
  const RequestContext request = StartRequest("generate", "cpu", "m.bin");
  const std::string summary = FormatRequestTraceSummary(request);

  ExpectContains(summary, "status=ok");
  ExpectContains(summary, "mode=generate");
  ExpectContains(summary, "backend=cpu");
  ExpectContains(summary, "model=m.bin");
  ExpectContains(summary, "trace_id=" + request.trace_id);
}

TEST(RequestContext, SummaryReportsErrorStatusAndMessage) {
  RequestContext request;
  request.trace_id = "req_x";
  request.SetError("Inference error: bad shape");

  const std::string summary = FormatRequestTraceSummary(request);

  ExpectContains(summary, "status=error");
  ExpectContains(summary, "error=\"Inference error: bad shape\"");
}

TEST(RequestContext, SummaryOmitsErrorFieldWhenOk) {
  const RequestContext request;
  const std::string summary = FormatRequestTraceSummary(request);

  EXPECT_EQ(summary.find("error="), std::string::npos) << summary;
}

TEST(RequestContext, SummaryFormatsMillisecondsWithThreeDecimals) {
  RequestContext request;
  request.model_load_ms = 1.5;
  request.prefill_ms = 0.25;
  request.total_ms = 12.0;

  const std::string summary = FormatRequestTraceSummary(request);

  ExpectContains(summary, "model_load_ms=1.500");
  ExpectContains(summary, "prefill_ms=0.250");
  ExpectContains(summary, "total_ms=12.000");
}

TEST(RequestContext, SummaryReportsTokenCounts) {
  RequestContext request;
  request.prompt_tokens = 5;
  request.prefill_tokens = 5;
  request.decode_tokens = 3;
  request.generated_tokens = 3;

  const std::string summary = FormatRequestTraceSummary(request);

  ExpectContains(summary, "prompt_tokens=5");
  ExpectContains(summary, "prefill_tokens=5");
  ExpectContains(summary, "decode_tokens=3");
  ExpectContains(summary, "generated_tokens=3");
}

// ===========================================================================
// FormatRequestTraceEvents
// ===========================================================================

TEST(RequestContext, NoEventsFormatsToEmptyVector) {
  const RequestContext request;
  EXPECT_TRUE(FormatRequestTraceEvents(request).empty());
}

TEST(RequestContext, EventsFormattedOneLineEach) {
  RequestContext request;
  request.trace_id = "req_x";
  request.RecordEvent("prefill", 1.25, 5, "radix_hit=2");
  request.RecordEvent("decode", 0.5, 1);

  const std::vector<std::string> lines = FormatRequestTraceEvents(request);

  ASSERT_EQ(lines.size(), 2u);
  ExpectContains(lines[0], "stage=prefill");
  ExpectContains(lines[0], "tokens=5");
  ExpectContains(lines[0], "elapsed_ms=1.250");
  ExpectContains(lines[0], "detail=\"radix_hit=2\"");
  ExpectContains(lines[0], "trace_id=req_x");

  ExpectContains(lines[1], "stage=decode");
  ExpectContains(lines[1], "tokens=1");
  ExpectContains(lines[1], "elapsed_ms=0.500");
}

TEST(RequestContext, EventDetailOmittedWhenEmpty) {
  RequestContext request;
  request.RecordEvent("decode", 0.5, 1);

  const std::vector<std::string> lines = FormatRequestTraceEvents(request);

  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].find("detail="), std::string::npos) << lines[0];
}

// ===========================================================================
// 两个 Format* 都是 const& —— 不改入参、可重复调用
// ===========================================================================

TEST(RequestContext, FormattersDoNotMutateInput) {
  RequestContext request;
  request.trace_id = "req_x";
  request.RecordEvent("decode", 1.0, 1, "d");

  const RequestContext before = request;
  const std::string summary_first = FormatRequestTraceSummary(request);
  const std::string summary_second = FormatRequestTraceSummary(request);
  const std::vector<std::string> events = FormatRequestTraceEvents(request);

  EXPECT_EQ(summary_first, summary_second);
  EXPECT_EQ(events.size(), before.events.size());
  EXPECT_EQ(request.trace_id, before.trace_id);
  EXPECT_EQ(request.error, before.error);
  EXPECT_EQ(request.events.size(), before.events.size());
}
