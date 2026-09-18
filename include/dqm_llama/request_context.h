#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace dqm {

// 测耗时专用时钟：单调，不受系统时间调整影响。
using RequestClock = std::chrono::steady_clock;

// 一条细粒度追踪事件。
struct TraceEvent {
  std::string stage;
  double elapsed_ms = 0.0;
  int tokens = 0;
  std::string detail;
};

// 一次请求的「黑匣子」：只记录，不参与任何计算。
// 删掉它对推理结果毫无影响。
struct RequestContext {
  std::string trace_id;
  std::string mode;
  std::string backend;
  std::string model_path;

  RequestClock::time_point start_time;
  double total_ms = 0.0;

  int prompt_tokens = 0;
  int prefill_tokens = 0;
  int decode_tokens = 0;
  int generated_tokens = 0;

  double model_load_ms = 0.0;
  double tokenize_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double sample_ms = 0.0;

  std::string error;
  std::vector<TraceEvent> events;

  void Start();
  void Finish();
  void RecordEvent(const std::string& stage, double elapsed_ms, int tokens = 0,
                   const std::string& detail = "");
  void SetError(const std::string& message);
  bool ok() const;
};

std::string NewTraceId();
double ElapsedMs(RequestClock::time_point start);
RequestContext StartRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path);
std::string FormatRequestTraceSummary(const RequestContext& request);
std::vector<std::string> FormatRequestTraceEvents(const RequestContext& request);

}  // namespace dqm
