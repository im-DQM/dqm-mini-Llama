#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "dqm_llama/thread_pool.h"

using dqm::GetThreadCount;
using dqm::ParallelFor;
using dqm::SetThreadCount;

namespace {

// 记录每个下标被访问的次数（区间正确时应恰好各 1 次）。
std::vector<int> VisitCounts(int n) {
  std::vector<int> hits(static_cast<size_t>(n), 0);
  ParallelFor(n, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      hits[static_cast<size_t>(i)] += 1;
    }
  });
  return hits;
}

// 收集本次 ParallelFor 实际落到 fn 的区间。
// 各线程并发进来，必须加锁 —— 无锁 emplace_back 是数据竞争。
std::vector<std::pair<int, int>> CollectRanges(int n) {
  std::vector<std::pair<int, int>> ranges;
  std::mutex mu;
  ParallelFor(n, [&](int begin, int end) {
    std::lock_guard<std::mutex> lock(mu);
    ranges.emplace_back(begin, end);
  });
  return ranges;
}

}  // namespace

// ===========================================================================
// 线程数配置
// ===========================================================================

TEST(ThreadCount, DefaultIsPositive) {
  SetThreadCount(0);  // 回到默认
  EXPECT_GT(GetThreadCount(), 0);
}

TEST(ThreadCount, SetOverridesAndZeroResets) {
  SetThreadCount(0);
  const int hw = GetThreadCount();
  SetThreadCount(3);
  EXPECT_EQ(GetThreadCount(), 3);
  SetThreadCount(0);
  EXPECT_EQ(GetThreadCount(), hw);  // 0 复位到硬件默认
}

// auto 模式（g_thread_count == 0）下，GetThreadCount 不应每次调用都重新
// 探测硬件并发数 —— 某些环境（大核数宿主 + 容器配额）单次探测可达数百微秒，
// 而 ParallelFor 每次调用都会问一次，会拖垮小矩阵。
TEST(ThreadCount, AutoModeDoesNotReprobeHardwareEveryCall) {
  SetThreadCount(0);
  constexpr int kCalls = 10000;

  const auto start = std::chrono::steady_clock::now();
  int value = 0;
  for (int i = 0; i < kCalls; ++i) {
    value = GetThreadCount();
  }
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  SetThreadCount(0);

  EXPECT_GT(value, 0);
  EXPECT_LT(elapsed_us, 2000)
      << kCalls << " calls took " << elapsed_us << " us";
}

// 反复调用 ParallelFor 时不应每次都重新创建线程 —— 小矩阵上线程创建
// 开销会远超计算本身（tiny 模型实测劣化约 8 倍）。
// 阈值取在 fork-join（约 64 ms）与常驻池（约 25 ms）之间。
TEST(ParallelFor, RepeatedCallsDoNotPayThreadSpawnCost) {
  SetThreadCount(4);
  constexpr int kCalls = 500;
  constexpr int kN = 64;  // >= 4 * 16，走并行分支
  std::atomic<int> sink{0};

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i) {
    ParallelFor(kN, [&sink](int begin, int end) { sink += end - begin; });
  }
  const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now() - start)
                              .count();
  SetThreadCount(0);

  EXPECT_EQ(sink.load(), kCalls * kN);
  EXPECT_LT(elapsed_us, 35000)
      << kCalls << " calls took " << elapsed_us << " us";
}

// 行为版：反复调用后观察到的线程 id 应只有少数几个（主线程 + 常驻 worker），
// 而不是每次调用都换一批新线程。
TEST(ParallelFor, RepeatedCallsReuseWorkerThreads) {
  SetThreadCount(4);
  std::mutex mutex;
  std::set<std::thread::id> ids;

  for (int i = 0; i < 100; ++i) {
    ParallelFor(64, [&](int, int) {
      std::lock_guard<std::mutex> lock(mutex);
      ids.insert(std::this_thread::get_id());
    });
  }
  SetThreadCount(0);

  EXPECT_GE(ids.size(), 2u);  // 至少证明真的并行过
  EXPECT_LE(ids.size(), 5u);  // 主线程 + 至多 4 个 worker；fork-join 会有 300+
}

TEST(ThreadCount, NegativeTreatedAsReset) {
  SetThreadCount(0);
  const int hw = GetThreadCount();
  SetThreadCount(-5);
  EXPECT_EQ(GetThreadCount(), hw);
}

// ===========================================================================
// ParallelFor：区间正确性
// ===========================================================================

TEST(ParallelFor, CoversEveryIndexExactlyOnce) {
  SetThreadCount(4);
  const int n = 10000;  // 远超门限，走并行路径
  const std::vector<int> hits = VisitCounts(n);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(hits[static_cast<size_t>(i)], 1) << "index " << i;
  }
}

TEST(ParallelFor, SmallInputGoesSerialSingleRange) {
  SetThreadCount(8);
  const std::vector<std::pair<int, int>> ranges = CollectRanges(4);  // 4 < 8*16
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].first, 0);
  EXPECT_EQ(ranges[0].second, 4);
}

TEST(ParallelFor, SingleThreadStillCoversAll) {
  SetThreadCount(1);
  const int n = 1000;
  const std::vector<std::pair<int, int>> ranges = CollectRanges(n);
  ASSERT_EQ(ranges.size(), 1u);
  EXPECT_EQ(ranges[0].first, 0);
  EXPECT_EQ(ranges[0].second, n);
}

TEST(ParallelFor, NonPositiveNIsNoop) {
  int calls = 0;
  ParallelFor(0, [&](int, int) { ++calls; });
  ParallelFor(-3, [&](int, int) { ++calls; });
  EXPECT_EQ(calls, 0);
}

TEST(ParallelFor, RangesAreContiguousAndDisjoint) {
  SetThreadCount(4);
  const int n = 4000;
  std::vector<std::pair<int, int>> ranges = CollectRanges(n);
  std::sort(ranges.begin(), ranges.end());
  int expected_start = 0;
  for (const auto& r : ranges) {
    EXPECT_EQ(r.first, expected_start);  // 首尾相接
    EXPECT_LT(r.first, r.second);
    expected_start = r.second;
  }
  EXPECT_EQ(expected_start, n);  // 覆盖到 n
}

// ===========================================================================
// ParallelFor：异常跨线程传播
// ===========================================================================

TEST(ParallelFor, RethrowsWorkerExceptionOnCaller) {
  SetThreadCount(4);
  const int n = 10000;
  EXPECT_THROW(
      ParallelFor(n,
                  [](int begin, int end) {
                    for (int i = begin; i < end; ++i) {
                      if (i == 0) {
                        throw std::runtime_error("boom");
                      }
                    }
                  }),
      std::runtime_error);
}
