#pragma once

#include <functional>

namespace dqm {

// 当前并行线程数。默认 std::thread::hardware_concurrency()。
int GetThreadCount();

// 设置并行线程数；传 0 表示回退到 hardware_concurrency()。
void SetThreadCount(int n);

// 简单并行 for：把 [0, n) 切成大致相等的连续区间，交给 GetThreadCount() 个线程跑。
// 每个线程负责的区间至少 kMinChunk 个元素时才真正开线程，否则直接串行跑 fn(0, n)。
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}  // namespace dqm
