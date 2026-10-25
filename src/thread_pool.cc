// thread_pool.cc —— 并行 for 原语。
//
// worker 线程常驻：第一次需要并行时创建，之后反复复用；
// 线程数变化（SetThreadCount）时重建。
// 调用方主线程也认领一份区间，不空等。

#include "dqm_llama/thread_pool.h"

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace dqm {

namespace {

// 用户指定的线程数覆盖值；0 表示用 hardware_concurrency 默认。
int g_thread_count = 0;

// 常驻 worker 池。线程总数（含调用方主线程）= 构造时传入的 n_threads。
class WorkerPool {
 public:
  explicit WorkerPool(int n_threads) : n_threads_(n_threads) {
    workers_.reserve(static_cast<size_t>(n_threads - 1));
    for (int i = 0; i < n_threads - 1; ++i) {
      workers_.emplace_back([this, i] { WorkerLoop(i + 1); });
    }
  }

  ~WorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    wake_cv_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  void Run(int n, const std::function<void(int, int)>& fn) {
    const int chunk = n / n_threads_;
    const int remainder = n % n_threads_;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = &fn;
      pending_ = n_threads_ - 1;
      first_exception_ = nullptr;
      ranges_.assign(static_cast<size_t>(n_threads_), {0, 0});
      int start = 0;
      for (int t = 0; t < n_threads_; ++t) {
        // 前 remainder 个线程各多拿 1 个元素，凑满 n。
        const int count = chunk + (t < remainder ? 1 : 0);
        ranges_[static_cast<size_t>(t)] = {start, start + count};
        start += count;
      }
      ++generation_;
    }
    wake_cv_.notify_all();

    RunRange(0);  // 主线程认领第一份

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return pending_ == 0; });
    const std::exception_ptr error = first_exception_;
    job_ = nullptr;
    lock.unlock();

    if (error) {
      std::rethrow_exception(error);
    }
  }

 private:
  void RunRange(int index) {
    const std::function<void(int, int)>* job = nullptr;
    int begin = 0;
    int end = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job = job_;
      begin = ranges_[static_cast<size_t>(index)].first;
      end = ranges_[static_cast<size_t>(index)].second;
    }
    if (job == nullptr || end <= begin) {
      return;
    }
    try {
      (*job)(begin, end);
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!first_exception_) {
        first_exception_ = std::current_exception();
      }
    }
  }

  void WorkerLoop(int index) {
    int seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      wake_cv_.wait(lock, [this, &seen] {
        return stop_ || generation_ != seen;
      });
      if (stop_) {
        return;
      }
      seen = generation_;
      const std::pair<int, int> range = ranges_[static_cast<size_t>(index)];
      const std::function<void(int, int)>* job = job_;
      lock.unlock();

      if (job != nullptr && range.second > range.first) {
        try {
          (*job)(range.first, range.second);
        } catch (...) {
          // 工作线程里抛异常不能穿越线程边界，先存下来；
          // 只保留第一个（其余丢弃），由主线程统一重抛。
          std::lock_guard<std::mutex> guard(mutex_);
          if (!first_exception_) {
            first_exception_ = std::current_exception();
          }
        }
      }

      lock.lock();
      if (--pending_ == 0) {
        done_cv_.notify_one();
      }
    }
  }

  const int n_threads_;
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable wake_cv_;
  std::condition_variable done_cv_;
  bool stop_ = false;
  int generation_ = 0;
  int pending_ = 0;
  const std::function<void(int, int)>* job_ = nullptr;
  std::vector<std::pair<int, int>> ranges_;
  std::exception_ptr first_exception_;
};

std::mutex g_pool_mutex;
std::unique_ptr<WorkerPool> g_pool;
int g_pool_threads = 0;

WorkerPool& GetPool(int n_threads) {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (!g_pool || g_pool_threads != n_threads) {
    g_pool.reset();  // 旧池析构会 join 掉已经在等待的 worker
    g_pool = std::make_unique<WorkerPool>(n_threads);
    g_pool_threads = n_threads;
  }
  return *g_pool;
}

}  // namespace

int GetThreadCount() {
  if (g_thread_count > 0) {
    return g_thread_count;
  }
  // 缓存首次探测：hardware_concurrency() 在大核数宿主 + 容器配额下
  // 单次可达数百微秒，而 ParallelFor 每次调用都会问一次。
  static const int kAutoThreads = [] {
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 4;
  }();
  return kAutoThreads;
}

void SetThreadCount(int n) { g_thread_count = n > 0 ? n : 0; }

void ParallelFor(int n, const std::function<void(int begin, int end)>& fn) {
  if (n <= 0) {
    return;
  }

  int n_threads = GetThreadCount();

  // 每个线程至少分到 16 个元素才值得并行；否则退回单线程。
  constexpr int kMinChunk = 16;
  if (n < n_threads * kMinChunk) {
    fn(0, n);
    return;
  }

  // 线程数不超过任务数，否则会出现空区间。
  if (n_threads > n) {
    n_threads = n;
  }
  if (n_threads <= 1) {
    fn(0, n);
    return;
  }

  GetPool(n_threads).Run(n, fn);
}

}  // namespace dqm
