// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace exploy::control {

/**
 * @brief Scheduling policy for an asynchronous worker thread.
 *
 * @note Only supported on Linux.
 */
enum class SchedulingPolicy {
  NORMAL,
  FIFO,
  ROUND_ROBIN,
};

/**
 * @brief Optional CPU affinity and real-time scheduling for an asynchronous worker.
 *
 * @note Only supported on Linux.
 */
struct ThreadSchedulingOptions {
  /** CPUs on which the thread may run. Empty preserves the inherited affinity. */
  std::vector<unsigned> cpu_affinity;
  /** Scheduler to use. NORMAL with priority 0 preserves the inherited scheduler. */
  SchedulingPolicy policy = SchedulingPolicy::NORMAL;
  /** Scheduler priority. FIFO and ROUND_ROBIN priorities are platform-dependent. */
  int priority = 0;
};

/**
 * @brief Abstract base class for controller execution strategies.
 *
 * A Worker owns the read → work → write pipeline. Concrete subclasses decide
 * *when* and *how* the three callbacks are invoked:
 *
 * - `read_fn`  — called on the main thread to snapshot robot state.
 * - `work_fn`  — runs the ONNX inference.
 * - `write_fn` — called on the main thread to dispatch joint targets.
 *
 * Use `setCallbacks()` to register all three before calling `update()`.
 */
class Worker {
 public:
  virtual ~Worker() = default;

  /** @brief Reset the worker to its initial state. */
  virtual void reset() {}

  /**
   * @brief Advance the control pipeline by one tick.
   *
   * @param time_us Current timestamp in microseconds.
   * @return `true` if the update succeeded or was safely skipped, `false` on error.
   */
  virtual bool update(uint64_t time_us) = 0;

  /**
   * @brief Register the read / work / write callbacks.
   *
   * Must be called exactly once before the first `update()`. All three
   * arguments must be non-null; the function returns `false` otherwise.
   *
   * @param read_fn  Reads observations from the robot state (main thread).
   * @param work_fn  Runs ONNX inference (may execute on a background thread).
   * @param write_fn Writes joint targets back to the robot (main thread).
   * @return `true` on success.
   */
  bool setCallbacks(std::function<bool()> read_fn, std::function<bool()> work_fn,
                    std::function<bool()> write_fn);

 protected:
  std::function<bool()> read_fn_;
  std::function<bool()> work_fn_;
  std::function<bool()> write_fn_;
};

/**
 * @brief Synchronous worker — runs read → work → write on the calling thread.
 *
 * All three callbacks execute inline inside `update()`. The call blocks until
 * inference is complete, so the caller's thread must afford the full inference
 * latency every control cycle.
 *
 * Phase is maintained across updates: the first call initialises the phase
 * reference and subsequent calls fire whenever the elapsed time exceeds the
 * configured period.
 */
class SyncWorker : public Worker {
 public:
  /**
   * @param update_rate_hz Desired control frequency in Hz.
   */
  explicit SyncWorker(double update_rate_hz);

  bool update(uint64_t time_us) override;
  void reset() override;

 private:
  uint64_t period_ms_;
  uint64_t last_scheduled_update_us_ = 0;
  bool first_run_ = true;
};

/**
 * @brief Asynchronous worker — offloads ONNX inference to a background thread.
 *
 * The pipeline is split across two consecutive `update()` calls:
 *
 * 1. **First call at cycle boundary** — `read_fn` executes on the main thread,
 *    then `work_fn` is dispatched to a dedicated background thread.
 * 2. **Subsequent calls** — if inference has finished, `write_fn` runs on the
 *    main thread to commit joint targets.  If the worker is still busy
 *    (overrun), the cycle is skipped and `update()` returns `true`.
 *
 * This decouples the main control loop from the inference latency, allowing
 * the robot to keep receiving state updates while the GPU or CPU is busy.
 *
 * Thread safety: the internal mutex guards all shared state between the main
 * thread and the worker thread.
 *
 * `reset()` stops the background thread and clears all state. The thread is
 * re-started on the next `update()` call.
 *
 * **Error handling**: when `work_fn` returns false the worker latches into a
 * faulted state. All subsequent `update()` calls return `false` immediately
 * without dispatching new work. Call `reset()` to clear the fault and resume.
 */
class AsyncWorker : public Worker {
 public:
  /**
   * @param update_rate_hz Desired control frequency in Hz.
   * @param scheduling Optional CPU affinity and scheduler settings for the background thread.
   */
  AsyncWorker(double update_rate_hz, ThreadSchedulingOptions scheduling = {});
  ~AsyncWorker() override;

  void reset() override;
  bool update(uint64_t time_us) override;

 private:
  void startWorker();
  void stopWorker();
  void threadLoop();
  bool configureThread();

  // Main-thread-only — no synchronization needed.
  uint64_t period_ms_;
  const ThreadSchedulingOptions scheduling_;
  uint64_t last_scheduled_update_us_ = 0;
  bool first_run_ = true;
  uint64_t consecutive_overruns_ = 0;
  std::thread thread_;

  // Shared between the main thread and the worker thread — all guarded by mutex_.
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_ = false;
  bool working_ = false;
  bool work_requested_ = false;
  bool work_finished_ = false;
  bool work_successful_ = true;
  bool faulted_ = false;
};

}  // namespace exploy::control
