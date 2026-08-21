// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

#include "exploy/worker.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef __linux__
#include <sched.h>
#endif

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace exploy::control {
namespace {

using ::testing::Return;

/// Waits until @p flag becomes true, or fails the test if @p timeout elapses.
void waitOrFail(const std::atomic<bool>& flag,
                std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!flag) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "Timed out waiting for worker";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

/// Retries worker.update(@p time_us) until it returns false, or fails the test
/// if @p timeout elapses without that happening.
void retryUpdateUntilFail(Worker& worker, uint64_t time_us,
                          std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (worker.update(time_us)) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline)
        << "Timed out waiting for update() to return false";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

class MockCallbacks {
 public:
  MOCK_METHOD(bool, Read, (), ());
  MOCK_METHOD(bool, Work, (), ());
  MOCK_METHOD(bool, Write, (), ());
};

class WorkerTest : public ::testing::Test {
 protected:
  MockCallbacks callbacks_;

  void SetupWorker(Worker& worker) {
    auto success = worker.setCallbacks(
        [this]() {
          return callbacks_.Read();
        },
        [this]() {
          return callbacks_.Work();
        },
        [this]() {
          return callbacks_.Write();
        });
    ASSERT_TRUE(success);
  }
};

TEST_F(WorkerTest, SyncWorker_UpdateTrigger) {
  // Rate 10Hz -> Period 100ms = 100000us
  SyncWorker worker(10.0);
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));

  // First call runs immediately
  EXPECT_TRUE(worker.update(1000000));

  // Second call within period should skip
  EXPECT_TRUE(worker.update(1000050));

  // Third call after period should run
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));
  EXPECT_TRUE(worker.update(1100000));
}

TEST_F(WorkerTest, SyncWorker_PhaseMaintenance) {
  SyncWorker worker(10.0);  // 100ms period
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));
  worker.update(1000000);

  // After period elapses should run again
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));
  worker.update(1100050);

  // Phase advances, next due at 1200000
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));
  worker.update(1200000);
}

TEST_F(WorkerTest, AsyncWorker_BasicFlow) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  // 1. First update: read + trigger work; no write yet
  std::atomic<bool> work_done{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done = true;
    return true;
  });
  EXPECT_CALL(callbacks_, Write()).Times(0);

  worker.update(1000000);

  waitOrFail(work_done);

  // 2. Mid-period update: write consumed, no new read
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Read()).Times(0);
  worker.update(1000050);

  // 3. Next period: read + trigger work; no write (already consumed above)
  std::atomic<bool> work_done2{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done2 = true;
    return true;
  });
  EXPECT_CALL(callbacks_, Write()).Times(0);
  worker.update(1100000);

  waitOrFail(work_done2);
}

TEST_F(WorkerTest, AsyncWorker_ThreadExecution) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  std::thread::id main_id = std::this_thread::get_id();
  std::promise<std::thread::id> work_id_promise;
  auto work_id_future = work_id_promise.get_future();

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_id_promise.set_value(std::this_thread::get_id());
    return true;
  });

  worker.update(1000000);

  ASSERT_EQ(work_id_future.wait_for(std::chrono::milliseconds(1000)), std::future_status::ready)
      << "Timed out waiting for worker thread id";
  std::thread::id work_id = work_id_future.get();

  EXPECT_NE(main_id, work_id);
  EXPECT_NE(work_id, std::thread::id());
}

#ifdef __linux__
TEST_F(WorkerTest, AsyncWorker_Affinity) {
  cpu_set_t available_cpus;
  CPU_ZERO(&available_cpus);
  ASSERT_EQ(sched_getaffinity(0, sizeof(available_cpus), &available_cpus), 0);

  int target_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &available_cpus)) {
      target_cpu = cpu;
      break;
    }
  }
  ASSERT_GE(target_cpu, 0);

  AsyncWorker worker(10.0, {.cpu_affinity = {static_cast<unsigned>(target_cpu)}});
  SetupWorker(worker);

  std::promise<int> work_cpu_promise;
  auto work_cpu_future = work_cpu_promise.get_future();
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_cpu_promise.set_value(sched_getcpu());
    return true;
  });

  EXPECT_TRUE(worker.update(1000000));
  ASSERT_EQ(work_cpu_future.wait_for(std::chrono::milliseconds(1000)), std::future_status::ready)
      << "Timed out waiting for worker CPU";
  EXPECT_EQ(work_cpu_future.get(), target_cpu);
}

TEST_F(WorkerTest, AsyncWorker_InvalidPriorityFaults) {
  AsyncWorker worker(10.0, {.priority = 1});
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).Times(0);

  EXPECT_TRUE(worker.update(1000000));
  retryUpdateUntilFail(worker, 1000050);
}
#endif

TEST_F(WorkerTest, AsyncWorker_Overrun) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  std::atomic<bool> work_started{false};
  std::atomic<bool> allow_finish{false};

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_started = true;
    while (!allow_finish) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  });

  EXPECT_TRUE(worker.update(1000000));
  waitOrFail(work_started);

  // Overrun: period elapsed but worker is busy — should skip and return true
  EXPECT_TRUE(worker.update(1100000));

  // Catch-up: after work finishes, next update should consume write AND dispatch a new
  // cycle immediately (accumulated lag = 1 period, no extra wait needed)
  std::atomic<bool> work_done2{false};
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done2 = true;
    return true;
  });
  allow_finish = true;

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(1000);
  while (!work_done2) {
    ASSERT_LT(std::chrono::steady_clock::now(), deadline) << "Timed out waiting for catch-up cycle";
    worker.update(1100000);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

TEST_F(WorkerTest, AsyncWorker_WorkFailure) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  std::atomic<bool> work_done{false};

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done = true;
    return false;
  });
  EXPECT_CALL(callbacks_, Write()).Times(0);

  EXPECT_TRUE(worker.update(1000000));

  // Poll until the worker has published its failure, then verify the fault latch:
  // update() must return false and keep returning false without dispatching new work.
  retryUpdateUntilFail(worker, 1000050);
  EXPECT_FALSE(worker.update(1100000));
  EXPECT_FALSE(worker.update(1200000));
}

TEST_F(WorkerTest, AsyncWorker_ReadFailure) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(false));
  EXPECT_CALL(callbacks_, Work()).Times(0);
  EXPECT_CALL(callbacks_, Write()).Times(0);

  EXPECT_FALSE(worker.update(1000000));
}

TEST_F(WorkerTest, AsyncWorker_WriteFailure) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  std::atomic<bool> work_done{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done = true;
    return true;
  });
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(false));

  EXPECT_TRUE(worker.update(1000000));
  waitOrFail(work_done);

  // Next update consumes finished work and calls write_fn_, which fails.
  retryUpdateUntilFail(worker, 1000050);
}

TEST_F(WorkerTest, SyncWorker_ReadFailure) {
  SyncWorker worker(10.0);
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(false));
  EXPECT_FALSE(worker.update(1000000));
}

TEST_F(WorkerTest, SyncWorker_WorkFailure) {
  SyncWorker worker(10.0);
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(false));
  EXPECT_FALSE(worker.update(1000000));
}

TEST_F(WorkerTest, SyncWorker_WriteFailure) {
  SyncWorker worker(10.0);
  SetupWorker(worker);

  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Write()).WillOnce(Return(false));
  EXPECT_FALSE(worker.update(1000000));
}

TEST_F(WorkerTest, AsyncWorker_FaultClearedByReset) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  std::atomic<bool> work_done{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done = true;
    return false;
  });

  EXPECT_TRUE(worker.update(1000000));
  retryUpdateUntilFail(worker, 1000050);

  worker.reset();

  // After reset the worker must accept new work.
  std::atomic<bool> work_done2{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done2 = true;
    return true;
  });
  EXPECT_TRUE(worker.update(2000000));
  waitOrFail(work_done2);
}

TEST_F(WorkerTest, AsyncWorker_Reset) {
  AsyncWorker worker(10.0);
  SetupWorker(worker);

  std::atomic<bool> work_done{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done = true;
    return true;
  });
  worker.update(1000000);
  waitOrFail(work_done);

  // reset() must stop the background thread cleanly.
  worker.reset();

  // After reset a new cycle should start from a clean state.
  std::atomic<bool> work_done2{false};
  EXPECT_CALL(callbacks_, Read()).WillOnce(Return(true));
  EXPECT_CALL(callbacks_, Work()).WillOnce([&]() {
    work_done2 = true;
    return true;
  });
  EXPECT_TRUE(worker.update(2000000));
  waitOrFail(work_done2);
}

}  // namespace
}  // namespace exploy::control
