// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#include "exploy/worker.hpp"

#include "exploy/logging_utils.hpp"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <cerrno>
#include <cstring>
#endif

#include <cmath>
#include <utility>

namespace exploy::control {

// Worker

bool Worker::setCallbacks(std::function<bool()> read_fn, std::function<bool()> work_fn,
                          std::function<bool()> write_fn) {
  if (!read_fn || !work_fn || !write_fn) {
    LOG(ERROR, "All callbacks must be non-null");
    return false;
  }
  read_fn_ = std::move(read_fn);
  work_fn_ = std::move(work_fn);
  write_fn_ = std::move(write_fn);
  return true;
}

// SyncWorker

SyncWorker::SyncWorker(double update_rate_hz) {
  period_ms_ = static_cast<uint64_t>(std::lround(1000.0 / update_rate_hz));
}

void SyncWorker::reset() {
  last_scheduled_update_us_ = 0;
  first_run_ = true;
}

bool SyncWorker::update(uint64_t time_us) {
  if (!read_fn_ || !work_fn_ || !write_fn_) {
    LOG(ERROR, "SyncWorker: callbacks not set. Call setCallbacks() before update().");
    return false;
  }

  if (first_run_) {
    last_scheduled_update_us_ = time_us - period_ms_ * 1000;
    first_run_ = false;
  }

  if (std::lround(static_cast<double>(time_us - last_scheduled_update_us_) / 1000.0) >=
      static_cast<long>(period_ms_)) {
    if (!read_fn_()) return false;
    if (!work_fn_()) return false;
    if (!write_fn_()) return false;

    last_scheduled_update_us_ += period_ms_ * 1000;
  }
  return true;
}

// AsyncWorker

AsyncWorker::AsyncWorker(double update_rate_hz, ThreadSchedulingOptions scheduling)
    : scheduling_(std::move(scheduling)) {
  period_ms_ = static_cast<uint64_t>(std::lround(1000.0 / update_rate_hz));
}

AsyncWorker::~AsyncWorker() {
  stopWorker();
}

void AsyncWorker::startWorker() {
  if (!thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = false;
      working_ = false;
      work_requested_ = false;
      work_finished_ = false;
      work_successful_ = true;
    }
    thread_ = std::thread(&AsyncWorker::threadLoop, this);
  }
}

void AsyncWorker::stopWorker() {
  if (thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }
}

void AsyncWorker::reset() {
  stopWorker();
  last_scheduled_update_us_ = 0;
  first_run_ = true;
  consecutive_overruns_ = 0;
  faulted_ = false;
}

bool AsyncWorker::update(uint64_t time_us) {
  if (!read_fn_ || !work_fn_ || !write_fn_) {
    LOG(ERROR, "AsyncWorker: callbacks not set. Call setCallbacks() before update().");
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (faulted_) return false;
  }

  if (first_run_) {
    startWorker();
    last_scheduled_update_us_ = time_us - period_ms_ * 1000;
    first_run_ = false;
  }

  // Consume finished work from previous cycle
  {
    bool finished = false;
    bool successful = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (work_finished_) {
        finished = true;
        successful = work_successful_;
        work_finished_ = false;
      }
    }
    if (finished) {
      if (!successful) return false;
      if (!write_fn_()) return false;
    }
  }

  // Trigger new cycle if period elapsed
  if (std::lround(static_cast<double>(time_us - last_scheduled_update_us_) / 1000.0) >=
      static_cast<long>(period_ms_)) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (work_requested_ || working_) {
        ++consecutive_overruns_;
        if (consecutive_overruns_ % 10 == 0) {
          LOG(WARN,
              "AsyncWorker: cycle overrun — inference still running at cycle boundary "
              "(%lu consecutive skipped).",
              consecutive_overruns_);
        }
        return true;
      }
    }

    if (!read_fn_()) return false;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      work_requested_ = true;
    }
    cv_.notify_one();

    last_scheduled_update_us_ += period_ms_ * 1000;
    consecutive_overruns_ = 0;
  }

  return true;
}

bool AsyncWorker::configureThread() {
#ifndef __linux__
  if (!scheduling_.cpu_affinity.empty() || scheduling_.policy != SchedulingPolicy::NORMAL ||
      scheduling_.priority != 0) {
    LOG(DEBUG,
        "AsyncWorker: thread scheduling options are unsupported on this platform and will be "
        "ignored.");
  }
  return true;
#else
  if (!scheduling_.cpu_affinity.empty()) {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (unsigned cpu : scheduling_.cpu_affinity) {
      if (cpu >= CPU_SETSIZE) {
        LOG(ERROR, "AsyncWorker: CPU %u exceeds CPU_SETSIZE (%d).", cpu, CPU_SETSIZE);
        return false;
      }
      CPU_SET(cpu, &cpu_set);
    }
    const int error = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);
    if (error != 0) {
      LOG(ERROR, "AsyncWorker: failed to set CPU affinity: %s", std::strerror(error));
      return false;
    }
  }

  if (scheduling_.policy == SchedulingPolicy::NORMAL && scheduling_.priority == 0) return true;

  int policy = SCHED_OTHER;
  switch (scheduling_.policy) {
    case SchedulingPolicy::NORMAL:
      break;
    case SchedulingPolicy::FIFO:
      policy = SCHED_FIFO;
      break;
    case SchedulingPolicy::ROUND_ROBIN:
      policy = SCHED_RR;
      break;
  }

  errno = 0;
  const int min_priority = sched_get_priority_min(policy);
  const int max_priority = sched_get_priority_max(policy);
  if (min_priority == -1 || max_priority == -1) {
    LOG(ERROR, "AsyncWorker: failed to query scheduler priority range: %s", std::strerror(errno));
    return false;
  }
  if (scheduling_.priority < min_priority || scheduling_.priority > max_priority) {
    LOG(ERROR, "AsyncWorker: priority %d is outside the valid range [%d, %d].",
        scheduling_.priority, min_priority, max_priority);
    return false;
  }

  sched_param parameters{.sched_priority = scheduling_.priority};
  const int error = pthread_setschedparam(pthread_self(), policy, &parameters);
  if (error != 0) {
    LOG(ERROR, "AsyncWorker: failed to set scheduler: %s", std::strerror(error));
    return false;
  }
  return true;
#endif
}

void AsyncWorker::threadLoop() {
  if (!configureThread()) {
    std::lock_guard<std::mutex> lock(mutex_);
    faulted_ = true;
    return;
  }

  while (true) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() {
        return stop_ || work_requested_;
      });
      if (stop_) return;
      work_requested_ = false;
      working_ = true;
    }

    bool success = work_fn_();
    if (!success) {
      LOG(ERROR, "AsyncWorker: Work function failed.");
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      working_ = false;
      work_finished_ = true;
      work_successful_ = success;
      if (!success) faulted_ = true;
    }
  }
}

}  // namespace exploy::control
