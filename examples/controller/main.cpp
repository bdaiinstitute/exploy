// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

/**
 * @file main.cpp
 * @brief Loopback controller example for an IsaacLab-exported ONNX policy.
 *
 * This example demonstrates how to wire the exploy OnnxRLController to concrete
 * implementations of the three required interfaces and run a closed-loop simulation
 * for a configurable number of cycles.
 *
 * Loopback semantics
 * ------------------
 * The LoopbackRobotStateInterface feeds commanded joint targets (position, velocity,
 * effort) back as the measured joint state in the following cycle. This models a
 * perfect, zero-delay actuator and is useful for verifying that an exported policy
 * produces sensible action sequences without a full physics simulation.
 *
 * Data collection
 * ---------------
 * This example uses a no-op data collection interface.
 *
 * Usage
 * -----
 *   loopback_controller_example  <onnx_model_path>
 *                               [--cycles   N]       (default 100)
 *                               [--vx       M_S]     (default 0.5)
 *                               [--vy       M_S]     (default 0.0)
 *                               [--omega    RAD_S]   (default 0.0)
 */

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "exploy/components.hpp"
#include "exploy/controller.hpp"
#include "exploy/logging_interface.hpp"
#include "exploy/matcher.hpp"

#include "fixed_command_interface.hpp"
#include "loopback_state_interface.hpp"
#include "memory_visualizer.hpp"

struct Args {
  std::string onnx_path;
  int num_cycles{100};
  double vx{0.5};
  double vy{0.0};
  double omega{0.0};
};

class NoOpDataCollection : public exploy::control::DataCollectionInterface {
 public:
  bool registerDataSource(const std::string&, std::span<const double>) override { return true; }
  bool registerDataSource(const std::string&, std::span<const float>) override { return true; }
  bool registerDataSource(const std::string&, const double&) override { return true; }
  bool collectData(uint64_t /*time_us*/) override { return true; }
};

class CustomBodyPositionMatcher : public exploy::control::Matcher {
 public:
  CustomBodyPositionMatcher() : Matcher("CustomBodyPositionMatcher") {}

  bool matches(const exploy::control::Match& maybe_match) override {
    std::smatch match;
    if (std::regex_match(maybe_match.name, match, kPattern) && match.size() > 2) {
      // Key by the full tensor name so identical body names across different
      // articulations don't overwrite each other.
      found_matches_[maybe_match.name] = maybe_match;
      return true;
    }
    return false;
  }

  std::vector<std::unique_ptr<exploy::control::Input>> createInputs() const override {
    std::vector<std::unique_ptr<exploy::control::Input>> inputs;
    for (const auto& [tensor_name, found_match] : found_matches_) {
      // Re-derive the articulation and body names from the tensor name (the key).
      std::smatch match;
      if (!std::regex_match(tensor_name, match, kPattern) || match.size() <= 2) continue;
      inputs.push_back(std::make_unique<exploy::control::BodyPositionInput>(
          found_match.name, match[1].str(), match[2].str()));
    }
    return inputs;
  }

 private:
  static inline const std::regex kPattern{
      R"(custom.obj\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\.pos_b_rt_w_in_w)"};
};

[[nodiscard]] std::optional<Args> parseArgs(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <onnx_model_path>"
                 " [--cycles N] [--vx M_S] [--vy M_S] [--omega RAD_S]\n";
    return std::nullopt;
  }

  Args args;
  args.onnx_path = argv[1];

  for (int i = 2; i < argc; ++i) {
    const std::string flag = argv[i];
    const bool has_next = (i + 1) < argc;
    if (flag == "--cycles" && has_next) {
      args.num_cycles = std::stoi(argv[++i]);
    } else if (flag == "--vx" && has_next) {
      args.vx = std::stod(argv[++i]);
    } else if (flag == "--vy" && has_next) {
      args.vy = std::stod(argv[++i]);
    } else if (flag == "--omega" && has_next) {
      args.omega = std::stod(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << flag << "\n";
      return std::nullopt;
    }
  }
  return args;
}

int main(int argc, char** argv) {
  auto maybe_args = parseArgs(argc, argv);
  if (!maybe_args.has_value()) return 1;
  const Args& args = *maybe_args;

  // Optional: set a custom logger. By default, the controller will log to stdout with a simple
  // logger that prefixes log messages with their level (e.g. "[ERROR]", "[WARNING]", etc.).
  exploy::control::StdoutLogger logger;
  exploy::control::setLogger(&logger);

  // Create a RobotStateInterface.
  exploy::control::examples::LoopbackRobotStateInterface state;

  // Create a CommandInterface.
  exploy::control::examples::FixedCommandInterface command(
      exploy::control::examples::FixedCommandConfig{.se2_velocity{args.vx, args.vy, args.omega}});

  // Create a DataCollectionInterface.
  NoOpDataCollection data_collection;

  // Create the controller.
  exploy::control::OnnxRLController controller(state, command, data_collection);
  // Register custom matchers.
  controller.context().registerMatcher(std::make_unique<CustomBodyPositionMatcher>());

  // Register a read-only observer matcher that visualizes recurrent memory state each step.
  // Observation does not claim tensors functionally, so it coexists with the built-in
  // MemoryMatcher that owns the same memory.<key>.in / memory.<key>.out tensors.
  controller.context().registerMatcher(
      std::make_unique<exploy::control::examples::MemoryVisualizationMatcher>());

  // Load the ONNX model.
  if (!controller.create(args.onnx_path)) {
    std::cerr << "[main] Failed to load ONNX model: " << args.onnx_path << "\n";
    return 1;
  }

  const int model_rate = controller.context().updateRate();
  const double update_rate_hz = static_cast<double>(model_rate);

  // Initialize the controller (calls init() on all components).
  if (!controller.init(/*enable_data_collection=*/false)) {
    std::cerr << "[main] Controller initialisation failed.\n";
    return 1;
  }

  const std::chrono::duration<double> dt(update_rate_hz > 0.0 ? 1.0 / update_rate_hz : 0.0);
  const uint64_t dt_us = static_cast<uint64_t>(dt.count() * 1e6);

  uint64_t time_us = 0;
  int failures = 0;

  for (int cycle = 0; cycle < args.num_cycles; ++cycle) {
    auto cycle_start = std::chrono::steady_clock::now();

    // Run one controller step (read state → infer → write commands).
    if (!controller.update(time_us)) {
      std::cerr << "[main] Cycle " << cycle << " FAILED\n";
      ++failures;
    }

    time_us += dt_us;

    // Rate-limit the loop if an update rate is specified.
    if (update_rate_hz > 0.0) {
      auto elapsed = std::chrono::steady_clock::now() - cycle_start;
      if (elapsed < dt) {
        std::this_thread::sleep_for(dt - elapsed);
      }
    }
  }

  return failures > 0 ? 1 : 0;
}
