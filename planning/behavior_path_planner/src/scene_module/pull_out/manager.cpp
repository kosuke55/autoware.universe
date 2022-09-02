// Copyright 2022 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "behavior_path_planner/scene_module/pull_out/manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

namespace behavior_path_planner
{

PullOutModuleManager::PullOutModuleManager(
  rclcpp::Node * node, const std::string & name, const size_t max_module_num, const size_t priority,
  const bool enable_simultaneous_execution)
: SceneModuleManagerInterface(node, name, max_module_num, priority, enable_simultaneous_execution)
{
  rtc_interface_ = std::make_shared<RTCInterface>(node, name);
  getModuleParams(node);
}

void PullOutModuleManager::updateModuleParams(
  [[maybe_unused]] const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  auto & p = module_params_;

  [[maybe_unused]] std::string ns = name_ + ".";

  std::for_each(registered_modules_.begin(), registered_modules_.end(), [&p](const auto & m) {
    m.second->updateModuleParams(p);
  });
}

void PullOutModuleManager::getModuleParams(rclcpp::Node * node)
{
  const auto dp = [this, &node](const std::string & str, auto def_val) {
    std::string name = name_ + "." + str;
    return node->declare_parameter(name, def_val);
  };

  auto & p = module_params_;

  p.th_arrived_distance = dp("th_arrived_distance", 1.0);
  p.th_stopped_velocity = dp("th_stopped_velocity", 0.01);
  p.th_stopped_time = dp("th_stopped_time", 1.0);
  p.collision_check_margin = dp("collision_check_margin", 1.0);
  p.pull_out_finish_judge_buffer = dp("pull_out_finish_judge_buffer", 1.0);
  // shift pull out
  p.enable_shift_pull_out = dp("enable_shift_pull_out", true);
  p.shift_pull_out_velocity = dp("shift_pull_out_velocity", 8.3);
  p.pull_out_sampling_num = dp("pull_out_sampling_num", 4);
  p.before_pull_out_straight_distance = dp("before_pull_out_straight_distance", 3.0);
  p.minimum_shift_pull_out_distance = dp("minimum_shift_pull_out_distance", 20.0);
  p.maximum_lateral_jerk = dp("maximum_lateral_jerk", 3.0);
  p.minimum_lateral_jerk = dp("minimum_lateral_jerk", 1.0);
  p.deceleration_interval = dp("deceleration_interval", 10.0);
  // geometric pull out
  p.enable_geometric_pull_out = dp("enable_geometric_pull_out", true);
  p.geometric_pull_out_velocity = dp("geometric_pull_out_velocity", 1.0);
  p.arc_path_interval = dp("arc_path_interval", 1.0);
  p.lane_departure_margin = dp("lane_departure_margin", 0.2);
  p.backward_velocity = dp("backward_velocity", -0.3);
  p.pull_out_max_steer_angle = dp("pull_out_max_steer_angle", 0.26);  // 15deg
  // search start pose backward
  p.search_priority =
    dp("search_priority", "efficient_path");  // "efficient_path" or "short_back_distance"
  p.enable_back = dp("enable_back", true);
  p.max_back_distance = dp("max_back_distance", 15.0);
  p.backward_search_resolution = dp("backward_search_resolution", 2.0);
  p.backward_path_update_duration = dp("backward_path_update_duration", 3.0);

  // validation of parameters
  if (p.pull_out_sampling_num < 1) {
    RCLCPP_FATAL_STREAM(
      logger_, "pull_out_sampling_num must be positive integer. Given parameter: "
                 << p.pull_out_sampling_num << std::endl
                 << "Terminating the program...");
    exit(EXIT_FAILURE);
  }
}
}  // namespace behavior_path_planner
