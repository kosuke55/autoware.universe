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

  p.min_stop_distance = dp("min_stop_distance", 5.0);
  p.stop_time = dp("stop_time", 2.0);
  p.hysteresis_buffer_distance = dp("hysteresis_buffer_distance", 2.0);
  p.pull_out_prepare_duration = dp("pull_out_prepare_duration", 2.0);
  p.pull_out_duration = dp("pull_out_duration", 4.0);
  p.pull_out_finish_judge_buffer = dp("pull_out_finish_judge_buffer", 1.0);
  p.minimum_pull_out_velocity = dp("minimum_pull_out_velocity", 8.3);
  p.prediction_duration = dp("prediction_duration", 8.0);
  p.prediction_time_resolution = dp("prediction_time_resolution", 0.5);
  p.static_obstacle_velocity_thresh = dp("static_obstacle_velocity_thresh", 0.1);
  p.maximum_deceleration = dp("maximum_deceleration", 1.0);
  p.pull_out_sampling_num = dp("pull_out_sampling_num", 4);
  p.enable_collision_check_at_prepare_phase = dp("enable_collision_check_at_prepare_phase", true);
  p.use_predicted_path_outside_lanelet = dp("use_predicted_path_outside_lanelet", true);
  p.use_all_predicted_path = dp("use_all_predicted_path", false);
  p.use_dynamic_object = dp("use_dynamic_object", false);
  p.enable_blocked_by_obstacle = dp("enable_blocked_by_obstacle", false);
  p.pull_out_search_distance = dp("pull_out_search_distance", 30.0);
  p.after_pull_out_straight_distance = dp("after_pull_out_straight_distance", 3.0);
  p.before_pull_out_straight_distance = dp("before_pull_out_straight_distance", 3.0);
  p.maximum_lateral_jerk = dp("maximum_lateral_jerk", 3.0);
  p.minimum_lateral_jerk = dp("minimum_lateral_jerk", 1.0);
  p.deceleration_interval = dp("deceleration_interval", 10.0);

  // validation of parameters
  if (p.pull_out_sampling_num < 1) {
    RCLCPP_FATAL_STREAM(
      logger_, "pull_out_sampling_num must be positive integer. Given parameter: "
                 << p.pull_out_sampling_num << std::endl
                 << "Terminating the program...");
    exit(EXIT_FAILURE);
  }
  if (p.maximum_deceleration < 0.0) {
    RCLCPP_FATAL_STREAM(
      logger_, "maximum_deceleration cannot be negative value. Given parameter: "
                 << p.maximum_deceleration << std::endl
                 << "Terminating the program...");
    exit(EXIT_FAILURE);
  }
}
}  // namespace behavior_path_planner
