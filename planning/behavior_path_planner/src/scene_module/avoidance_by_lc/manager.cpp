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

#include "behavior_path_planner/scene_module/avoidance_by_lc/manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

namespace behavior_path_planner
{

AvoidanceByLCModuleManager::AvoidanceByLCModuleManager(
  rclcpp::Node * node, const std::string & name, const size_t max_module_num, const size_t priority,
  const bool enable_simultaneous_execution)
: SceneModuleManagerInterface(node, name, max_module_num, priority, enable_simultaneous_execution)
{
  rtc_interface_left_ = std::make_shared<RTCInterface>(node, name + "_left");
  rtc_interface_right_ = std::make_shared<RTCInterface>(node, name + "_right");
  getModuleParams(node);
}

void AvoidanceByLCModuleManager::updateModuleParams(
  const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  auto p = module_params_;

  std::string ns = name_ + ".";
  updateParam<bool>(parameters, ns + "publish_debug_marker", p->publish_debug_marker);

  std::for_each(registered_modules_.begin(), registered_modules_.end(), [&p](const auto & m) {
    m.second->updateModuleParams(p);
  });
}

void AvoidanceByLCModuleManager::getModuleParams(rclcpp::Node * node)
{
  using tier4_autoware_utils::deg2rad;

  AvoidanceByLCParameters p{};

  // avoidance parameters
  // general params
  {
    std::string ns = name_ + ".";
    p.resample_interval_for_planning =
      node->declare_parameter<double>(ns + "resample_interval_for_planning");
    p.resample_interval_for_output =
      node->declare_parameter<double>(ns + "resample_interval_for_output");
    p.detection_area_right_expand_dist =
      node->declare_parameter<double>(ns + "detection_area_right_expand_dist");
    p.detection_area_left_expand_dist =
      node->declare_parameter<double>(ns + "detection_area_left_expand_dist");
    p.object_envelope_buffer = node->declare_parameter<double>(ns + "object_envelope_buffer");
    p.enable_avoidance_over_same_direction =
      node->declare_parameter<bool>(ns + "enable_avoidance_over_same_direction");
    p.enable_avoidance_over_opposite_direction =
      node->declare_parameter<bool>(ns + "enable_avoidance_over_opposite_direction");
    p.enable_avoidance_all_parking_vehicle =
      node->declare_parameter<bool>(ns + "enable_avoidance_all_parking_vehicle");
    p.enable_avoidance_yield = node->declare_parameter<bool>(ns + "enable_avoidance_yield");
    p.enable_slow_down = node->declare_parameter<bool>(ns + "enable_slow_down");
    p.publish_debug_marker = node->declare_parameter<bool>(ns + "publish_debug_marker");
    p.print_debug_info = node->declare_parameter<bool>(ns + "print_debug_info");
    p.print_processing_time = node->declare_parameter<bool>(ns + "print_processing_time");
    p.min_executable_distance = node->declare_parameter<double>(ns + "min_executable_distance");
  }

  // target filtering
  {
    std::string ns = name_ + ".target_filtiering.";
    p.threshold_distance_object_is_on_center =
      node->declare_parameter<double>(ns + "threshold_distance_object_is_on_center");
    p.threshold_speed_object_is_stopped =
      node->declare_parameter<double>(ns + "threshold_speed_object_is_stopped");
    p.threshold_time_object_is_moving =
      node->declare_parameter<double>(ns + "threshold_time_object_is_moving");
    p.threshold_time_force_avoidance =
      node->declare_parameter<double>(ns + "threshold_time_force_avoidance");
    p.threshold_intersection_force_avoidance =
      node->declare_parameter<double>(ns + "threshold_intersection_force_avoidance");
    p.object_check_forward_distance =
      node->declare_parameter<double>(ns + "object_check_forward_distance");
    p.object_check_backward_distance =
      node->declare_parameter<double>(ns + "object_check_backward_distance");
    p.object_check_overhang = node->declare_parameter<double>(ns + "object_check_overhang");
    p.object_check_yaw = node->declare_parameter<double>(ns + "object_check_yaw");
    p.object_check_road_shoulder_ratio =
      node->declare_parameter<double>(ns + "object_check_road_shoulder_ratio");
    p.object_last_seen_threshold =
      node->declare_parameter<double>(ns + "object_last_seen_threshold");
    p.adjacent_lane_check_backward_distance =
      node->declare_parameter<double>(ns + "adjacent_lane_check_backward_distance");
    p.minimum_road_shoulder_width =
      node->declare_parameter<double>(ns + "minimum_road_shoulder_width");
  }

  // target object
  {
    std::string ns = name_ + ".target_object.";
    p.avoid_car = node->declare_parameter<bool>(ns + "car");
    p.avoid_truck = node->declare_parameter<bool>(ns + "truck");
    p.avoid_bus = node->declare_parameter<bool>(ns + "bus");
    p.avoid_trailer = node->declare_parameter<bool>(ns + "trailer");
    p.avoid_unknown = node->declare_parameter<bool>(ns + "unknown");
    p.avoid_bicycle = node->declare_parameter<bool>(ns + "bicycle");
    p.avoid_motorcycle = node->declare_parameter<bool>(ns + "motorcycle");
    p.avoid_pedestrian = node->declare_parameter<bool>(ns + "pedestrian");
  }

  // lane change parameters
  {
    const auto dp = [this, &node](const std::string & str, auto def_val) {
      std::string name = name_ + "." + str;
      return node->declare_parameter(name, def_val);
    };

    p.min_stop_distance = dp("min_stop_distance", 5.0);
    p.stop_time = dp("stop_time", 2.0);
    p.hysteresis_buffer_distance = dp("hysteresis_buffer_distance", 2.0);
    p.lane_change_prepare_duration = dp("lane_change_prepare_duration", 2.0);
    p.lane_changing_duration = dp("lane_changing_duration", 4.0);
    p.lane_change_finish_judge_buffer = dp("lane_change_finish_judge_buffer", 3.0);
    p.minimum_lane_change_velocity = dp("minimum_lane_change_velocity", 8.3);
    p.prediction_duration = dp("prediction_duration", 8.0);
    p.prediction_time_resolution = dp("prediction_time_resolution", 0.5);
    p.static_obstacle_velocity_thresh = dp("static_obstacle_velocity_thresh", 0.1);
    p.maximum_deceleration = dp("maximum_deceleration", 1.0);
    p.lane_change_sampling_num = dp("lane_change_sampling_num", 10);
    p.enable_abort_lane_change = dp("enable_abort_lane_change", false);
    p.enable_collision_check_at_prepare_phase = dp("enable_collision_check_at_prepare_phase", true);
    p.use_predicted_path_outside_lanelet = dp("use_predicted_path_outside_lanelet", true);
    p.use_all_predicted_path = dp("use_all_predicted_path", false);
    p.abort_lane_change_velocity_thresh = dp("abort_lane_change_velocity_thresh", 0.5);
    p.abort_lane_change_angle_thresh =
      dp("abort_lane_change_angle_thresh", tier4_autoware_utils::deg2rad(10.0));
    p.abort_lane_change_distance_thresh = dp("abort_lane_change_distance_thresh", 0.3);
    p.enable_blocked_by_obstacle = dp("enable_blocked_by_obstacle", false);

    // validation of parameters
    if (p.lane_change_sampling_num < 1) {
      RCLCPP_FATAL_STREAM(
        logger_, "lane_change_sampling_num must be positive integer. Given parameter: "
                   << p.lane_change_sampling_num << std::endl
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

  module_params_ = std::make_shared<AvoidanceByLCParameters>(p);
}
}  // namespace behavior_path_planner
