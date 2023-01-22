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

#include "behavior_path_planner/scene_module/avoidance/manager.hpp"

#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>
#include <vector>

namespace behavior_path_planner
{

AvoidanceModuleManager::AvoidanceModuleManager(
  rclcpp::Node * node, const std::string & name, const size_t max_module_num, const size_t priority,
  const bool enable_simultaneous_execution)
: SceneModuleManagerInterface(node, name, max_module_num, priority, enable_simultaneous_execution)
{
  rtc_interface_left_ = std::make_shared<RTCInterface>(node, name + "_left");
  rtc_interface_right_ = std::make_shared<RTCInterface>(node, name + "_right");
  getModuleParams(node);
}

void AvoidanceModuleManager::updateModuleParams(const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  auto p = module_params_;

  std::string ns = "avoidance.";
  updateParam<bool>(parameters, ns + "publish_debug_marker", p->publish_debug_marker);
  updateParam<bool>(parameters, ns + "print_debug_info", p->print_debug_info);

  std::for_each(registered_modules_.begin(), registered_modules_.end(), [&p](const auto & m) {
    m.second->updateModuleParams(p);
  });
}

void AvoidanceModuleManager::getModuleParams(rclcpp::Node * node)
{
  AvoidanceParameters p{};

  // general params
  {
    std::string ns = "avoidance.";
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
  }

  // constraints
  {
    std::string ns = "avoidance.constraints.";
    p.hard_constraints = node->declare_parameter<bool>(ns + "hard_constraints");
  }

  // constraints (longitudinal)
  {
    std::string ns = "avoidance.constraints.longitudinal.";
    p.nominal_deceleration = node->declare_parameter<double>(ns + "nominal_deceleration");
    p.nominal_jerk = node->declare_parameter<double>(ns + "nominal_jerk");
    p.max_deceleration = node->declare_parameter<double>(ns + "max_deceleration");
    p.max_jerk = node->declare_parameter<double>(ns + "max_jerk");
    p.min_avoidance_speed_for_acc_prevention =
      node->declare_parameter<double>(ns + "min_avoidance_speed_for_acc_prevention");
    p.max_avoidance_acceleration =
      node->declare_parameter<double>(ns + "max_avoidance_acceleration");
  }

  // constraints (lateral)
  {
    std::string ns = "avoidance.constraints.lateral.";
    p.nominal_lateral_jerk = node->declare_parameter<double>(ns + "nominal_lateral_jerk");
    p.max_lateral_jerk = node->declare_parameter<double>(ns + "max_lateral_jerk");
  }

  // target filtering
  {
    std::string ns = "avoidance.target_filtiering.";
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

  // path generation (longitudinal)
  {
    std::string ns = "avoidance.path_generation.longitudinal.";
    p.longitudinal_collision_safety_buffer_front =
      node->declare_parameter<double>(ns + "longitudinal_collision_safety_buffer_front");
    p.longitudinal_collision_safety_buffer_back =
      node->declare_parameter<double>(ns + "longitudinal_collision_safety_buffer_back");
    p.prepare_time = node->declare_parameter<double>(ns + "prepare_time");
    p.min_prepare_distance = node->declare_parameter<double>(ns + "min_prepare_distance");
    p.min_avoidance_distance = node->declare_parameter<double>(ns + "min_avoidance_distance");
    p.avoidance_speed_step = node->declare_parameter<double>(ns + "avoidance_speed_step");
    p.min_nominal_avoidance_speed =
      node->declare_parameter<double>(ns + "min_nominal_avoidance_speed");
    p.min_sharp_avoidance_speed = node->declare_parameter<double>(ns + "min_sharp_avoidance_speed");
    p.longitudinal_collision_margin_min_distance =
      node->declare_parameter<double>(ns + "longitudinal_collision_margin_min_distance");
    p.longitudinal_collision_margin_time =
      node->declare_parameter<double>(ns + "longitudinal_collision_margin_time");
  }

  // path generation (lateral)
  {
    std::string ns = "avoidance.path_generation.lateral.";
    p.lateral_collision_safety_buffer =
      node->declare_parameter<double>(ns + "lateral_collision_safety_buffer");
    p.lateral_passable_safety_buffer =
      node->declare_parameter<double>(ns + "lateral_passable_safety_buffer");
    p.road_shoulder_safety_margin =
      node->declare_parameter<double>(ns + "road_shoulder_safety_margin");
    p.max_right_shift_length = node->declare_parameter<double>(ns + "max_right_shift_length");
    p.max_left_shift_length = node->declare_parameter<double>(ns + "max_left_shift_length");
    p.avoidance_execution_lateral_threshold =
      node->declare_parameter<double>(ns + "avoidance_execution_lateral_threshold");
  }

  // stop
  {
    std::string ns = "avoidance.stop.";
    p.stop_min_distance = node->declare_parameter<double>(ns + "min_distance");
    p.stop_max_distance = node->declare_parameter<double>(ns + "max_distance");
  }

  // yield
  {
    std::string ns = "avoidance.yield.";
    p.yield_velocity = node->declare_parameter<double>(ns + "yield_velocity");
    p.min_longitudinal_margin_for_moving_object =
      node->declare_parameter<double>(ns + "min_longitudinal_margin_for_moving_object");
    p.safety_check_hysteresis_factor =
      node->declare_parameter<double>(ns + "safety_check_hysteresis_factor");
  }

  // target object
  {
    std::string ns = "avoidance.target_object.";
    p.avoid_car = node->declare_parameter<bool>(ns + "car");
    p.avoid_truck = node->declare_parameter<bool>(ns + "truck");
    p.avoid_bus = node->declare_parameter<bool>(ns + "bus");
    p.avoid_trailer = node->declare_parameter<bool>(ns + "trailer");
    p.avoid_unknown = node->declare_parameter<bool>(ns + "unknown");
    p.avoid_bicycle = node->declare_parameter<bool>(ns + "bicycle");
    p.avoid_motorcycle = node->declare_parameter<bool>(ns + "motorcycle");
    p.avoid_pedestrian = node->declare_parameter<bool>(ns + "pedestrian");
  }

  // velocity matrix
  {
    std::string ns = "avoidance.target_velocity_matrix.";
    p.col_size = node->declare_parameter<int>(ns + "col_size");
    p.target_velocity_matrix = node->declare_parameter<std::vector<double>>(ns + "matrix");
  }

  module_params_ = std::make_shared<AvoidanceParameters>(p);
}
}  // namespace behavior_path_planner
