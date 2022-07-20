
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

#ifndef BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_PARAMETERS_HPP_
#define BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_PARAMETERS_HPP_

namespace behavior_path_planner
{
struct PullOutParameters
{
  double min_stop_distance;
  double stop_time;
  double hysteresis_buffer_distance;
  double pull_out_prepare_duration;
  double pull_out_duration;
  double pull_out_finish_judge_buffer;
  double shift_pull_out_velocity;
  double prediction_duration;
  double prediction_time_resolution;
  double static_obstacle_velocity_thresh;
  double maximum_deceleration;
  int pull_out_sampling_num;
  bool enable_collision_check_at_prepare_phase;
  bool use_predicted_path_outside_lanelet;
  bool use_all_predicted_path;
  bool use_dynamic_object;
  bool enable_blocked_by_obstacle;
  double pull_out_search_distance;
  double before_pull_out_straight_distance;
  double after_pull_out_straight_distance;
  double maximum_lateral_jerk;
  double minimum_lateral_jerk;
  double deceleration_interval;
};

}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_PARAMETERS_HPP_
