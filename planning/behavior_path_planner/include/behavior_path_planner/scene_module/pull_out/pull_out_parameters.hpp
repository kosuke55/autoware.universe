
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
  bool enable_shift_pull_out;
  bool enable_geometric_pull_out;
  bool enable_back;
  double th_arrived_distance_m;
  double th_stopped_velocity_mps;
  double th_stopped_time_sec;
  double collision_check_margin;
  double pull_out_finish_judge_buffer;
  // search start pose backward
  double max_back_distance;
  double backward_search_resolution;
  double min_stop_distance;
  // geometric pull out
  double geometric_pull_out_velocity;
  double arc_path_interval;
  // shift pull out
  double shift_pull_out_velocity;
  int pull_out_sampling_num;
  double before_pull_out_straight_distance;
  double after_pull_out_straight_distance;
  double maximum_lateral_jerk;
  double minimum_lateral_jerk;
  double deceleration_interval;
};

}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_PARAMETERS_HPP_
