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

#include "behavior_path_planner/scene_module/pull_out/shift_pull_out.hpp"

#include "behavior_path_planner/scene_module/pull_out/util.hpp"

namespace behavior_path_planner
{
ShiftPullOut::ShiftPullOut(rclcpp::Node & node, const PullOutParameters & parameters)
: PullOutBase(node, parameters)
{
}

boost::optional<PathWithLaneId> ShiftPullOut::plan(Pose start_pose, Pose goal_pose)
{
  PullOutPath safe_path;
  std::vector<PullOutPath> valid_paths;
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto current_twist = planner_data_->self_odometry->twist.twist;
  const auto common_parameters = planner_data_->parameters;
  const auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto pull_out_lanes = pull_out_utils::getPullOutLanes(current_lanes, planner_data_);
  const double check_distance = 100.0;

  if (!pull_out_lanes.empty()) {
    // find candidate paths
    const auto pull_out_paths = pull_out_utils::getPullOutPaths(
      *route_handler, current_lanes, pull_out_lanes, start_pose, common_parameters, parameters_);

    // get lanes used for detection
    lanelet::ConstLanelets check_lanes;
    if (!pull_out_paths.empty()) {
      const auto & longest_path = pull_out_paths.front();
      // we want to see check_distance [m] behind vehicle so add lane changing length
      const double check_distance_with_path =
        check_distance + longest_path.preparation_length + longest_path.pull_out_length;
      check_lanes = route_handler->getCheckTargetLanesFromPath(
        longest_path.path, pull_out_lanes, check_distance_with_path);
    }
    // select valid path
    valid_paths = pull_out_utils::selectValidPaths(
      pull_out_paths, current_lanes, check_lanes, *route_handler->getOverallGraphPtr(),
      start_pose, route_handler->isInGoalRouteSection(current_lanes.back()),
      route_handler->getGoalPose());
    if (valid_paths.empty()) {
      return boost::none;
    }
    // select safe path
    bool found_safe_path = pull_out_utils::selectSafePath(
      valid_paths, current_lanes, check_lanes, planner_data_->dynamic_object, start_pose,
      current_twist, common_parameters.vehicle_width, parameters_, vehicle_footprint_, &safe_path);

    if (found_safe_path) return safe_path.path;
  }
  return boost::none;
}

}  // namespace behavior_path_planner
