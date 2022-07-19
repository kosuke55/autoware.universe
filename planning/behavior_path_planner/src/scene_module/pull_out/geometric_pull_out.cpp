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

#include "behavior_path_planner/scene_module/pull_out/geometric_pull_out.hpp"

#include "behavior_path_planner/scene_module/pull_out/util.hpp"

using lanelet::utils::getArcCoordinates;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcOffsetPose;
using motion_utils::findNearestIndex;
namespace behavior_path_planner
{
using pull_out_utils::combineReferencePath;
using pull_out_utils::getPullOutLanes;

GeometricPullOut::GeometricPullOut(
  rclcpp::Node & node, const PullOutParameters & parameters,
  const ParallelParkingParameters & parallel_parking_parameters,
  std::shared_ptr<LaneDepartureChecker> & lane_departure_checker)
: PullOutBase{node, parameters},
  parallel_parking_parameters_{parallel_parking_parameters},
  lane_departure_checker_{lane_departure_checker}
{
}

boost::optional<PullOutPath> GeometricPullOut::plan(Pose start_pose, Pose goal_pose)
{
  PullOutPath output;

  const auto road_lanes = util::getCurrentLanes(planner_data_);
  const auto shoulder_lanes = getPullOutLanes(road_lanes, planner_data_);
  auto lanes = road_lanes;
  lanes.insert(lanes.end(), shoulder_lanes.begin(), shoulder_lanes.end());

  // todo: set params only once?
  planner_.setData(planner_data_, parallel_parking_parameters_);
  planner_.planReverse(start_pose, lanes);

  // collsion check with objects in shoulder lanes
  const double collision_margin = 1.0;  // todo: make param
  const auto full_path = planner_.getFullPath();
  const auto shoulder_lane_objects =
    util::filterObjectsByLanelets(*(planner_data_->dynamic_object), shoulder_lanes);
  if (util::checkCollisionBetweenPathFootprintsAndObjects(
        vehicle_footprint_, full_path, shoulder_lane_objects, collision_margin)) {
    return boost::none;
  }

  full_path_ = full_path;

  // sync paths_
  paths_ = planner_.getPaths();

  output.path = planner_.getCurrentPath();
  output.start_pose = planner_.getPathByIdx(0).points.back().point.pose;
  output.end_pose = planner_.getPathByIdx(1).points.back().point.pose;

  // tmp
  double max_vel = 0.0;
  for(auto & p: output.path.points)
  {
    max_vel = std::max(static_cast<double>(p.point.longitudinal_velocity_mps), max_vel);
  }
  std::cerr << "max_vel: " << max_vel << std::endl;

  return output;
}

void GeometricPullOut::incrementPathIndex()
{
  current_path_idx_ = std::min(current_path_idx_ + 1, paths_.size() - 1);
  // also need the internal index for planner_.getCurrentPath()
  planner_.incrementPathIndex();
}

}  // namespace behavior_path_planner
