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

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/pull_out/util.hpp"

using lanelet::utils::getArcCoordinates;
using motion_utils::findNearestIndex;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcOffsetPose;
namespace behavior_path_planner
{
using pull_out_utils::combineReferencePath;
using pull_out_utils::getPullOutLanes;

ShiftPullOut::ShiftPullOut(
  rclcpp::Node & node, const PullOutParameters & parameters,
  std::shared_ptr<LaneDepartureChecker> & lane_departure_checker)
: PullOutPlannerBase{node, parameters}, lane_departure_checker_{lane_departure_checker}
{
}

boost::optional<PullOutPath> ShiftPullOut::plan(Pose start_pose, Pose goal_pose)
{
  PullOutPath safe_path;
  std::vector<PullOutPath> valid_paths;
  const auto & route_handler = planner_data_->route_handler;
  const auto common_parameters = planner_data_->parameters;
  const auto dynamic_objects = planner_data_->dynamic_object;
  const auto road_lanes = util::getCurrentLanes(planner_data_);
  const auto shoulder_lanes = getPullOutLanes(road_lanes, planner_data_);

  lanelet::ConstLanelets lanes;
  lanes.insert(lanes.end(), road_lanes.begin(), road_lanes.end());
  lanes.insert(lanes.end(), shoulder_lanes.begin(), shoulder_lanes.end());

  if (!shoulder_lanes.empty()) {
    // find candidate paths
    const auto pull_out_paths = getPullOutPaths(
      *route_handler, road_lanes, shoulder_lanes, start_pose, goal_pose, common_parameters,
      parameters_);

    // select valid path
    valid_paths = selectValidPaths(
      pull_out_paths, road_lanes, start_pose,
      route_handler->isInGoalRouteSection(road_lanes.back()), goal_pose);
    if (valid_paths.empty()) {
      return boost::none;
    }

    const auto shoulder_lane_objects =
      util::filterObjectsByLanelets(*dynamic_objects, shoulder_lanes);

    // get safe path
    for (auto & pull_out_path : valid_paths) {
      auto & shift_path =
        pull_out_path.partial_paths.front();  // shift path is not separate but only one.
      if (util::checkCollisionBetweenPathFootprintsAndObjects(
            vehicle_footprint_, pull_out_path.partial_paths.front(), shoulder_lane_objects,
            parameters_.collision_check_margin)) {
        continue;
      }
      full_path_ = shift_path;

      // Generate drivable area
      const double resolution = common_parameters.drivable_area_resolution;
      shift_path.drivable_area = util::generateDrivableArea(
        lanes, resolution, common_parameters.vehicle_length, planner_data_);

      shift_path.header = planner_data_->route_handler->getRouteHeader();

      return pull_out_path;
    }
  }
  return boost::none;
}

std::vector<PullOutPath> ShiftPullOut::getPullOutPaths(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & road_lanelets,
  const lanelet::ConstLanelets & shoulder_lanelets, const Pose & start_pose, const Pose & goal_pose,
  const BehaviorPathPlannerParameters & common_parameter, const PullOutParameters & parameter)
{
  std::vector<PullOutPath> candidate_paths;

  if (road_lanelets.empty() || shoulder_lanelets.empty()) {
    return candidate_paths;
  }

  // rename parameter
  const double backward_path_length = common_parameter.backward_path_length;
  const double shift_pull_out_velocity = parameter.shift_pull_out_velocity;
  const double before_pull_out_straight_distance = parameter.before_pull_out_straight_distance;
  const double minimum_lateral_jerk = parameter.minimum_lateral_jerk;
  const double maximum_lateral_jerk = parameter.maximum_lateral_jerk;
  const int pull_out_sampling_num = parameter.pull_out_sampling_num;
  const double jerk_resolution =
    std::abs(maximum_lateral_jerk - minimum_lateral_jerk) / pull_out_sampling_num;

  for (double lateral_jerk = minimum_lateral_jerk; lateral_jerk <= maximum_lateral_jerk;
       lateral_jerk += jerk_resolution) {
    PathShifter path_shifter;
    ShiftedPath shifted_path;
    const double distance_to_road_center = getArcCoordinates(road_lanelets, start_pose).distance;

    PathWithLaneId shoulder_reference_path;
    {
      const auto arc_position = getArcCoordinates(shoulder_lanelets, start_pose);
      const double s_start = arc_position.length - backward_path_length;
      double s_end = arc_position.length + before_pull_out_straight_distance;
      s_end = std::max(s_end, s_start + std::numeric_limits<double>::epsilon());
      shoulder_reference_path = route_handler.getCenterLinePath(shoulder_lanelets, s_start, s_end);

      // lateral shift to start_pose
      for (auto & p : shoulder_reference_path.points) {
        p.point.pose = calcOffsetPose(p.point.pose, 0, arc_position.distance, 0);
      }
    }

    PathWithLaneId road_lane_reference_path;
    {
      const lanelet::ArcCoordinates arc_position_shift =
        getArcCoordinates(road_lanelets, shoulder_reference_path.points.back().point.pose);
      const lanelet::ArcCoordinates arc_position_goal = getArcCoordinates(road_lanelets, goal_pose);

      double s_start = arc_position_shift.length;
      double s_end = arc_position_goal.length;
      road_lane_reference_path = route_handler.getCenterLinePath(road_lanelets, s_start, s_end);
    }

    const double pull_out_distance = path_shifter.calcLongitudinalDistFromJerk(
      abs(distance_to_road_center), lateral_jerk, shift_pull_out_velocity);

    // get shift point start/end
    const auto shift_point_start = shoulder_reference_path.points.back();
    const auto shift_point_end = [&]() {
      const auto arc_position_shift_start =
        lanelet::utils::getArcCoordinates(road_lanelets, shift_point_start.point.pose);
      const double s_start = arc_position_shift_start.length + pull_out_distance;
      const double s_end = s_start + std::numeric_limits<double>::epsilon();
      const auto path = route_handler.getCenterLinePath(shoulder_lanelets, s_start, s_end, true);
      return path.points.front();
    }();

    ShiftPoint shift_point;
    {
      shift_point.start = shift_point_start.point.pose;
      shift_point.end = shift_point_end.point.pose;
      shift_point.length = distance_to_road_center;
    }
    path_shifter.addShiftPoint(shift_point);
    path_shifter.setPath(road_lane_reference_path);

    // offset front side
    const bool offset_back = false;
    if (!path_shifter.generate(&shifted_path, offset_back)) {
      continue;
    }

    const auto pull_out_end_idx =
      findNearestIndex(shifted_path.path.points, shift_point_end.point.pose);
    const auto goal_idx = findNearestIndex(shifted_path.path.points, goal_pose);

    PullOutPath candidate_path;
    candidate_path.preparation_length = before_pull_out_straight_distance;
    candidate_path.pull_out_length = pull_out_distance;

    if (pull_out_end_idx && goal_idx) {
      // set velocity
      for (size_t i = 0; i < shifted_path.path.points.size(); ++i) {
        auto & point = shifted_path.path.points.at(i);
        if (i < *pull_out_end_idx) {
          point.point.longitudinal_velocity_mps = std::min(
            point.point.longitudinal_velocity_mps, static_cast<float>(shift_pull_out_velocity));
          continue;
        } else if (i > *goal_idx) {
          point.point.longitudinal_velocity_mps = 0.0;
          continue;
        }
      }

      const auto combined_path = combineReferencePath(shoulder_reference_path, shifted_path.path);
      candidate_path.partial_paths.push_back(util::resamplePathWithSpline(combined_path, 1.0));
      candidate_path.shifted_path = shifted_path;
      candidate_path.shift_point = shift_point;
      candidate_path.start_pose = shift_point.start;
      candidate_path.end_pose = shift_point.end;

      // add goal pose because resampling removes it
      // but this point will be removed by SmoothGoalConnection again
      PathPointWithLaneId goal_path_point = shifted_path.path.points.back();
      goal_path_point.point.pose = goal_pose;
      goal_path_point.point.longitudinal_velocity_mps = 0.0;
      candidate_path.partial_paths.front().points.push_back(goal_path_point);
    } else {
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("behavior_path_planner").get_child("pull_out").get_child("util"),
        "lane change end idx not found on target path.");
      continue;
    }

    // check lane departure to shift end point
    PathWithLaneId path_to_shift_end;
    {
      const auto & path_to_goal = candidate_path.partial_paths.front();
      path_to_shift_end.points.insert(
        path_to_shift_end.points.begin(), path_to_goal.points.begin(),
        path_to_goal.points.begin() + *pull_out_end_idx + 1);
    }
    auto lanes = road_lanelets;
    lanes.insert(lanes.end(), shoulder_lanelets.begin(), shoulder_lanelets.end());
    if (lane_departure_checker_->checkPathWillLeaveLane(lanes, path_to_shift_end)) {
      continue;
    }

    candidate_paths.push_back(candidate_path);
  }

  return candidate_paths;
}

std::vector<PullOutPath> ShiftPullOut::selectValidPaths(
  const std::vector<PullOutPath> & paths, const lanelet::ConstLanelets & road_lanes,
  const Pose & current_pose, const bool isInGoalRouteSection, const Pose & goal_pose)
{
  std::vector<PullOutPath> available_paths;

  for (const auto & path : paths) {
    if (hasEnoughDistance(path, road_lanes, current_pose, isInGoalRouteSection, goal_pose)) {
      available_paths.push_back(path);
    }
  }

  return available_paths;
}

bool ShiftPullOut::hasEnoughDistance(
  const PullOutPath & path, const lanelet::ConstLanelets & road_lanes, const Pose & current_pose,
  const bool isInGoalRouteSection, const Pose & goal_pose)
{
  const double pull_out_prepare_distance = path.preparation_length;
  const double pull_out_distance = path.pull_out_length;
  const double pull_out_total_distance = pull_out_prepare_distance + pull_out_distance;

  if (pull_out_total_distance > util::getDistanceToEndOfLane(current_pose, road_lanes)) {
    return false;
  }

  if (
    isInGoalRouteSection &&
    pull_out_total_distance > util::getSignedDistance(current_pose, goal_pose, road_lanes)) {
    return false;
  }

  return true;
}

}  // namespace behavior_path_planner
