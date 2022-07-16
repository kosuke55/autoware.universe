// Copyright 2021 Tier IV, Inc.
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

#include "behavior_path_planner/scene_module/pull_out/util.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/utils/path_shifter.hpp"
#include "behavior_path_planner/util/create_vehicle_footprint.hpp"

#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/geometry/boost_geometry.hpp>

#include <boost/geometry/algorithms/dispatch/distance.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <tf2/utils.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using tier4_autoware_utils::calcOffsetPose;

namespace behavior_path_planner
{
namespace pull_out_utils
{
PathWithLaneId combineReferencePath(const PathWithLaneId path1, const PathWithLaneId path2)
{
  PathWithLaneId path;
  path.points.insert(path.points.end(), path1.points.begin(), path1.points.end());

  // skip overlapping point
  path.points.insert(path.points.end(), next(path2.points.begin()), path2.points.end());

  return path;
}

PathWithLaneId getBackwardPath(
  const RouteHandler & route_handler, const lanelet::ConstLanelets & shoulder_lanelets,
  const Pose & current_pose, const Pose & backed_pose)
{
  const double backward_velocity = -1.0;

  const auto current_pose_arc_coords =
    lanelet::utils::getArcCoordinates(shoulder_lanelets, current_pose);
  const auto backed_pose_arc_coords =
    lanelet::utils::getArcCoordinates(shoulder_lanelets, backed_pose);

  const double s_start = backed_pose_arc_coords.length;
  const double s_end = current_pose_arc_coords.length;

  PathWithLaneId backward_path;
  {
    // foward center line path
    backward_path = route_handler.getCenterLinePath(shoulder_lanelets, s_start, s_end, true);

    // backward center line path
    std::reverse(backward_path.points.begin(), backward_path.points.end());
    for (auto & p : backward_path.points) {
      p.point.longitudinal_velocity_mps = backward_velocity;
    }
    backward_path.points.back().point.longitudinal_velocity_mps = 0.0;

    // lateral shift to current_pose
    const double lateral_distance_to_shoulder_center = current_pose_arc_coords.distance;
    for (size_t i = 0; i < backward_path.points.size(); ++i) {
      auto & p = backward_path.points.at(i).point.pose;
      p = calcOffsetPose(p, 0, lateral_distance_to_shoulder_center, 0);
    }
  }

  return backward_path;
}

Pose getBackedPose(
  const Pose & current_pose, const double & yaw_shoulder_lane, const double & back_distance)
{
  Pose backed_pose;
  backed_pose = current_pose;
  backed_pose.position.x -= std::cos(yaw_shoulder_lane) * back_distance;
  backed_pose.position.y -= std::sin(yaw_shoulder_lane) * back_distance;

  return backed_pose;
}

// getShoulderLanesOnCurrentPose?
lanelet::ConstLanelets getPullOutLanes(
  const lanelet::ConstLanelets & current_lanes,
  const std::shared_ptr<const PlannerData> & planner_data)
{
  const double pull_out_lane_length = 200.0;
  lanelet::ConstLanelets shoulder_lanes;
  const auto & route_handler = planner_data->route_handler;
  const auto current_pose = planner_data->self_pose->pose;
  lanelet::ConstLanelet shoulder_lane;

  if (current_lanes.empty()) {
    return shoulder_lanes;
  }

  // Get pull out lanes
  lanelet::ConstLanelet current_lane;
  lanelet::utils::query::getClosestLanelet(
    current_lanes, planner_data->self_pose->pose, &current_lane);

  if (route_handler->getPullOutStart(
        route_handler->getShoulderLanelets(), &shoulder_lane, current_pose,
        planner_data->parameters.vehicle_width)) {
    shoulder_lanes = route_handler->getShoulderLaneletSequence(
      shoulder_lane, current_pose, pull_out_lane_length, pull_out_lane_length);

  } else {
    shoulder_lanes.clear();
  }

  return shoulder_lanes;
}

}  // namespace pull_out_utils
}  // namespace behavior_path_planner
