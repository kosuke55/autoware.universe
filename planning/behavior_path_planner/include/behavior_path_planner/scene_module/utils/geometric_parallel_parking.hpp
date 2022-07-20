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

#ifndef BEHAVIOR_PATH_PLANNER__SCENE_MODULE__UTILS__GEOMETRIC_PARALLEL_PARKING_HPP_
#define BEHAVIOR_PATH_PLANNER__SCENE_MODULE__UTILS__GEOMETRIC_PARALLEL_PARKING_HPP_

#include "behavior_path_planner/data_manager.hpp"
#include "behavior_path_planner/parameters.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/ros/marker_helper.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace behavior_path_planner
{
using autoware_auto_planning_msgs::msg::PathPointWithLaneId;
using autoware_auto_planning_msgs::msg::PathWithLaneId;
using geometry_msgs::msg::Point;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::PoseStamped;

struct ParallelParkingParameters
{
  double th_arrived_distance_m;
  double th_stopped_velocity_mps;
  double after_forward_parking_straight_distance;
  double after_backward_parking_straight_distance;
  double forward_parking_velocity;
  double backward_parking_velocity;
  double arc_path_interval;
  double min_acc;
};

class GeometricParallelParking
{
public:
  bool isParking() const;
  bool plan(const Pose & goal_pose, const lanelet::ConstLanelets & lanes, const bool is_forward);
  bool planDeparting(const Pose & goal_pose, const lanelet::ConstLanelets & lanes);
  void setData(
    const std::shared_ptr<const PlannerData> & planner_data,
    const ParallelParkingParameters & parameters);
  void incrementPathIndex();
  void clear();

  std::vector<PathWithLaneId> getPaths() const { return paths_; }
  PathWithLaneId getPathByIdx(size_t const idx) const;
  PathWithLaneId getCurrentPath() const;
  PathWithLaneId getFullPath() const;
  PathWithLaneId getArcPath() const;
  PoseStamped getCr() const { return Cr_; }
  PoseStamped getCl() const { return Cl_; }
  PoseStamped getStartPose() const { return start_pose_; }
  PoseStamped getArcEndPose() const { return arc_end_pose_; }
  PoseArray getPathPoseArray() const { return path_pose_array_; }

private:
  std::shared_ptr<const PlannerData> planner_data_;
  ParallelParkingParameters parameters_;

  // todo: use vehicle info after merging
  // https://github.com/autowarefoundation/autoware.universe/pull/740
  double max_steer_deg_ = 40.0;  // max steering angle [deg].
  double max_steer_rad_;
  double R_E_min_;   // base_link
  double R_Bl_min_;  // front_left

  std::vector<PathWithLaneId> paths_;
  size_t current_path_idx_ = 0;

  bool isEnoughDistanceToStart(const Pose & start_pose) const;
  std::vector<PathWithLaneId> planOneTraial(
    const Pose & start_pose, const Pose & goal_pose, const double R_E_r,
    const lanelet::ConstLanelets & lanes, const bool is_forward, const double end_pose_offset);
  PathWithLaneId generateArcPath(
    const Pose & center, const double radius, const double start_yaw, double end_yaw,
    const bool is_left_turn, const bool is_forward);
  PathPointWithLaneId generateArcPathPoint(
    const Pose & center, const double radius, const double yaw, const bool is_left_turn,
    const bool is_forward);
  Pose calcStartPose(
    const Pose & goal_pose, const double start_pose_offset, const double R_E_r,
    const bool is_forward);
  std::vector<PathWithLaneId> generateParkingPaths(
    const Pose & start_pose, const Pose & goal_pose, const double R_E_r,
    const lanelet::ConstLanelets lanes, const bool is_forward, const double end_pose_offset,
    const double velocity);
  PathWithLaneId generateStraightPath(const Pose & start_pose);
  void setVelocityToArcPaths(std::vector<PathWithLaneId> & arc_paths, const double velocity);

  PoseStamped Cr_;
  PoseStamped Cl_;
  PoseStamped start_pose_;
  PoseStamped arc_end_pose_;
  PoseArray path_pose_array_;
};
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__UTILS__GEOMETRIC_PARALLEL_PARKING_HPP_
