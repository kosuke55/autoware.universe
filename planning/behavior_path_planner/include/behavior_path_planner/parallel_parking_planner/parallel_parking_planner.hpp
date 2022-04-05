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

#ifndef BEHAVIOR_PATH_PLANNER__PARALELL_PARKING_PLANNER_HPP_
#define BEHAVIOR_PATH_PLANNER__PARALELL_PARKING_PLANNER_HPP_

#include "behavior_path_planner/data_manager.hpp"
#include "behavior_path_planner/parameters.hpp"

#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/ros/marker_helper.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace behavior_path_planner
{
using autoware_auto_planning_msgs::msg::PathPointWithLaneId;
using autoware_auto_planning_msgs::msg::PathWithLaneId;
using geometry_msgs::msg::Point;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseStamped;
using autoware_auto_planning_msgs::msg::PathWithLaneId;

class ParallelParkingPlanner
{
public:
  // ParallelParkingPlanner();
  PathWithLaneId generate();
  void setParams(const std::shared_ptr<const PlannerData> & planner_data);

  // debug
  PoseStamped Cr_;
  PoseStamped Cl_;
  PoseArray path_pose_array_;

private:
  // std::shared_ptr<PlannerData> planner_data_;
  std::shared_ptr<const PlannerData> planner_data_;
  float max_steer_deg_ = 40.0;  // max steering angle [deg]
  float max_steer_rad_;

  // struct GeometricParams
  // {
  //   float R_base_link_min;
  //   float R_frontleft_l_min;
  // } geometric_params_;
    float R_E_min_; // base_link
    float R_Bl_min_; // front_lef
    std::vector<PathWithLaneId> paths_;

    PathWithLaneId planOneTraial();
    PathWithLaneId generateArcPath(
      const Pose & center, const float radius, const float start_rad, float end_rad,
      const bool is_left_turn);
    PathPointWithLaneId generateArcPathPoint(
      const Pose & center, const float radius, const float yaw, const bool is_left_turn);
};

}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__PARALLEL_PARKING_HPP_
