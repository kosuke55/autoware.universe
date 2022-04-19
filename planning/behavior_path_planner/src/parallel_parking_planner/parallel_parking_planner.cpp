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

#include "behavior_path_planner/parallel_parking_planner/parallel_parking_planner.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/utilities.hpp"
#include "tier4_autoware_utils/geometry/geometry.hpp"

#include <interpolation/spline_interpolation.hpp>
#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2_eigen/tf2_eigen.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

using autoware_auto_planning_msgs::msg::PathWithLaneId;
using behavior_path_planner::util::concatePath;
using behavior_path_planner::util::convertToGeometryPoseArray;
using behavior_path_planner::util::removeOverlappingPoints;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::Transform;
using geometry_msgs::msg::TransformStamped;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcOffsetPose;
using tier4_autoware_utils::calcSignedArcLength;
using tier4_autoware_utils::deg2rad;
using tier4_autoware_utils::fromMsg;
using tier4_autoware_utils::inverseTransformPoint;
using tier4_autoware_utils::normalizeRadian;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Point3d;
using tier4_autoware_utils::pose2transform;
using tier4_autoware_utils::toMsg;
using tier4_autoware_utils::transformPose;

namespace behavior_path_planner
{
PathWithLaneId ParallelParkingPlanner::getCurrentPath()
{
  const auto current_path = paths_.at(current_path_idx_);
  const auto current_target = current_path.points.back();
  const auto self_pose = planner_data_->self_pose->pose;

  const float th_arrived_distance_m = 0.5;
  const bool is_near_target =
    tier4_autoware_utils::calcDistance2d(current_target, self_pose) < th_arrived_distance_m;

  const float th_stopped_velocity_mps = 0.1;
  const bool is_stopped =
    std::abs(planner_data_->self_odometry->twist.twist.linear.x) < th_stopped_velocity_mps;

  if (is_near_target && is_stopped) {
    current_path_idx_ += 1;
    // rclcpp::Rate(1.0).sleep();
  }
  std::cerr << "before_min current_path_idx_: " << current_path_idx_ << std::endl;
  current_path_idx_ = std::min(current_path_idx_, paths_.size() - 1);
  std::cerr << "current_path_idx_: " << current_path_idx_ << std::endl;

  return paths_.at(current_path_idx_);
}

PathWithLaneId ParallelParkingPlanner::getFullPath()
{
 PathWithLaneId path = paths_.front();
 for (size_t i = 1; i < paths_.size() - 1; i++) {  // todo remove -1
   path = concatePath(path, paths_.at(i));
 }
 return path;
}

void ParallelParkingPlanner::clear()
{
  std::cerr << "clear : " << current_path_idx_ << std::endl;
  current_path_idx_ = 0;
  paths_.clear();
}

bool ParallelParkingPlanner::isParking() const { return current_path_idx_ > 0; }

void ParallelParkingPlanner::plan(const Pose goal_pose, const double start_pose_offset)
{
  // plan path only when parking has not started
  if (!isParking()) {
    paths_.clear();
    getStraightPath(goal_pose, start_pose_offset);
    planOneTraial(goal_pose, start_pose_offset);
  }
}

Pose ParallelParkingPlanner::getStartPose(const Pose goal_pose, const double start_pose_offset)
{
  auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto arc_coordinates = lanelet::utils::getArcCoordinates(current_lanes, goal_pose);

  const float dx =
    2 * std::sqrt(std::pow(R_E_min_, 2) - std::pow(-arc_coordinates.distance / 2 + R_E_min_, 2));
  Pose start_pose = calcOffsetPose(goal_pose, dx + start_pose_offset, -arc_coordinates.distance, 0);

  return start_pose;
}

void ParallelParkingPlanner::getStraightPath(const Pose goal_pose, const double start_pose_offset)
{
  // get stright path before parking.
  auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto next_lanes = planner_data_->route_handler->getNextLanelets(current_lanes.back());

  current_lanes.push_back(next_lanes.front());

  const Pose start_pose = getStartPose(goal_pose, start_pose_offset);
  const auto start_arc_position = lanelet::utils::getArcCoordinates(current_lanes, start_pose);

  const Pose current_pose = planner_data_->self_pose->pose;
  const auto crrent_arc_position = lanelet::utils::getArcCoordinates(current_lanes, current_pose);

  auto path = planner_data_->route_handler->getCenterLinePath(
    current_lanes, crrent_arc_position.length, start_arc_position.length);
  path.header = planner_data_->route_handler->getRouteHeader();

  const auto common_params = planner_data_->parameters;
  path.drivable_area = util::generateDrivableArea(
    current_lanes, common_params.drivable_area_resolution, common_params.vehicle_length,
    planner_data_);

  path.points.back().point.longitudinal_velocity_mps = 0;

  paths_.push_back(path);
}

void ParallelParkingPlanner::planOneTraial(const Pose goal_pose, const double start_pose_offset)
{
  // debug
  start_pose_.pose = getStartPose(goal_pose, start_pose_offset);
  start_pose_.header = planner_data_->route_handler->getRouteHeader();

  PathWithLaneId path;
  const auto start_pose = getStartPose(goal_pose, start_pose_offset);

  const float self_yaw = tf2::getYaw(start_pose.orientation);
  const float goal_yaw = tf2::getYaw(goal_pose.orientation);
  const float psi = normalizeRadian(self_yaw - goal_yaw);
  const auto common_params = planner_data_->parameters;

  Pose Cr = calcOffsetPose(goal_pose, 0, -R_E_min_, 0);
  Cr_.pose = Cr;
  Cr_.header = planner_data_->route_handler->getRouteHeader();
  const float d_Cr_Einit = calcDistance2d(Cr, start_pose);

  geometry_msgs::msg::Point Cr_goalcoords = inverseTransformPoint(Cr.position, goal_pose);
  geometry_msgs::msg::Point self_point_goalcoords =
    inverseTransformPoint(start_pose.position, goal_pose);

  const float alpha =
    M_PI_2 - psi + std::asin((self_point_goalcoords.y - Cr_goalcoords.y) / d_Cr_Einit);

  const float R_Einit_l = (std::pow(d_Cr_Einit, 2) - std::pow(R_E_min_, 2)) /
                          (2 * (R_E_min_ + d_Cr_Einit * std::cos(alpha)));
  if (R_Einit_l <= 0) {
    return;
  }

  const float steer_l = std::atan(common_params.wheel_base / R_Einit_l);

  Pose Cl = calcOffsetPose(start_pose, 0, R_Einit_l, 0);
  Cl_.pose = Cl;
  Cl_.header = planner_data_->route_handler->getRouteHeader();

  const float theta_l = std::acos(
    (std::pow(R_Einit_l, 2) + std::pow(R_Einit_l + R_E_min_, 2) - std::pow(d_Cr_Einit, 2)) /
    (2 * R_Einit_l * (R_Einit_l + R_E_min_)));

  lanelet::ConstLanelet current_lane;
  planner_data_->route_handler->getClosestLaneletWithinRoute(start_pose, &current_lane);
  lanelet::ConstLanelet goal_lane;
  planner_data_->route_handler->getClosestLaneletWithinRoute(goal_pose, &goal_lane);

  PathWithLaneId path_turn_left =
    generateArcPath(Cl, R_Einit_l, -M_PI_2, normalizeRadian(-M_PI_2 - theta_l), false);
  path_turn_left.header = planner_data_->route_handler->getRouteHeader();
  // Generate drivable area
  {
    lanelet::ConstLanelets lanes;
    lanes.push_back(current_lane);
    lanes.push_back(goal_lane);
    path_turn_left.drivable_area = util::generateDrivableArea(
      lanes, common_params.drivable_area_resolution, common_params.vehicle_length, planner_data_);
  }
  paths_.push_back(path_turn_left);

  PathWithLaneId path_turn_right =
    generateArcPath(Cr, R_E_min_, normalizeRadian(psi + M_PI_2 - theta_l), M_PI_2, true);
  path_turn_right.header = planner_data_->route_handler->getRouteHeader();
  // Generate drivable area
  {
    lanelet::ConstLanelets lanes;
    lanes.push_back(current_lane);
    lanes.push_back(goal_lane);
    path_turn_right.drivable_area = util::generateDrivableArea(
      lanes, common_params.drivable_area_resolution, common_params.vehicle_length, planner_data_);
  }
  paths_.push_back(path_turn_right);

  PathWithLaneId concat_path = concatePath(path_turn_right, path_turn_left);

  path.header = planner_data_->route_handler->getRouteHeader();
  path_pose_array_ = convertToGeometryPoseArray(concat_path);
}

PathWithLaneId ParallelParkingPlanner::generateArcPath(
  const Pose & center, const float radius, const float start_yaw, float end_yaw,
  const bool is_left_turn)
{
  PathWithLaneId path;

  const float interval = 0.5;
  const float yaw_interval = interval / radius;
  float yaw = start_yaw;

  if (is_left_turn) {
    if (end_yaw < start_yaw) end_yaw += M_PI_2;
    while (yaw < end_yaw) {
      PathPointWithLaneId p = generateArcPathPoint(center, radius, yaw, is_left_turn);
      p.point.longitudinal_velocity_mps = -0.5;
      path.points.push_back(p);
      yaw += yaw_interval;
    }
  } else {  // right_turn
    if (end_yaw > start_yaw) end_yaw -= M_PI_2;
    while (yaw > end_yaw) {
      PathPointWithLaneId p = generateArcPathPoint(center, radius, yaw, is_left_turn);
      p.point.longitudinal_velocity_mps = -0.5;
      path.points.push_back(p);
      yaw -= yaw_interval;
    }
  }
  PathPointWithLaneId p = generateArcPathPoint(center, radius, end_yaw, is_left_turn);
  p.point.longitudinal_velocity_mps = 0.0;
  path.points.push_back(p);

  return path;
}

PathPointWithLaneId ParallelParkingPlanner::generateArcPathPoint(
  const Pose & center, const float radius, const float yaw, const bool is_left_turn)
{
  Pose pose_centercoords;
  pose_centercoords.position.x = radius * std::cos(yaw);
  pose_centercoords.position.y = radius * std::sin(yaw);
  // pose_centercoords.position.z = center.position.z;

  tf2::Quaternion quat;
  if (is_left_turn) {
    quat.setRPY(0, 0, normalizeRadian(yaw - M_PI_2));
  } else {
    quat.setRPY(0, 0, normalizeRadian(yaw + M_PI_2));
  }
  pose_centercoords.orientation = tf2::toMsg(quat);

  PathPointWithLaneId p{};
  p.point.pose = transformPose(pose_centercoords, center);
  lanelet::ConstLanelet current_lane;
  planner_data_->route_handler->getClosestLaneletWithinRoute(p.point.pose, &current_lane);

  // Use z of lanelet closest point
  double min_distance = std::numeric_limits<double>::max();
  for (const auto pt : current_lane.centerline3d()) {
    const double distance =
      calcDistance2d(p.point.pose, lanelet::utils::conversion::toGeomMsgPt(pt));
    if (distance < min_distance) {
      min_distance = distance;
      p.point.pose.position.z = pt.z();
    }
  }

  p.lane_ids.push_back(current_lane.id());

  return p;
}

void ParallelParkingPlanner::setParams(const std::shared_ptr<const PlannerData> & planner_data)
{
  // planner_data_ = std::make_shared<PlannerData>();
  planner_data_ = planner_data;
  auto common_params = planner_data_->parameters;
  max_steer_rad_ = deg2rad(max_steer_deg_);

  R_E_min_ = common_params.wheel_base / std::tan(max_steer_rad_);
  R_Bl_min_ = std::hypot(
    R_E_min_ + common_params.wheel_tread / 2 + common_params.left_over_hang,
    common_params.wheel_base + common_params.front_overhang);
}
}  // namespace behavior_path_planner
