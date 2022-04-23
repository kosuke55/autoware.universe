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
ParallelParkingPlanner::ParallelParkingPlanner()
{
  lane_departure_checker_ = std::make_unique<LaneDepartureChecker>();
}

PathWithLaneId ParallelParkingPlanner::getCurrentPath()
{
  const auto current_path = paths_.at(current_path_idx_);
  const auto current_target = current_path.points.back();
  const auto self_pose = planner_data_->self_pose->pose;

  const float th_arrived_distance_m = 0.5;
  const bool is_near_target =
    tier4_autoware_utils::calcDistance2d(current_target, self_pose) < th_arrived_distance_m;

  const float th_stopped_velocity_mps = 0.01;
  const bool is_stopped =
    std::abs(planner_data_->self_odometry->twist.twist.linear.x) < th_stopped_velocity_mps;

  if (is_near_target && is_stopped) {
    current_path_idx_ += 1;
    // rclcpp::Rate(1.0).sleep();
  }
  current_path_idx_ = std::min(current_path_idx_, paths_.size() - 1);
  std::cerr << "current_path_idx_: " << current_path_idx_ << " / " << paths_.size() - 1
            << std::endl;

  return paths_.at(current_path_idx_);
}

PathWithLaneId ParallelParkingPlanner::getFullPath()
{
  PathWithLaneId path{};
  for (const auto & p: paths_){
    path.points.insert(path.points.end(), p.points.begin(), p.points.end());
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

void ParallelParkingPlanner::plan(
  const Pose goal_pose, const lanelet::ConstLanelets lanes, const bool is_forward)
{
  const auto common_params = planner_data_->parameters;
  // plan path only when parking has not started
  if (!isParking()) {
    if (is_forward) {
      // When turning forward to the right, the front left goes out,
      // so reduce the steer angle at that time.
      for (double steer = max_steer_rad_; steer > 0.05; steer -= 0.05) {
        paths_.clear();
        const double R_E_r = common_params.wheel_base / std::tan(steer);
        getStraightPath(goal_pose, 0, R_E_r, is_forward);
        // Find path witout lane departure
        if (planOneTraial(goal_pose, 0, R_E_r, lanes, is_forward)) break; 
      }
    } else {
      // When turning backward to the left, the front right goes out,
      // so make the parking start point in front(same to Equivalent to reducing the steer angle).
      for (double dx = 0; dx < 20; dx += 0.5) {
        paths_.clear();
        getStraightPath(goal_pose, dx, R_E_min_, is_forward);
        // Find path witout lane departure
        if (planOneTraial(goal_pose, dx, R_E_min_, lanes, is_forward)) break;
      }
    }
  }
}

Pose ParallelParkingPlanner::getStartPose(
  const Pose goal_pose, const double start_pose_offset, const double R_E_r, const bool is_forward)
{
  // Not use shoulder lanes.
  // auto current_lanes = util::getExtendedCurrentLanes(planner_data_);
  auto current_lanes = util::getCurrentLanes(planner_data_);
  const auto arc_coordinates = lanelet::utils::getArcCoordinates(current_lanes, goal_pose);

  const float dx =
    2 * std::sqrt(std::pow(R_E_r, 2) - std::pow(-arc_coordinates.distance / 2 + R_E_r, 2));

  Pose start_pose;
  if (is_forward) {
    start_pose = calcOffsetPose(goal_pose, -dx + start_pose_offset, -arc_coordinates.distance, 0);
  } else {
    start_pose = calcOffsetPose(goal_pose, dx + start_pose_offset, -arc_coordinates.distance, 0);
  }

  return start_pose;
}

void ParallelParkingPlanner::getStraightPath(
  const Pose goal_pose, const double start_pose_offset, const double R_E_r,
  const bool is_forward)
{
  // get stright path before parking.

  // Not use shoulder lanes.
  auto current_lanes = util::getCurrentLanes(planner_data_);

  const Pose start_pose = getStartPose(goal_pose, start_pose_offset, R_E_r, is_forward);

  // debug
  start_pose_.pose = getStartPose(goal_pose, start_pose_offset, R_E_r, is_forward);
  start_pose_.header = planner_data_->route_handler->getRouteHeader();

  const auto start_arc_position = lanelet::utils::getArcCoordinates(current_lanes, start_pose);

  const Pose current_pose = planner_data_->self_pose->pose;
  const auto crrent_arc_position = lanelet::utils::getArcCoordinates(current_lanes, current_pose);

  auto path = planner_data_->route_handler->getCenterLinePath(
    current_lanes, crrent_arc_position.length, start_arc_position.length, true);
  path.header = planner_data_->route_handler->getRouteHeader();

  const auto common_params = planner_data_->parameters;
  path.drivable_area = util::generateDrivableArea(
    current_lanes, common_params.drivable_area_resolution, common_params.vehicle_length,
    planner_data_);

  path.points.back().point.longitudinal_velocity_mps = 0;

  paths_.push_back(path);
}


bool ParallelParkingPlanner::planOneTraial(
  const Pose goal_pose, const double start_pose_offset,  const double R_E_r,
  const lanelet::ConstLanelets lanes, const bool is_forward)
{
  // debug
  // start_pose_.pose = getStartPose(goal_pose, start_pose_offset, R_E_r, is_forward);
  // start_pose_.header = planner_data_->route_handler->getRouteHeader();
  path_pose_array_.poses.clear();

  const auto start_pose = getStartPose(goal_pose, start_pose_offset, R_E_r, is_forward);

  const float self_yaw = tf2::getYaw(start_pose.orientation);
  const float goal_yaw = tf2::getYaw(goal_pose.orientation);
  const float psi = normalizeRadian(self_yaw - goal_yaw);
  const auto common_params = planner_data_->parameters;

  Pose Cr = calcOffsetPose(goal_pose, 0, -R_E_r, 0);
  Cr_.pose = Cr;
  Cr_.header = planner_data_->route_handler->getRouteHeader();
  const float d_Cr_Einit = calcDistance2d(Cr, start_pose);

  geometry_msgs::msg::Point Cr_goalcoords = inverseTransformPoint(Cr.position, goal_pose);
  geometry_msgs::msg::Point self_point_goalcoords =
    inverseTransformPoint(start_pose.position, goal_pose);

  const float alpha =
    M_PI_2 - psi + std::asin((self_point_goalcoords.y - Cr_goalcoords.y) / d_Cr_Einit);

  const float R_E_l = (std::pow(d_Cr_Einit, 2) - std::pow(R_E_r, 2)) /
                          (2 * (R_E_r + d_Cr_Einit * std::cos(alpha)));
  if (R_E_l <= 0) {
    return false;
  }

  // If start_pose is prallel to goal_pose, we can know lateral deviation of eges of vehicle,
  // and detect lane departure.
  // Check left bound
  if (is_forward) {
    const float R_front_left =
      std::hypot(R_E_r + common_params.vehicle_width / 2, common_params.base_link2front);
    const double distance_to_left_bound = util::getDistanceToShoulderBoundary(lanes, goal_pose);
    const float left_deviation = R_front_left - R_E_r;
    if (std::abs(distance_to_left_bound) < left_deviation) {
      return false;
    }
  }
  // Check right bound
  else {
    const float R_front_right =
      std::hypot(R_E_l + common_params.vehicle_width / 2, common_params.base_link2front);
    const float right_deviation = R_front_right - R_E_l;
    const double distance_to_right_bound = util::getDistanceToRightBoundary(lanes, start_pose);
    if (distance_to_right_bound < right_deviation) {
      return false;
    }
  }

  const float steer_l = std::atan(common_params.wheel_base / R_E_l);

  Pose Cl = calcOffsetPose(start_pose, 0, R_E_l, 0);
  Cl_.pose = Cl;
  Cl_.header = planner_data_->route_handler->getRouteHeader();

  float theta_l = std::acos(
    (std::pow(R_E_l, 2) + std::pow(R_E_l + R_E_r, 2) - std::pow(d_Cr_Einit, 2)) /
    (2 * R_E_l * (R_E_l + R_E_r)));
  theta_l = is_forward ? theta_l : -theta_l;
  PathWithLaneId path_turn_left =
    generateArcPath(Cl, R_E_l, -M_PI_2, normalizeRadian(-M_PI_2 + theta_l), is_forward, is_forward);
  PathWithLaneId path_turn_right =
    generateArcPath(Cr, R_E_r, normalizeRadian(psi + M_PI_2 + theta_l), M_PI_2, !is_forward, is_forward);

  path_turn_left.header = planner_data_->route_handler->getRouteHeader();
  path_turn_left.drivable_area = util::generateDrivableArea(
    lanes, common_params.drivable_area_resolution, common_params.vehicle_length, planner_data_);
  paths_.push_back(path_turn_left);

  path_turn_right.header = planner_data_->route_handler->getRouteHeader();
  path_turn_right.drivable_area = util::generateDrivableArea(
    lanes, common_params.drivable_area_resolution, common_params.vehicle_length, planner_data_);
  paths_.push_back(path_turn_right);

  PathWithLaneId concat_path = path_turn_right;
  concat_path.points.insert(
    concat_path.points.end(), path_turn_left.points.begin(), path_turn_left.points.end());

  path_pose_array_ = convertToGeometryPoseArray(concat_path);

  return true;
}

PathWithLaneId ParallelParkingPlanner::generateArcPath(
  const Pose & center, const float radius, const float start_yaw, float end_yaw,
  const bool is_left_turn,  // is_left_turn means clockwise around center.
  const bool is_forward)
{
  PathWithLaneId path;

  const float interval = 1.0;
  const float yaw_interval = interval / radius;
  float yaw = start_yaw;
  const float velocity = is_forward ? 1.0 : -0.5;

  if (is_left_turn) {
    if (end_yaw < start_yaw) end_yaw += M_PI_2;
    while (yaw < end_yaw) {
      PathPointWithLaneId p = generateArcPathPoint(center, radius, yaw, is_left_turn, is_forward);
      p.point.longitudinal_velocity_mps = velocity;
      path.points.push_back(p);
      yaw += yaw_interval;
    }
  } else {  // right_turn
    if (end_yaw > start_yaw) end_yaw -= M_PI_2;
    while (yaw > end_yaw) {
      PathPointWithLaneId p = generateArcPathPoint(center, radius, yaw, is_left_turn, is_forward);
      p.point.longitudinal_velocity_mps = velocity;
      path.points.push_back(p);
      yaw -= yaw_interval;
    }
  }
  PathPointWithLaneId p = generateArcPathPoint(center, radius, end_yaw, is_left_turn, is_forward);
  p.point.longitudinal_velocity_mps = 0.0;
  path.points.push_back(p);

  return path;
}

PathPointWithLaneId ParallelParkingPlanner::generateArcPathPoint(
  const Pose & center, const float radius, const float yaw, const bool is_left_turn, const bool is_forward)
{
  Pose pose_centercoords;
  pose_centercoords.position.x = radius * std::cos(yaw);
  pose_centercoords.position.y = radius * std::sin(yaw);

  tf2::Quaternion quat;
  if ((is_left_turn && !is_forward) || (!is_left_turn && is_forward)) {
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
  planner_data_ = planner_data;
  lane_departure_checker_->setVehicleInfo(planner_data_->parameters.vehicle_info);
  auto common_params = planner_data_->parameters;
  max_steer_rad_ = deg2rad(max_steer_deg_);

  R_E_min_ = common_params.wheel_base / std::tan(max_steer_rad_);
  R_Bl_min_ = std::hypot(
    R_E_min_ + common_params.wheel_tread / 2 + common_params.left_over_hang,
    common_params.wheel_base + common_params.front_overhang);
}
}  // namespace behavior_path_planner
