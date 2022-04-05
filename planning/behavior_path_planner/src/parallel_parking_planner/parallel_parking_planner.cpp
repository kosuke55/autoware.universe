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
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <tf2_eigen/tf2_eigen.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

using autoware_auto_planning_msgs::msg::PathWithLaneId;
using behavior_path_planner::util::removeOverlappingPoints;
using geometry_msgs::msg::Pose;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::Transform;
using geometry_msgs::msg::TransformStamped;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::deg2rad;
using tier4_autoware_utils::fromMsg;
using tier4_autoware_utils::inverseTransformPoint;
using tier4_autoware_utils::normalizeRadian;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Point3d;
using tier4_autoware_utils::pose2transform;
using tier4_autoware_utils::toMsg;
using tier4_autoware_utils::translateLocal;

namespace
{
PoseArray PathWithLaneId2PoseArray(const PathWithLaneId & path)
{
  PoseArray pose_array;
  pose_array.header = path.header;

  for (const auto & point : path.points) {
    pose_array.poses.push_back(point.point.pose);
  }

  return pose_array;
}

Pose transformPose(const Pose & pose, const TransformStamped & transform)
{
  PoseStamped transformed_pose;
  PoseStamped orig_pose;
  orig_pose.pose = pose;
  tf2::doTransform(orig_pose, transformed_pose, transform);

  return transformed_pose.pose;
}

Pose transformPose(const Pose & pose, const Transform & transform)
{
  TransformStamped transform_stamped;
  transform_stamped.transform = transform;
  PoseStamped transformed_pose;
  PoseStamped orig_pose;
  orig_pose.pose = pose;
  tf2::doTransform(orig_pose, transformed_pose, transform_stamped);

  return transformed_pose.pose;
}

PathWithLaneId concatePath(const PathWithLaneId path1, const PathWithLaneId path2)
{
  PathWithLaneId path = path1;
  for (const auto & point : path2.points) {
    path.points.push_back(point);
  }
  return path;
}
}  // namespace

namespace behavior_path_planner
{
// ParallelParkingPlanner::ParallelParkingPlanner()
// {
//   max_steer_rad_ = tier4_autoware_utils::deg2rad(max_steer_deg_);
// }

PathWithLaneId ParallelParkingPlanner::getCurrentPath()
{
  const auto current_path = paths_.at(current_path_idx_);
  const auto current_target = current_path.points.back();
  const auto self_pose = planner_data_->self_pose->pose;

  const float th_arrived_distance_m = 1.0;
  const bool is_near_target =
    tier4_autoware_utils::calcDistance2d(current_target, self_pose) < th_arrived_distance_m;

  const float ego_speed = std::abs(planner_data_->self_odometry->twist.twist.linear.x);

  const float th_stopped_velocity_mps = 0.1;
  const bool is_stopped = std::abs(ego_speed) < th_stopped_velocity_mps;

  if (is_near_target && is_stopped) {
    current_path_idx_ += 1;
    rclcpp::Rate(1.0).sleep();
  }
  current_path_idx_ = std::min(current_path_idx_, paths_.size() - 1);

  return paths_.at(current_path_idx_);
}

PathWithLaneId ParallelParkingPlanner::generate()
{
  paths_.clear();
  current_path_idx_ = 0;
  return planOneTraial();
}

PathWithLaneId ParallelParkingPlanner::planOneTraial()
{
  PathWithLaneId path;
  const auto self_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();

  const float self_yaw = tf2::getYaw(self_pose.orientation);
  const float goal_yaw = tf2::getYaw(goal_pose.orientation);
  const float psi = normalizeRadian(self_yaw - goal_yaw);
  std::cerr << "self_yaw: " << self_yaw << "goal_yaw: " << goal_yaw << "psi: " << psi << std::endl;
  const auto common_params = planner_data_->parameters;

  Pose Cr = translateLocal(goal_pose, Eigen::Vector3d(0, -R_E_min_, 0));
  Cr_.pose = Cr;
  Cr_.header = planner_data_->route_handler->getRouteHeader();
  const float d_Cr_Einit = calcDistance2d(Cr, self_pose);
  std::cerr << "R_E_min_: : " << R_E_min_ << std::endl;
  std::cerr << "d_Cr_Einit: " << d_Cr_Einit << std::endl;

  geometry_msgs::msg::Point Cr_goalcoords = inverseTransformPoint(Cr.position, goal_pose);
  geometry_msgs::msg::Point self_point_goalcoords =
    inverseTransformPoint(self_pose.position, goal_pose);
  std::cerr << "Cr_goalcoords: " << Cr_goalcoords.x << ", " << Cr_goalcoords.y << std::endl;
  std::cerr << "self_point_goalcoords: " << self_point_goalcoords.x << ", "
            << self_point_goalcoords.y << std::endl;

  const float alpha =
    M_PI_2 - psi + std::asin((self_point_goalcoords.y - Cr_goalcoords.y) / d_Cr_Einit);
  std::cerr << "M_PI_2: " << M_PI_2 << " psi: " << psi
            << " Cr_goalcoords.y - self_point_goalcoords.y / d_Cr_Einit "
            << (Cr_goalcoords.y - self_point_goalcoords.y) / d_Cr_Einit
            << " std::asin(Cr_goalcoords.y - self_point_goalcoords.y / d_Cr_Einit): "
            << std::asin((Cr_goalcoords.y - self_point_goalcoords.y) / d_Cr_Einit) << std::endl;
  std::cerr << "alpha: " << alpha << std::endl;

  const float R_Einit_l = (std::pow(d_Cr_Einit, 2) - std::pow(R_E_min_, 2)) /
                          (2 * (R_E_min_ + d_Cr_Einit * std::cos(alpha)));
  std::cerr << "R_Einit_l: " << R_Einit_l << std::endl;
  if (R_Einit_l <= 0) {
    return path;
  }

  const float steer_l = std::atan(common_params.wheel_base / R_Einit_l);
  std::cerr << "steer_l: " << steer_l << std::endl;

  Pose Cl = translateLocal(self_pose, Eigen::Vector3d(0, R_Einit_l, 0));
  Cl_.pose = Cl;
  Cl_.header = planner_data_->route_handler->getRouteHeader();

  const float theta_l = std::acos(
    (std::pow(R_Einit_l, 2) + std::pow(R_Einit_l + R_E_min_, 2) - std::pow(d_Cr_Einit, 2)) /
    (2 * R_Einit_l * (R_Einit_l + R_E_min_)));

  lanelet::ConstLanelet current_lane;
  planner_data_->route_handler->getClosestLaneletWithinRoute(self_pose, &current_lane);
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

  PathWithLaneId concat_path = concatePath(paths_.at(0), paths_.at(1));
  path.header = planner_data_->route_handler->getRouteHeader();
  path_pose_array_ = PathWithLaneId2PoseArray(concat_path);

  // geometry_msgs::msg::Transform transform_Cl;
  // transform_Cl.translation.y = R_Einit_l;
  // const auto Cl = transformPoint(fromMsg(self_pose.position), transform_Cl);
  // Cl_.pose = self_pose;
  // Cl_.pose.position = toMsg(Cl);
  // Cl_.header = planner_data_->route_handler->getRouteHeader();

  return concat_path;
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
  pose_centercoords.position.z = center.position.z;

  tf2::Quaternion quat;
  if (is_left_turn) {
    quat.setRPY(0, 0, normalizeRadian(yaw - M_PI_2));
  } else {
    quat.setRPY(0, 0, normalizeRadian(yaw + M_PI_2));
  }
  pose_centercoords.orientation = tf2::toMsg(quat);

  PathPointWithLaneId p{};
  p.point.pose = transformPose(pose_centercoords, pose2transform(center));
  lanelet::ConstLanelet current_lane;
  planner_data_->route_handler->getClosestLaneletWithinRoute(p.point.pose, &current_lane);
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
