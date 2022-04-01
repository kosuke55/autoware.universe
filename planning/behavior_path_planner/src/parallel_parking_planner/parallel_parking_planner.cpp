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

using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::deg2rad;
using tier4_autoware_utils::fromMsg;
using tier4_autoware_utils::inverseTransformPoint;
using tier4_autoware_utils::normalizeRadian;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Point3d;
using tier4_autoware_utils::toMsg;
using tier4_autoware_utils::translateLocal;
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::Pose;
using autoware_auto_planning_msgs::msg::PathWithLaneId;

namespace
{
PoseArray pathWithLaneId2PoseArray(const PathWithLaneId & path)
{
  PoseArray pose_array;
  pose_array.header = path.header;

  for (const auto & point : path.points) {
    pose_array.poses.push_back(point.point.pose);
  }

  return pose_array;
}
}  // namespace

namespace behavior_path_planner
{
// ParallelParkingPlanner::ParallelParkingPlanner()
// {
//   max_steer_rad_ = tier4_autoware_utils::deg2rad(max_steer_deg_);
// }

bool ParallelParkingPlanner::generate()
{
  planOneTraial();
  return true;
}

bool ParallelParkingPlanner::planOneTraial()
{
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
  std::cerr << "d_Cr_Einit: " << d_Cr_Einit << std::endl;

  geometry_msgs::msg::Point Cr_goalcoords = inverseTransformPoint(Cr.position, goal_pose);
  geometry_msgs::msg::Point self_point_goalcoords =
    inverseTransformPoint(self_pose.position, goal_pose);
  std::cerr << "Cr_goalcoords: " << Cr_goalcoords.x << ", " << Cr_goalcoords.y << std::endl;
  std::cerr << "self_point_goalcoords: " << self_point_goalcoords.x << ", "
            << self_point_goalcoords.y << std::endl;

  const float alpha =
    M_PI_2 - psi + std::asin((Cr_goalcoords.y - self_point_goalcoords.y) / d_Cr_Einit);
  std::cerr << "M_PI_2: " << M_PI_2 << " psi: " << psi
            << " Cr_goalcoords.y - self_point_goalcoords.y / d_Cr_Einit "
            << (Cr_goalcoords.y - self_point_goalcoords.y) / d_Cr_Einit
            << " std::asin(Cr_goalcoords.y - self_point_goalcoords.y / d_Cr_Einit): "
            << std::asin((Cr_goalcoords.y - self_point_goalcoords.y) / d_Cr_Einit) << std::endl;
  std::cerr << "alpha: " << alpha << std::endl;

  const float R_Einit_l = (std::pow(d_Cr_Einit, 2) - std::pow(R_E_min_, 2)) /
                          (2 * R_E_min_ + d_Cr_Einit * std::cos(alpha));
  std::cerr << "R_Einit_l: " << R_Einit_l << std::endl;

  const float steer_l = std::atan(common_params.wheel_tread / 2 / R_Einit_l);
  std::cerr << "steer_l: " << steer_l << std::endl;

  Pose Cl = translateLocal(self_pose, Eigen::Vector3d(0, R_Einit_l, 0));
  Cl_.pose = Cl;
  Cl_.header = planner_data_->route_handler->getRouteHeader();

  const float theta_l = std::acos(
    (std::pow(R_Einit_l, 2) + std::pow(R_Einit_l + R_E_min_, 2) - std::pow(d_Cr_Einit, 2)) /
    (2 * R_Einit_l * (R_Einit_l + R_E_min_)));

  PathWithLaneId path{};
  generateArcPath(Cl, R_Einit_l, psi - M_PI_2, psi - M_PI_2 - theta_l, false, path);
  generateArcPath(Cr, R_E_min_, psi + M_PI_2 - theta_l, M_PI_2, true, path);
  path.header = planner_data_->route_handler->getRouteHeader();
  path_pose_array_ = pathWithLaneId2PoseArray(path);

  // geometry_msgs::msg::Transform transform_Cl;
  // transform_Cl.translation.y = R_Einit_l;
  // const auto Cl = transformPoint(fromMsg(self_pose.position), transform_Cl);
  // Cl_.pose = self_pose;
  // Cl_.pose.position = toMsg(Cl);
  // Cl_.header = planner_data_->route_handler->getRouteHeader();

  return true;
}

bool ParallelParkingPlanner::generateArcPath(
  const Pose & center, const float radius, const float start_yaw,
  float end_yaw, const bool is_left_turn, PathWithLaneId & path)
{
  const float interval = 0.3;
  const float yaw_interval = interval / radius;
  float yaw = start_yaw;

  if (is_left_turn) {
    if (end_yaw < start_yaw) end_yaw += M_PI_2;
    while(yaw < end_yaw){
      path.points.push_back(generateArcPathPoint(center, radius, yaw));
      yaw += yaw_interval;
    }
  } else {  // right_turn
    if (end_yaw > start_yaw) end_yaw -= M_PI_2;
    while (yaw > end_yaw) {
      path.points.push_back(generateArcPathPoint(center, radius, yaw));
      yaw -= yaw_interval;
    }
  }
  path.points.push_back(generateArcPathPoint(center, radius, end_yaw));

  return !path.points.empty();
}

PathPointWithLaneId ParallelParkingPlanner::generateArcPathPoint(
  const Pose & center, const float radius, const float yaw)
{ 
  // たぶん間違っている。centerの座標系で+radius * std::cos(yaw); する必要がある。
  PathPointWithLaneId p{};
  p.point.pose.position.x = center.position.x + radius * std::cos(yaw);
  p.point.pose.position.y = center.position.y + radius * std::sin(yaw);
  p.point.pose.position.z = center.position.z;
  //  TODO
  p.point.pose.orientation = center.orientation;

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
