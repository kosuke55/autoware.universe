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

#include <interpolation/spline_interpolation.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tier4_autoware_utils/geometry/geometry.hpp"
#include <tf2_eigen/tf2_eigen.h>

using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::deg2rad;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::toMsg;
using tier4_autoware_utils::fromMsg;

namespace behavior_path_planner
{
// ParallelParkingPlanner::ParallelParkingPlanner()
// {
//   max_steer_rad_ = tier4_autoware_utils::deg2rad(max_steer_deg_);
// }

bool ParallelParkingPlanner::generate() {
  planOneTraial();
  return true; }

bool ParallelParkingPlanner::planOneTraial()
{
  const auto self_pose = planner_data_->self_pose->pose;
  const float self_yaw = tf2::getYaw(self_pose.orientation);
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto common_params = planner_data_->parameters;

  geometry_msgs::msg::Transform transform_Cr;
  transform_Cr.translation.y = -R_Er_min_;
  const auto Cr = transformPoint(fromMsg(goal_pose.position), transform_Cr);
  Cr_.pose = goal_pose;
  Cr_.pose.position = toMsg(Cr);
  Cr_.header = planner_data_->route_handler->getRouteHeader();
  const float d_Cr_Einit = calcDistance2d(toMsg(Cr), self_pose);

  const float alpha = M_PI_2 + self_yaw + std::asin(Cr.y() - self_pose.position.y) / d_Cr_Einit;
  const float R_Einit_l = (std::pow(d_Cr_Einit, 2) - std::pow(d_Cr_Einit, 2)) /
                          (2 * R_Er_min_ + d_Cr_Einit * std::cos(alpha));
  const float steer_l = std::atan(common_params.wheel_tread / 2 / R_Einit_l);

  geometry_msgs::msg::Transform transform_Cl;
  transform_Cl.translation.y = R_Einit_l;
  const auto Cl = transformPoint(fromMsg(self_pose.position), transform_Cl);
  Cl_.pose = self_pose;
  Cl_.pose.position = toMsg(Cl);
  Cl_.header = planner_data_->route_handler->getRouteHeader();

  return true;
}

void ParallelParkingPlanner::setParams(const std::shared_ptr<const PlannerData> & planner_data)
{
  // planner_data_ = std::make_shared<PlannerData>();
  planner_data_ = planner_data;
  auto common_params = planner_data_->parameters;
  max_steer_rad_ = deg2rad(max_steer_deg_);

  R_Er_min_ = common_params.wheel_base / std::tan(max_steer_rad_);
  R_Bl_min_ = std::hypot(
    R_Er_min_ + common_params.wheel_tread / 2 + common_params.left_over_hang,
    common_params.wheel_base + common_params.front_overhang);
}
}  // namespace behavior_path_planner
