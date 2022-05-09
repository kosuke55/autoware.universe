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

#include "behavior_path_planner/scene_module/pull_over/pull_over_module.hpp"

#include "behavior_path_planner/behavior_path_planner_node.hpp"
#include "behavior_path_planner/occupancy_grid_map/occupancy_grid_map.hpp"
#include "behavior_path_planner/parallel_parking_planner/parallel_parking_planner.hpp"
#include "behavior_path_planner/path_shifter/path_shifter.hpp"
#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/avoidance/debug.hpp"
#include "behavior_path_planner/scene_module/pull_over/util.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using nav_msgs::msg::OccupancyGrid;
using tier4_autoware_utils::calcOffsetPose;
using tier4_autoware_utils::createQuaternionFromYaw;
using tier4_autoware_utils::inverseTransformPose;
using tier4_autoware_utils::transformPose;

using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcSignedArcLength;
using tier4_autoware_utils::createDefaultMarker;
using tier4_autoware_utils::createMarkerColor;
using tier4_autoware_utils::createMarkerScale;
using tier4_autoware_utils::createPoint;
using tier4_autoware_utils::findNearestIndex;

namespace behavior_path_planner
{
PullOverModule::PullOverModule(
  const std::string & name, rclcpp::Node & node, const PullOverParameters & parameters)
: SceneModuleInterface{name, node}, parameters_{parameters}
{
  approval_handler_.waitApproval();
  goal_pose_pub_ =
    node.create_publisher<PoseStamped>("/planning/scenario_planning/modified_goal", 1);
  parking_area_pub_ = node.create_publisher<MarkerArray>("~/pull_over/debug/parking_area", 1);
  // Only for arc paths
  Cl_pub_ = node.create_publisher<PoseStamped>("~/pull_over/debug/Cl", 1);
  Cr_pub_ = node.create_publisher<PoseStamped>("~/pull_over/debug/Cr", 1);
  start_pose_pub_ = node.create_publisher<PoseStamped>("~/pull_over/debug/start_pose", 1);
  path_pose_array_pub_ = node.create_publisher<PoseArray>("~/pull_over/debug/path_pose_array", 1);
}

// This function is needed for waiting for planner_data_
void PullOverModule::updateOccupancyGrid()
{
  occupancy_grid_map_.setMap(*(planner_data_->occupancy_grid));
}

BehaviorModuleOutput PullOverModule::run()
{
  approval_handler_.clearWaitApproval();
  current_state_ = BT::NodeStatus::RUNNING;
  updateOccupancyGrid();
  return plan();
}

void PullOverModule::onEntry()
{
  RCLCPP_DEBUG(getLogger(), "PULL_OVER onEntry");
  current_state_ = BT::NodeStatus::SUCCESS;

  // Initialize occupancy grid map
  OccupancyGridMapParam occupancy_grid_map_param;
  const double margin = parameters_.collision_check_margin;
  occupancy_grid_map_param.vehicle_shape.length =
    planner_data_->parameters.vehicle_length + 2 * margin;
  occupancy_grid_map_param.vehicle_shape.width =
    planner_data_->parameters.vehicle_width + 2 * margin;
  occupancy_grid_map_param.vehicle_shape.base2back =
    planner_data_->parameters.base_link2rear + margin;
  occupancy_grid_map_param.theta_size = parameters_.theta_size;
  occupancy_grid_map_param.obstacle_threshold = parameters_.obstacle_threshold;
  occupancy_grid_map_.setParam(occupancy_grid_map_param);

  // Initialize sratus
  parallel_parking_planner_.clear();
  parallel_parking_prameters_ = ParallelParkingParameters{
    parameters_.th_arrived_distance_m, parameters_.th_stopped_velocity_mps,
    parameters_.after_forward_parking_straight_distance,
    parameters_.after_backward_parking_straight_distance, parameters_.decide_path_distance};
  status_.has_decided_path = false;
  status_.path_type = PathType::NONE;
  status_.is_safe = false;

  // Use refined goal as modified goal when disabling goal research
  if (!parameters_.enable_goal_research) {
    goal_candidates_.clear();
    GoalCandidate goal_candidate;
    goal_candidate.goal_pose = getRefinedGoal();
    goal_candidates_.push_back(goal_candidate);
  }

  approval_handler_.waitApproval();
}

void PullOverModule::onExit()
{
  RCLCPP_DEBUG(getLogger(), "PULL_OVER onExit");
  approval_handler_.clearWaitApproval();
  current_state_ = BT::NodeStatus::IDLE;
}

bool PullOverModule::isExecutionRequested() const
{
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  lanelet::Lanelet closest_shoulder_lanelet;
  bool goal_is_in_shoulder_lane = false;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto current_lanes = util::getExtendedCurrentLanes(planner_data_);

  // check if goal_pose is in shoulder lane
  if (lanelet::utils::query::getClosestLanelet(
        planner_data_->route_handler->getShoulderLanelets(), goal_pose,
        &closest_shoulder_lanelet)) {
    // check if goal pose is in shoulder lane
    if (lanelet::utils::isInLanelet(goal_pose, closest_shoulder_lanelet, 0.1)) {
      const auto lane_yaw =
        lanelet::utils::getLaneletAngle(closest_shoulder_lanelet, goal_pose.position);
      const auto goal_yaw = tf2::getYaw(goal_pose.orientation);
      const auto angle_diff = tier4_autoware_utils::normalizeRadian(lane_yaw - goal_yaw);
      constexpr double th_angle = M_PI / 4;
      if (std::abs(angle_diff) < th_angle) {
        goal_is_in_shoulder_lane = true;
      }
    }
  }

  // check if self pose is NOT in shoulder lane
  bool self_is_in_shoulder_lane = false;
  const auto self_pose = planner_data_->self_pose->pose;
  if (lanelet::utils::query::getClosestLanelet(
        planner_data_->route_handler->getShoulderLanelets(), self_pose,
        &closest_shoulder_lanelet)) {
    self_is_in_shoulder_lane =
      lanelet::utils::isInLanelet(self_pose, closest_shoulder_lanelet, 0.1);
  }

  return goal_is_in_shoulder_lane && !self_is_in_shoulder_lane &&
         isLongEnough(current_lanes, goal_pose);
}

bool PullOverModule::isExecutionReady() const { return true; }

Pose PullOverModule::getRefinedGoal()
{
  lanelet::ConstLanelet goal_lane;
  Pose goal_pose = planner_data_->route_handler->getGoalPose();

  lanelet::Lanelet closest_shoulder_lanelet;

  lanelet::utils::query::getClosestLanelet(
    planner_data_->route_handler->getShoulderLanelets(), planner_data_->self_pose->pose,
    &closest_shoulder_lanelet);

  Pose refined_goal_pose =
    lanelet::utils::getClosestCenterPose(closest_shoulder_lanelet, goal_pose.position);

  const double distance_to_left_bound = util::getDistanceToShoulderBoundary(
    planner_data_->route_handler->getShoulderLanelets(), refined_goal_pose);
  const double offset_from_center_line = distance_to_left_bound +
                                         planner_data_->parameters.vehicle_width / 2 +
                                         parameters_.margin_from_boundary;
  refined_goal_pose = calcOffsetPose(refined_goal_pose, 0, -offset_from_center_line, 0);

  return refined_goal_pose;
}

void PullOverModule::researchGoal()
{
  const auto common_param = occupancy_grid_map_.getParam();
  const Pose current_pose = planner_data_->self_pose->pose;
  const Pose goal_pose = getRefinedGoal();
  double dx = -parameters_.backward_goal_search_length;

  // Avoid adding areas that are in conflict from the start.
  bool prev_is_collided = true;

  pull_over_areas_.clear();
  const Pose goal_pose_map_coords = global2local(occupancy_grid_map_.getMap(), goal_pose);
  Pose start_pose = calcOffsetPose(goal_pose, dx, 0, 0);
  // Serch non collision areas around the goal
  while (true) {
    bool is_last_search = (dx >= parameters_.forward_goal_search_length);
    Pose serach_pose = calcOffsetPose(goal_pose_map_coords, dx, 0, 0);
    bool is_collided = occupancy_grid_map_.detectCollision(
      pose2index(occupancy_grid_map_.getMap(), serach_pose, common_param.theta_size), false);
    // Add area when (1) chnage non-collision -> collison or (2) last serach without collision
    if ((!prev_is_collided && is_collided) || (!is_collided && is_last_search)) {
      Pose end_pose = calcOffsetPose(goal_pose, dx, 0, 0);
      if (!pull_over_areas_.empty()) {
        auto prev_area = pull_over_areas_.back();
        // If the current area overlaps the previous area, merge them.
        if (
          calcDistance2d(prev_area.end_pose, start_pose) <
          planner_data_->parameters.vehicle_length) {
          pull_over_areas_.pop_back();
          start_pose = prev_area.start_pose;
        }
      }
      pull_over_areas_.push_back(PullOverArea{start_pose, end_pose});
    }
    if (is_last_search) break;

    if ((prev_is_collided && !is_collided)) {
      start_pose = calcOffsetPose(goal_pose, dx, 0, 0);
    }
    prev_is_collided = is_collided;
    dx += 0.05;
  }

  // Find goals in pull over areas.
  goal_candidates_.clear();
  for (double dx = -parameters_.backward_goal_search_length;
       dx <= parameters_.forward_goal_search_length; dx += parameters_.goal_search_interval) {
    Pose search_pose = calcOffsetPose(goal_pose, dx, 0, 0);
    for (const auto area : pull_over_areas_) {
      const Pose start_to_search = inverseTransformPose(search_pose, area.start_pose);
      const Pose end_to_search = inverseTransformPose(search_pose, area.end_pose);
      const Pose current_to_search = inverseTransformPose(search_pose, current_pose);
      if (
        start_to_search.position.x > parameters_.goal2obj_margin &&
        end_to_search.position.x < -parameters_.goal2obj_margin &&
        current_to_search.position.x > -parameters_.backward_ignore_distance) {
        GoalCandidate goal_candidate;
        goal_candidate.goal_pose = search_pose;
        goal_candidate.distance_from_original_goal =
          std::abs(inverseTransformPose(search_pose, goal_pose).position.x);
        goal_candidates_.push_back(goal_candidate);
      }
    }
  }
  // Sort with distance from original goal
  std::sort(goal_candidates_.begin(), goal_candidates_.end());
}

BT::NodeStatus PullOverModule::updateState()
{
  if (hasFinishedPullOver()) {
    current_state_ = BT::NodeStatus::SUCCESS;
    return current_state_;
  }
  current_state_ = BT::NodeStatus::RUNNING;

  return current_state_;
}

BehaviorModuleOutput PullOverModule::plan()
{
  const auto common_parameters = planner_data_->parameters;

  const auto current_lanes = util::getExtendedCurrentLanes(planner_data_);
  const auto pull_over_lanes = getPullOverLanes(current_lanes);
  lanelet::ConstLanelets lanes;
  lanes.insert(lanes.end(), current_lanes.begin(), current_lanes.end());
  lanes.insert(lanes.end(), pull_over_lanes.begin(), pull_over_lanes.end());

  // Research goal when enabling research and final path has not been decieded
  if (parameters_.enable_goal_research && !status_.has_decided_path) researchGoal();

  // Check if we have to deciede path
  if (status_.is_safe) {
    Pose parking_start_pose;
    if (status_.path_type == PathType::SHIFT) {
      parking_start_pose = shift_parking_path_.shift_point.start;
    } else if (
      status_.path_type == PathType::ARC_FORWARD || status_.path_type == PathType::ARC_BACK) {
      parking_start_pose = parallel_parking_planner_.getStartPose().pose;
    }
    const auto dist_to_parking_start_pose = calcSignedArcLength(
      status_.path.points, planner_data_->self_pose->pose, parking_start_pose.position,
      std::numeric_limits<double>::max(), M_PI_2);

    if (*dist_to_parking_start_pose < parameters_.decide_path_distance) {
      status_.has_decided_path = true;
    }
  }

  // Use decided path
  if (status_.has_decided_path) {
    if (status_.path_type == PathType::ARC_FORWARD || status_.path_type == PathType::ARC_BACK) {
      status_.path = parallel_parking_planner_.getCurrentPath();
    }
  }
  // Replan shift -> arc forward -> arc backward path with each goal candidate.
  else {
    status_.path_type = PathType::NONE;
    status_.is_safe = false;
    for (const auto goal_candidate : goal_candidates_) {
      if (status_.is_safe) break;
      modified_goal_pose_ = goal_candidate.goal_pose;
      if (isLongEnough(current_lanes, modified_goal_pose_) && planShiftPath()) {
        // shift parking path already confirm safe in it's own function.
        status_.path = shift_parking_path_.path;
        status_.path_type = PathType::SHIFT;
        status_.is_safe = true;
      } else {
        // Generate arc forward path then arc backward path.
        for (const auto is_forward : {true, false}) {
          parallel_parking_planner_.setParams(planner_data_, parallel_parking_prameters_);
          if (
            parallel_parking_planner_.plan(modified_goal_pose_, lanes, is_forward) &&
            !occupancy_grid_map_.hasObstacleOnPath(
              parallel_parking_planner_.getFullPath(), false)) {
            status_.path = parallel_parking_planner_.getCurrentPath();
            status_.path_type = is_forward ? PathType::ARC_FORWARD : PathType::ARC_BACK;
            status_.is_safe = true;
            break;
          }
        }
      }
    }
    // Decelerate before the minimum shift distance from the goal search area.
    const Pose goal_pose = getRefinedGoal();
    const auto arc_coordinates = lanelet::utils::getArcCoordinates(current_lanes, goal_pose);
    const Pose search_start_pose = calcOffsetPose(
      goal_pose, -parameters_.backward_goal_search_length, -arc_coordinates.distance, 0);
    const auto search_start_signed_arg_length = calcSignedArcLength(
      status_.path.points, planner_data_->self_pose->pose, search_start_pose.position,
      std::numeric_limits<double>::max(), M_PI_2);
    double dist_sum = 0;
    for (size_t i = 0; i < status_.path.points.size() - 1; i++) {
      dist_sum += calcDistance2d(status_.path.points.at(i), status_.path.points.at(i + 1));
      if (dist_sum > *search_start_signed_arg_length - calcMinimumShiftPathDistance()) {
        status_.path.points.at(i).point.longitudinal_velocity_mps = std::min(
          status_.path.points.at(i).point.longitudinal_velocity_mps,
          static_cast<float>(parameters_.pull_over_velocity));
      }
    }
  }

  BehaviorModuleOutput output;
  output.path = status_.is_safe ? std::make_shared<PathWithLaneId>(status_.path)
                                : std::make_shared<PathWithLaneId>(getReferencePath());

  publishDebugData();

  return output;
}

// No specific path for the cadidte. It's same to the one generated by plan().
PathWithLaneId PullOverModule::planCandidate() const { return {}; }

BehaviorModuleOutput PullOverModule::planWaitingApproval()
{
  updateOccupancyGrid();
  BehaviorModuleOutput out;
  out.path = std::make_shared<PathWithLaneId>(getReferencePath());
  out.path_candidate = std::make_shared<PathWithLaneId>(*(plan().path));
  if (
    status_.is_safe &&
    (status_.path_type == PathType::ARC_FORWARD || status_.path_type == PathType::ARC_BACK)) {
    out.path_candidate = std::make_shared<PathWithLaneId>(parallel_parking_planner_.getFullPath());
  }
  return out;
}

void PullOverModule::setParameters(const PullOverParameters & parameters)
{
  parameters_ = parameters;
}

bool PullOverModule::planShiftPath()
{
  // RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  const auto & route_handler = planner_data_->route_handler;
  const auto common_parameters = planner_data_->parameters;

  const auto current_lanes = util::getExtendedCurrentLanes(planner_data_);

  lanelet::ConstLanelet target_shoulder_lane;

  if (route_handler->getPullOverTarget(
        route_handler->getShoulderLanelets(), &target_shoulder_lane)) {
    route_handler->setPullOverGoalPose(
      target_shoulder_lane, common_parameters.vehicle_width, parameters_.margin_from_boundary);
  } else {
    RCLCPP_ERROR(getLogger(), "failed to get shoulder lane!!!");
  }

  // Find pull_over path
  bool found_valid_path, found_safe_path;
  const auto pull_over_lanes = getPullOverLanes(current_lanes);
  std::tie(found_valid_path, found_safe_path) =
    getSafePath(pull_over_lanes, check_distance_, modified_goal_pose_, shift_parking_path_);

  // Generate drivable area
  {
    lanelet::ConstLanelets lanes;
    lanes.insert(lanes.end(), current_lanes.begin(), current_lanes.end());
    lanes.insert(lanes.end(), pull_over_lanes.begin(), pull_over_lanes.end());
    shift_parking_path_.path.drivable_area = util::generateDrivableArea(
      lanes, common_parameters.drivable_area_resolution, common_parameters.vehicle_length,
      planner_data_);
  }
  shift_parking_path_.path.header = planner_data_->route_handler->getRouteHeader();
  return found_safe_path;
}

PathWithLaneId PullOverModule::getReferencePath() const
{
  PathWithLaneId reference_path;

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto common_parameters = planner_data_->parameters;

  // Set header
  reference_path.header = route_handler->getRouteHeader();

  const auto current_lanes = util::getExtendedCurrentLanes(planner_data_);

  if (current_lanes.empty()) {
    return reference_path;
  }

  reference_path = util::getCenterLinePath(
    *route_handler, current_lanes, current_pose, common_parameters.backward_path_length,
    common_parameters.forward_path_length, common_parameters);

  reference_path = util::setDecelerationVelocity(
    *route_handler, reference_path, current_lanes, parameters_.after_pull_over_straight_distance,
    common_parameters.minimum_pull_over_length, parameters_.before_pull_over_straight_distance,
    parameters_.deceleration_interval, goal_pose);

  reference_path.drivable_area = util::generateDrivableArea(
    current_lanes, common_parameters.drivable_area_resolution, common_parameters.vehicle_length,
    planner_data_);

  return reference_path;
}

lanelet::ConstLanelets PullOverModule::getPullOverLanes(
  const lanelet::ConstLanelets & current_lanes) const
{
  lanelet::ConstLanelets pull_over_lanes;
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  lanelet::ConstLanelet target_shoulder_lane;

  if (current_lanes.empty()) {
    return pull_over_lanes;
  }

  // Get shoulder lanes
  lanelet::ConstLanelet current_lane;
  lanelet::utils::query::getClosestLanelet(
    current_lanes, planner_data_->self_pose->pose, &current_lane);

  if (route_handler->getPullOverTarget(
        route_handler->getShoulderLanelets(), &target_shoulder_lane)) {
    pull_over_lanes = route_handler->getShoulderLaneletSequence(
      target_shoulder_lane, current_pose, pull_over_lane_length_, pull_over_lane_length_);

  } else {
    pull_over_lanes.clear();
  }

  return pull_over_lanes;
}

std::pair<bool, bool> PullOverModule::getSafePath(
  const lanelet::ConstLanelets & pull_over_lanes, const double check_distance, const Pose goal_pose,
  ShiftParkingPath & safe_path) const
{
  std::vector<ShiftParkingPath> valid_paths;

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto current_twist = planner_data_->self_odometry->twist.twist;
  const auto common_parameters = planner_data_->parameters;

  const auto current_lanes = util::getExtendedCurrentLanes(planner_data_);
  if (!isLongEnough(current_lanes, goal_pose)) {
    return std::make_pair(false, false);
  }
  if (!pull_over_lanes.empty()) {
    // find candidate paths
    const auto pull_over_paths = pull_over_utils::getShiftParkingPaths(
      *route_handler, current_lanes, pull_over_lanes, current_pose, goal_pose, current_twist,
      common_parameters, parameters_);

    // get lanes used for detection
    lanelet::ConstLanelets check_lanes;
    if (!pull_over_paths.empty()) {
      const auto & longest_path = pull_over_paths.front();
      // we want to see check_distance [m] behind vehicle so add lane changing length
      const double check_distance_with_path =
        check_distance + longest_path.preparation_length + longest_path.pull_over_length;
      check_lanes = route_handler->getCheckTargetLanesFromPath(
        longest_path.path, pull_over_lanes, check_distance_with_path);
    }

    // select valid path
    valid_paths = pull_over_utils::selectValidPaths(
      pull_over_paths, current_lanes, check_lanes, *route_handler->getOverallGraphPtr(),
      current_pose, route_handler->isInGoalRouteSection(current_lanes.back()), goal_pose);

    if (valid_paths.empty()) {
      return std::make_pair(false, false);
    }
    // select safe path
    bool found_safe_path = pull_over_utils::selectSafePath(
      valid_paths, current_lanes, check_lanes, planner_data_->dynamic_object, current_pose,
      current_twist, common_parameters.vehicle_width, parameters_, occupancy_grid_map_, &safe_path);
    safe_path.is_safe = found_safe_path;
    return std::make_pair(true, found_safe_path);
  }
  return std::make_pair(false, false);
}

double PullOverModule::calcMinimumShiftPathDistance() const
{
  PathShifter path_shifter;
  const double maximum_jerk = parameters_.maximum_lateral_jerk;
  const double pull_over_velocity = parameters_.pull_over_velocity;
  const auto current_pose = planner_data_->self_pose->pose;
  const double distance_after_pull_over = parameters_.after_pull_over_straight_distance;
  const double distance_before_pull_over = parameters_.before_pull_over_straight_distance;
  const auto & route_handler = planner_data_->route_handler;

  double distance_to_left_bound =
    util::getDistanceToShoulderBoundary(route_handler->getShoulderLanelets(), current_pose);
  double offset_from_center_line = distance_to_left_bound +
                                   planner_data_->parameters.vehicle_width / 2 +
                                   parameters_.margin_from_boundary;

  // calculate minimum pull over distance at pull over velocity, maximum jerk and side offset
  const double pull_over_distance_min = path_shifter.calcLongitudinalDistFromJerk(
    abs(offset_from_center_line), maximum_jerk, pull_over_velocity);
  const double pull_over_total_distance_min =
    distance_after_pull_over + pull_over_distance_min + distance_before_pull_over;

  return pull_over_total_distance_min;
}

bool PullOverModule::isLongEnough(
  const lanelet::ConstLanelets & lanelets, const Pose goal_pose, const double buffer) const
{
  const auto current_pose = planner_data_->self_pose->pose;
  const double distance_to_goal =
    std::abs(util::getSignedDistance(current_pose, goal_pose, lanelets));

  return distance_to_goal > calcMinimumShiftPathDistance() + buffer;
}

bool PullOverModule::hasFinishedPullOver()
{
  // check ego car is close enough to goal pose
  const auto current_pose = planner_data_->self_pose->pose;
  const bool car_is_on_goal =
    calcDistance2d(current_pose, modified_goal_pose_) < parameters_.th_arrived_distance_m;

  // check ego car is stopping
  odometry_buffer_.push_back(planner_data_->self_odometry);
  // Delete old data in buffer
  while (true) {
    const auto time_diff = rclcpp::Time(odometry_buffer_.back()->header.stamp) -
                           rclcpp::Time(odometry_buffer_.front()->header.stamp);
    if (time_diff.seconds() < parameters_.th_stopped_time_sec) {
      break;
    }
    odometry_buffer_.pop_front();
  }
  bool is_stopped = true;
  for (const auto odometry : odometry_buffer_) {
    const double ego_vel = util::l2Norm(odometry->twist.twist.linear);
    if (ego_vel > parameters_.th_stopped_velocity_mps) {
      is_stopped = false;
      break;
    }
  }
  return car_is_on_goal && is_stopped;
}

// Not used.
std::pair<HazardLightsCommand, double> PullOverModule::getHazard(
  const lanelet::ConstLanelets & target_lanes, const Pose & current_pose, const Pose & goal_pose,
  const double & velocity, const double & hazard_on_threshold_dis,
  const double & hazard_on_threshold_vel, const double & base_link2front) const
{
  HazardLightsCommand hazard_signal;
  const double max_distance = std::numeric_limits<double>::max();

  double distance_to_target_pose;   // distance from current pose to target pose on target lanes
  double distance_to_target_point;  // distance from vehicle front to target point on target lanes.
  {
    const auto arc_position_target_pose =
      lanelet::utils::getArcCoordinates(target_lanes, goal_pose);
    const auto arc_position_current_pose =
      lanelet::utils::getArcCoordinates(target_lanes, current_pose);
    distance_to_target_pose = arc_position_target_pose.length - arc_position_current_pose.length;
    distance_to_target_point = distance_to_target_pose - base_link2front;
  }

  if (
    distance_to_target_pose < hazard_on_threshold_dis && abs(velocity) < hazard_on_threshold_vel) {
    hazard_signal.command = HazardLightsCommand::ENABLE;
    return std::make_pair(hazard_signal, distance_to_target_point);
  }

  return std::make_pair(hazard_signal, max_distance);
}

Marker PullOverModule::createParkingAreaMarker(
  const Pose start_pose, const Pose end_pose, const int32_t id)
{
  const auto color = status_.has_decided_path ? createMarkerColor(1.0, 1.0, 0.0, 0.999)   // yellow
                                              : createMarkerColor(0.0, 1.0, 0.0, 0.999);  // green
  Marker marker = createDefaultMarker(
    "map", planner_data_->route_handler->getRouteHeader().stamp, "collision_polygons", id,
    visualization_msgs::msg::Marker::LINE_STRIP, createMarkerScale(0.1, 0.0, 0.0), color);

  auto p_left_front = calcOffsetPose(
                        end_pose, planner_data_->parameters.base_link2front,
                        planner_data_->parameters.vehicle_width / 2, 0)
                        .position;
  marker.points.push_back(createPoint(p_left_front.x, p_left_front.y, p_left_front.z));

  auto p_right_front = calcOffsetPose(
                         end_pose, planner_data_->parameters.base_link2front,
                         -planner_data_->parameters.vehicle_width / 2, 0)
                         .position;
  marker.points.push_back(createPoint(p_right_front.x, p_right_front.y, p_right_front.z));

  auto p_right_back = calcOffsetPose(
                        start_pose, -planner_data_->parameters.base_link2rear,
                        -planner_data_->parameters.vehicle_width / 2, 0)
                        .position;
  marker.points.push_back(createPoint(p_right_back.x, p_right_back.y, p_right_back.z));

  auto p_left_back = calcOffsetPose(
                       start_pose, -planner_data_->parameters.base_link2rear,
                       planner_data_->parameters.vehicle_width / 2, 0)
                       .position;
  marker.points.push_back(createPoint(p_left_back.x, p_left_back.y, p_left_back.z));
  marker.points.push_back(createPoint(p_left_front.x, p_left_front.y, p_left_front.z));

  return marker;
}

void PullOverModule::publishDebugData()
{
  auto header = planner_data_->route_handler->getRouteHeader();
  // Publish the modified goal only when it's path is safe.
  PoseStamped goal_pose_stamped;
  goal_pose_stamped.header = header;
  if (status_.is_safe) goal_pose_stamped.pose = modified_goal_pose_;
  goal_pose_pub_->publish(goal_pose_stamped);

  // Visualize pull over areas
  if (parameters_.enable_goal_research) {
    MarkerArray marker_array;
    for (int32_t i = 0; i < pull_over_areas_.size(); i++) {
      marker_array.markers.push_back(createParkingAreaMarker(
        pull_over_areas_.at(i).start_pose, pull_over_areas_.at(i).end_pose, i));
    }
    parking_area_pub_->publish(marker_array);
  }

  // Only for arc paths. Initialize data not to publish them when using other path.
  PoseStamped Cl, Cr, start_pose;
  PoseArray path_pose_array;
  Cl.header = header;
  Cr.header = header;
  start_pose.header = header;
  path_pose_array.header = header;
  if (
    (status_.path_type == PathType::ARC_FORWARD) ||
    (status_.path_type == PathType::ARC_BACK) && status_.is_safe) {
    Cl = parallel_parking_planner_.getCl();
    Cr = parallel_parking_planner_.getCr();
    start_pose = parallel_parking_planner_.getStartPose();
    path_pose_array = parallel_parking_planner_.getPathPoseArray();
  } else if (status_.is_safe) {
    path_pose_array = util::convertToGeometryPoseArray(status_.path);
  }
  Cl_pub_->publish(Cl);
  Cr_pub_->publish(Cr);
  start_pose_pub_->publish(start_pose);
  path_pose_array_pub_->publish(path_pose_array);
}
}  // namespace behavior_path_planner
