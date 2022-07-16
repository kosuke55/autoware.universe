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

#include "behavior_path_planner/scene_module/pull_out/pull_out_module.hpp"

#include "behavior_path_planner/behavior_path_planner_node.hpp"
#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/avoidance/debug.hpp"
#include "behavior_path_planner/scene_module/pull_out/util.hpp"
#include "behavior_path_planner/scene_module/utils/path_shifter.hpp"
#include "behavior_path_planner/util/create_vehicle_footprint.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <vehicle_info_util/vehicle_info.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using motion_utils::calcLongitudinalOffsetPose;
using tier4_autoware_utils::calcOffsetPose;
namespace behavior_path_planner
{
PullOutModule::PullOutModule(
  const std::string & name, rclcpp::Node & node, const PullOutParameters & parameters)
: SceneModuleInterface{name, node}, parameters_{parameters}
{
  rtc_interface_ptr_ = std::make_shared<RTCInterface>(&node, "pull_out");

  backed_pose_pub_ = node.create_publisher<PoseStamped>("~/pull_out/debug/backed_pose", 1);

  lane_departure_checker_ = std::make_shared<LaneDepartureChecker>();
  lane_departure_checker_->setVehicleInfo(
    vehicle_info_util::VehicleInfoUtil(node).getVehicleInfo());

  pull_out_planner_ = std::make_shared<ShiftPullOut>(node, parameters, lane_departure_checker_);
}

BehaviorModuleOutput PullOutModule::run()
{
  clearWaitingApproval();
  current_state_ = BT::NodeStatus::RUNNING;
  return plan();
}

void PullOutModule::onEntry()
{
  RCLCPP_DEBUG(getLogger(), "PULL_OUT onEntry");
  current_state_ = BT::NodeStatus::SUCCESS;
  updatePullOutStatus();

  // Get arclength to start lane change
  const auto current_pose = planner_data_->self_pose->pose;
  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_.pull_out_lanes, current_pose);
  status_.start_distance = arclength_start.length;
}

void PullOutModule::onExit()
{
  clearWaitingApproval();
  removeRTCStatus();
  current_state_ = BT::NodeStatus::IDLE;
  RCLCPP_DEBUG(getLogger(), "PULL_OUT onExit");
}

bool PullOutModule::isExecutionRequested() const
{
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  const bool car_is_stopping =
    (util::l2Norm(planner_data_->self_odometry->twist.twist.linear) <= 1.5) ? true : false;

  lanelet::Lanelet closest_shoulder_lanelet;

  if (
    lanelet::utils::query::getClosestLanelet(
      planner_data_->route_handler->getShoulderLanelets(), planner_data_->self_pose->pose,
      &closest_shoulder_lanelet) &&
    car_is_stopping) {
    // Create vehicle footprint
    const auto vehicle_info = getVehicleInfo(planner_data_->parameters);
    const auto local_vehicle_footprint = createVehicleFootprint(vehicle_info);
    const auto vehicle_footprint = transformVector(
      local_vehicle_footprint,
      tier4_autoware_utils::pose2transform(planner_data_->self_pose->pose));
    const auto road_lanes = getCurrentLanes();

    // check if goal pose is in shoulder lane and distance is long enough for pull out
    if (isInLane(closest_shoulder_lanelet, vehicle_footprint) && isLongEnough(road_lanes)) {
      return true;
    }
  }

  return false;
}

bool PullOutModule::isExecutionReady() const { return true; }

// this runs only when RUNNING
BT::NodeStatus PullOutModule::updateState()
{
  RCLCPP_DEBUG(getLogger(), "PULL_OUT updateState");

  if (hasFinishedPullOut()) {
    current_state_ = BT::NodeStatus::SUCCESS;
    return current_state_;
  }

  checkBackFinished();

  return current_state_;
}

BehaviorModuleOutput PullOutModule::plan()
{
  constexpr double RESAMPLE_INTERVAL = 1.0;

  PathWithLaneId path;
  if (!status_.back_finished) {
    path = status_.backward_path;
  } else {
    path = util::resamplePathWithSpline(status_.pull_out_path.path, RESAMPLE_INTERVAL);
    status_.back_finished = true;
  }

  if (status_.is_retreat_path_valid && status_.back_finished) {
    path = util::resamplePathWithSpline(status_.retreat_path.path, RESAMPLE_INTERVAL);
  }

  path.drivable_area = status_.pull_out_path.path.drivable_area;

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(path);
  output.turn_signal_info = calcTurnSignalInfo(status_.pull_out_path.shift_point);

  return output;
}

CandidateOutput PullOutModule::planCandidate() const { return CandidateOutput{}; }

PathWithLaneId PullOutModule::getFullPath() const
{
  const auto pull_out_path = pull_out_planner_->getFullPath();

  if (status_.back_finished) {
    // not need backward path or finish it
    return pull_out_path;
  } else {
    // concat back_path and pull_out_path and
    auto full_path = status_.backward_path;
    full_path.points.insert(
      full_path.points.end(), pull_out_path.points.begin(), pull_out_path.points.end());
    return full_path;
  }
}

BehaviorModuleOutput PullOutModule::planWaitingApproval()
{
  BehaviorModuleOutput out;
  const auto common_parameters = planner_data_->parameters;
  const auto current_lanes = getCurrentLanes();
  const auto shoulder_lanes = pull_out_utils::getPullOutLanes(current_lanes, planner_data_);

  updatePullOutStatus();

  PathWithLaneId candidate_path;
  // Generate drivable area
  {
    candidate_path = status_.back_finished ? status_.pull_out_path.path : status_.backward_path;

    lanelet::ConstLanelets lanes;
    lanes.insert(lanes.end(), current_lanes.begin(), current_lanes.end());
    lanes.insert(lanes.end(), shoulder_lanes.begin(), shoulder_lanes.end());
    const double resolution = common_parameters.drivable_area_resolution;
    candidate_path.drivable_area = util::generateDrivableArea(
      lanes, resolution, common_parameters.vehicle_length, planner_data_);
    updateRTCStatus(0);
  }
  for (size_t i = 1; i < candidate_path.points.size(); i++) {
    candidate_path.points.at(i).point.longitudinal_velocity_mps = 0.0;
  }

  PathWithLaneId stop_path = candidate_path;
  for (auto & p : stop_path.points) {
    p.point.longitudinal_velocity_mps = 0.0;
  }
  out.path = std::make_shared<PathWithLaneId>(stop_path);

  out.path_candidate = std::make_shared<PathWithLaneId>(candidate_path);

  waitApproval();

  return out;
}

void PullOutModule::setParameters(const PullOutParameters & parameters)
{
  parameters_ = parameters;
}

void PullOutModule::updatePullOutStatus()
{

  const auto & route_handler = planner_data_->route_handler;
  const auto common_parameters = planner_data_->parameters;

  const auto current_lanes = getCurrentLanes();
  status_.current_lanes = current_lanes;

  // Get pull_out lanes
  const auto pull_out_lanes = pull_out_utils::getPullOutLanes(current_lanes, planner_data_);
  status_.pull_out_lanes = pull_out_lanes;

  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();

  // plan pull_out path with each backed pose
  PullOutPath selected_path;
  bool found_safe_path = false;
  backed_pose_candidates_ = searchBackedPoses();  // the first backed_pose is current_pose
  for (auto const & backed_pose : backed_pose_candidates_) {
    pull_out_planner_->setPlannerData(planner_data_);
    const auto pull_out_path = pull_out_planner_->plan(backed_pose, goal_pose);
    if (pull_out_path) {  // found safe path
      found_safe_path = true;
      selected_path.path = *pull_out_path;
      status_.backed_pose = backed_pose;

      // for debug
      backed_pose_.pose = backed_pose;
      backed_pose_.header = planner_data_->route_handler->getRouteHeader();
      backed_pose_pub_->publish(backed_pose_);
      break;
    }

    // backed_pose in not current_pose, so need back.
    status_.back_finished = false;
  }
  if (!found_safe_path) return;

  checkBackFinished();
  if (!status_.back_finished) {
    status_.backward_path = pull_out_utils::getBackwardPath(
      *route_handler, pull_out_lanes, current_pose, status_.backed_pose);
  }

  // Update status
  status_.is_safe = found_safe_path;
  status_.pull_out_path = selected_path;

  status_.lane_follow_lane_ids = util::getIds(current_lanes);
  status_.pull_out_lane_ids = util::getIds(pull_out_lanes);

  // Generate drivable area
  {
    lanelet::ConstLanelets lanes;
    lanes.insert(lanes.end(), current_lanes.begin(), current_lanes.end());
    lanes.insert(lanes.end(), pull_out_lanes.begin(), pull_out_lanes.end());

    const double resolution = common_parameters.drivable_area_resolution;
    status_.pull_out_path.path.drivable_area = util::generateDrivableArea(
      lanes, resolution, common_parameters.vehicle_length, planner_data_);
  }

  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_.pull_out_lanes, current_pose);
  status_.start_distance = arclength_start.length;

  status_.pull_out_path.path.header = planner_data_->route_handler->getRouteHeader();
}

PathWithLaneId PullOutModule::getReferencePath() const
{
  PathWithLaneId reference_path;

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto common_parameters = planner_data_->parameters;

  // Set header
  reference_path.header = route_handler->getRouteHeader();

  const auto current_lanes = getCurrentLanes();
  const auto pull_out_lanes = pull_out_utils::getPullOutLanes(current_lanes, planner_data_);

  if (current_lanes.empty()) {
    return reference_path;
  }

  reference_path = util::getCenterLinePath(
    *route_handler, pull_out_lanes, current_pose, common_parameters.backward_path_length,
    common_parameters.forward_path_length, common_parameters);

  reference_path = util::setDecelerationVelocity(
    *route_handler, reference_path, current_lanes, parameters_.after_pull_out_straight_distance,
    common_parameters.minimum_pull_out_length, parameters_.before_pull_out_straight_distance,
    parameters_.deceleration_interval, goal_pose);

  reference_path.drivable_area = util::generateDrivableArea(
    pull_out_lanes, common_parameters.drivable_area_resolution, common_parameters.vehicle_length,
    planner_data_);
  return reference_path;
}

lanelet::ConstLanelets PullOutModule::getCurrentLanes() const
{
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto common_parameters = planner_data_->parameters;

  lanelet::ConstLanelet current_lane;
  if (!route_handler->getClosestLaneletWithinRoute(current_pose, &current_lane)) {
    RCLCPP_ERROR_THROTTLE(
      getLogger(), *clock_, 5000, "failed to find closest lanelet within route!!!");
    return {};  // TODO(Horibe) what should be returned?
  }

  // For current_lanes with desired length
  return route_handler->getLaneletSequence(
    current_lane, current_pose, common_parameters.backward_path_length,
    common_parameters.forward_path_length);
}

std::vector<Pose> PullOutModule::searchBackedPoses()
{
  const auto current_pose = planner_data_->self_pose->pose;
  const double search_resolution = 1.0;
  const double maximum_back_distance = 15;
  const double collision_check_margin = 1.0;

  const auto current_lanes = getCurrentLanes();
  const auto pull_out_lanes = pull_out_utils::getPullOutLanes(current_lanes, planner_data_);

  // get backward shoulder path
  const auto arc_position_pose = lanelet::utils::getArcCoordinates(pull_out_lanes, current_pose);
  auto backward_shoulder_path = planner_data_->route_handler->getCenterLinePath(
    pull_out_lanes, arc_position_pose.length - pull_out_lane_length_,
    arc_position_pose.length + pull_out_lane_length_);

  // lateral shift to current_pose
  const double distance_from_center_line = arc_position_pose.distance;
  for (auto & p : backward_shoulder_path.points) {
    p.point.pose = calcOffsetPose(p.point.pose, 0, distance_from_center_line, 0);
  }

  // check collision between footprint and onject at the backed pose
  const auto vehicle_info = getVehicleInfo(planner_data_->parameters);
  const auto local_vehicle_footprint = createVehicleFootprint(vehicle_info);
  std::vector<Pose> backed_poses;
  for (double back_distance = 0.0; back_distance <= maximum_back_distance;
       back_distance += search_resolution) {
    const auto backed_pose = calcLongitudinalOffsetPose(
      backward_shoulder_path.points, current_pose.position, -back_distance);
    if (!backed_pose) continue;
    if (util::checkCollisionBetweenFootprintAndObjects(
          local_vehicle_footprint, *backed_pose, *(planner_data_->dynamic_object),
          collision_check_margin)) {
      break;  // poses behind this has a collision, so break.
    };
    backed_poses.push_back(*backed_pose);
  }
  return backed_poses;
}

bool PullOutModule::isInLane(
  const lanelet::ConstLanelet & candidate_lanelet,
  const tier4_autoware_utils::LinearRing2d & vehicle_footprint) const
{
  for (const auto & point : vehicle_footprint) {
    if (boost::geometry::within(point, candidate_lanelet.polygon2d().basicPolygon())) {
      return true;
    }
  }

  return false;
}

bool PullOutModule::isLongEnough(const lanelet::ConstLanelets & lanelets) const
{
  PathShifter path_shifter;
  const double maximum_jerk = parameters_.maximum_lateral_jerk;
  const double pull_out_velocity = parameters_.minimum_pull_out_velocity;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const double distance_before_pull_out = parameters_.before_pull_out_straight_distance;
  const double distance_after_pull_out = parameters_.after_pull_out_straight_distance;
  const double distance_to_road_center =
    lanelet::utils::getArcCoordinates(lanelets, planner_data_->self_pose->pose).distance;

  // calculate minimum pull_out distance at pull_out velocity,
  // maximum jerk and calculated side offset
  const double pull_out_distance_min = path_shifter.calcLongitudinalDistFromJerk(
    abs(distance_to_road_center), maximum_jerk, pull_out_velocity);
  const double pull_out_total_distance_min =
    distance_before_pull_out + pull_out_distance_min + distance_after_pull_out;
  const double distance_to_goal_on_road_lane =
    util::getSignedDistance(current_pose, goal_pose, lanelets);

  return distance_to_goal_on_road_lane > pull_out_total_distance_min;
}

bool PullOutModule::isSafe() const { return status_.is_safe; }

bool PullOutModule::isNearEndOfLane() const
{
  const auto current_pose = planner_data_->self_pose->pose;
  const auto common_parameters = planner_data_->parameters;
  const double threshold = 5 + common_parameters.minimum_pull_over_length;

  return std::max(0.0, util::getDistanceToEndOfLane(current_pose, status_.current_lanes)) <
         threshold;
}

bool PullOutModule::isCurrentSpeedLow() const
{
  const auto current_twist = planner_data_->self_odometry->twist.twist;
  const double threshold_kmph = 10;
  return util::l2Norm(current_twist.linear) < threshold_kmph * 1000 / 3600;
}

bool PullOutModule::hasFinishedPullOut() const
{
  // check ego car is close enough to goal pose
  const auto current_pose = planner_data_->self_pose->pose;
  const auto arclength_current =
    lanelet::utils::getArcCoordinates(status_.current_lanes, current_pose);
  const auto arclength_shift_end =
    lanelet::utils::getArcCoordinates(status_.current_lanes, status_.pull_out_path.shift_point.end);
  const bool car_is_on_goal = (arclength_shift_end.length - arclength_current.length <
                               parameters_.pull_out_finish_judge_buffer)
                                ? true
                                : false;

  return car_is_on_goal;
}

void PullOutModule::checkBackFinished()
{
  // check ego car is close enough to goal pose
  const auto current_pose = planner_data_->self_pose->pose;
  const auto backed_pose = status_.backed_pose;
  const auto distance = tier4_autoware_utils::calcDistance2d(current_pose, backed_pose);

  const bool is_near_backed_pose = distance < 1.0;
  const double ego_vel = util::l2Norm(planner_data_->self_odometry->twist.twist.linear);
  const bool is_stopped = ego_vel < 0.01;

  if (!status_.back_finished && is_near_backed_pose && is_stopped) {
    std::cerr << "back finished" << std::endl;
    status_.back_finished = true;

    // requst pull_out approval
    waitApproval();
    removeRTCStatus();
    uuid_ = generateUUID();
    updateRTCStatus(0.0);
    current_state_ = BT::NodeStatus::SUCCESS;  // for breaking loop
  }
}

TurnSignalInfo PullOutModule::calcTurnSignalInfo(const ShiftPoint & shift_point) const
{
  TurnSignalInfo turn_signal;

  if (!status_.back_finished) {
    turn_signal.hazard_signal.command = HazardLightsCommand::ENABLE;
    turn_signal.signal_distance =
      tier4_autoware_utils::calcDistance2d(status_.backed_pose, planner_data_->self_pose->pose);
    return turn_signal;
  }

  const auto current_lanes = getCurrentLanes();
  const auto pull_out_lanes = pull_out_utils::getPullOutLanes(current_lanes, planner_data_);
  const double turn_signal_on_threshold = 30;
  const double turn_signal_off_threshold = -3;
  const double turn_hazard_on_threshold = 3;

  // calculate distance to pull_out start on current lanes
  double distance_to_pull_out_start;
  {
    const auto pull_out_start = shift_point.start;
    const auto arc_position_pull_out_start =
      lanelet::utils::getArcCoordinates(current_lanes, pull_out_start);
    const auto arc_position_current_pose =
      lanelet::utils::getArcCoordinates(current_lanes, planner_data_->self_pose->pose);
    distance_to_pull_out_start =
      arc_position_pull_out_start.length - arc_position_current_pose.length;
  }

  // calculate distance to pull_out end on target lanes
  double distance_to_pull_out_end;
  {
    const auto pull_out_end = shift_point.end;
    const auto arc_position_pull_out_end =
      lanelet::utils::getArcCoordinates(pull_out_lanes, pull_out_end);
    const auto arc_position_current_pose =
      lanelet::utils::getArcCoordinates(pull_out_lanes, planner_data_->self_pose->pose);
    distance_to_pull_out_end = arc_position_pull_out_end.length - arc_position_current_pose.length;
  }

  // calculate distance to pull_out start on target lanes
  double distance_to_target_pose;
  {
    const auto arc_position_target_pose = lanelet::utils::getArcCoordinates(
      pull_out_lanes, planner_data_->route_handler->getGoalPose());
    const auto arc_position_current_pose =
      lanelet::utils::getArcCoordinates(pull_out_lanes, planner_data_->self_pose->pose);
    distance_to_target_pose = arc_position_target_pose.length - arc_position_current_pose.length;
  }

  if (distance_to_pull_out_start < turn_signal_on_threshold) {
    turn_signal.turn_signal.command = TurnIndicatorsCommand::ENABLE_RIGHT;
    if (distance_to_pull_out_end < turn_signal_off_threshold) {
      turn_signal.turn_signal.command = TurnIndicatorsCommand::DISABLE;
      if (distance_to_target_pose < turn_hazard_on_threshold) {
        turn_signal.hazard_signal.command = HazardLightsCommand::ENABLE;
      }
    }
  }
  turn_signal.signal_distance = distance_to_pull_out_end;

  return turn_signal;
}

vehicle_info_util::VehicleInfo PullOutModule::getVehicleInfo(
  const BehaviorPathPlannerParameters & parameters) const
{
  vehicle_info_util::VehicleInfo vehicle_info;
  vehicle_info.front_overhang_m = parameters.front_overhang;
  vehicle_info.wheel_base_m = parameters.wheel_base;
  vehicle_info.rear_overhang_m = parameters.rear_overhang;
  vehicle_info.wheel_tread_m = parameters.wheel_tread;
  vehicle_info.left_overhang_m = parameters.left_over_hang;
  vehicle_info.right_overhang_m = parameters.right_over_hang;
  return vehicle_info;
}

}  // namespace behavior_path_planner
