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
#include "behavior_path_planner/path_shifter/path_shifter.hpp"
#include "behavior_path_planner/parallel_parking_planner/parallel_parking_planner.hpp"
#include "behavior_path_planner/occupancy_grid_map/occupancy_grid_map.hpp"
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
using tier4_autoware_utils::transformPose;
using tier4_autoware_utils::inverseTransformPose;
using tier4_autoware_utils::calcOffsetPose;
using tier4_autoware_utils::createQuaternionFromYaw;

using tier4_autoware_utils::createDefaultMarker;
using tier4_autoware_utils::createMarkerColor;
using tier4_autoware_utils::createMarkerScale;
using tier4_autoware_utils::createPoint;
using tier4_autoware_utils::calcDistance2d;

// namespace bg = boost::geometry;
// using Point = bg::model::d2::point_xy<double>;
// using Polygon = bg::model::polygon<Point>;

namespace behavior_path_planner
{
PullOverModule::PullOverModule(
  const std::string & name, rclcpp::Node & node, const PullOverParameters & parameters)
: SceneModuleInterface{name, node}, parameters_{parameters}
{
  approval_handler_.waitApproval();
  Cl_publisher_ = node.create_publisher<PoseStamped>("~/pull_over/debug/Cl", 1);
  Cr_publisher_ = node.create_publisher<PoseStamped>("~/pull_over/debug/Cr", 1);
  start_pose_publisher_ = node.create_publisher<PoseStamped>("~/pull_over/debug/start_pose", 1);
  goal_pose_publisher_ = node.create_publisher<PoseStamped>("~/pull_over/debug/goal_pose", 1);
  path_pose_array_pub_ = node.create_publisher<PoseArray>("~/pull_over/debug/path", 1);
  parking_area_pub_ = node.create_publisher<MarkerArray>("~/pull_over/parking_area", 1);

  using std::placeholders::_1;
  occupancy_grid_sub_ = node.create_subscription<nav_msgs::msg::OccupancyGrid>(
    "/perception/occupancy_grid_map/map", 1, std::bind(&PullOverModule::onOccupancyGrid, this, _1));
    // "~/input/occupancy_grid", 1, std::bind(&BehaviorModuleOutput::onOccupancyGrid, this, _1));
}

void PullOverModule::onOccupancyGrid(const OccupancyGrid::ConstSharedPtr msg)
{
  occupancy_grid_ = msg;
}

void PullOverModule::updateOccupancyGrid(){
  // need waiting for planner_data_
  CommonParam common_param;
  const float margin = 0;
  common_param.vehicle_shape.length = planner_data_->parameters.vehicle_length + margin;
  common_param.vehicle_shape.width = planner_data_->parameters.vehicle_width + margin;
  common_param.vehicle_shape.base2back = planner_data_->parameters.base_link2rear + margin / 2;
  common_param.theta_size = 360;
  common_param.obstacle_threshold = 98;

  occupancy_grid_map_.setParam(common_param);
  occupancy_grid_map_.setMap(*occupancy_grid_);
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
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  current_state_ = BT::NodeStatus::SUCCESS;
  updateOccupancyGrid();
  updatePullOverStatus();
  // Get arclength to start lane change
  const auto current_pose = planner_data_->self_pose->pose;
  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_.pull_over_lanes, current_pose);
  status_.start_distance = arclength_start.length;
  approval_handler_.waitApproval();
}

void PullOverModule::onExit()
{
  RCLCPP_DEBUG(getLogger(), "PULL_OVER onExit");
  approval_handler_.clearWaitApproval();
  current_state_ = BT::NodeStatus::IDLE;
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
}

bool PullOverModule::isExecutionRequested() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  lanelet::Lanelet closest_shoulder_lanelet;
  bool goal_is_in_shoulder_lane = false;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto current_lanes = getCurrentLanes();

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
  RCLCPP_ERROR(
    getLogger(), "(%s): goal_is_in_shoulder_lane:%d isLongEnough%d", __func__,
    goal_is_in_shoulder_lane, isLongEnough(current_lanes));
  // return goal_is_in_shoulder_lane && isLongEnough(current_lanes);
  return goal_is_in_shoulder_lane;
}

bool PullOverModule::isExecutionReady() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  if (current_state_ == BT::NodeStatus::RUNNING) {
    return true;
  }

  return true;

  // const auto current_lanes = getCurrentLanes();
  // const auto pull_over_lanes = getPullOverLanes(current_lanes);

  // Find pull_over path
  // bool found_valid_path, found_safe_path;
  // PullOverPath selected_path;
  // std::tie(found_valid_path, found_safe_path) =
  //   getSafePath(pull_over_lanes, check_distance_, selected_path);

  // return found_safe_path;
}

Marker PullOverModule::createParkingAreaMarker(const Pose start_pose, const Pose end_pose, const int32_t id){
  Marker marker = createDefaultMarker(
    "map", planner_data_->route_handler->getRouteHeader().stamp, "collision_polygons", id,
    visualization_msgs::msg::Marker::LINE_STRIP, createMarkerScale(0.1, 0.0, 0.0),
    createMarkerColor(0.0, 1.0, 0.0, 0.999));

  // create debug area maker
  auto p_left_front = calcOffsetPose(end_pose, planner_data_->parameters.base_link2front, planner_data_->parameters.vehicle_width / 2, 0).position;
  marker.points.push_back(createPoint(p_left_front.x, p_left_front.y, p_left_front.z));

  auto p_right_front = calcOffsetPose(end_pose, planner_data_->parameters.base_link2front, -planner_data_->parameters.vehicle_width / 2, 0).position;
  marker.points.push_back(createPoint(p_right_front.x, p_right_front.y, p_right_front.z));

  auto p_right_back = calcOffsetPose(start_pose, -planner_data_->parameters.base_link2rear, -planner_data_->parameters.vehicle_width / 2, 0).position;
  marker.points.push_back(createPoint(p_right_back.x, p_right_back.y, p_right_back.z));

  auto p_left_back = calcOffsetPose(start_pose, -planner_data_->parameters.base_link2rear, planner_data_->parameters.vehicle_width / 2, 0).position;
  marker.points.push_back(createPoint(p_left_back.x, p_left_back.y, p_left_back.z));
  marker.points.push_back(createPoint(p_left_front.x, p_left_front.y, p_left_front.z));

  return marker;
}

Pose PullOverModule::getRefinedGoal(){
  lanelet::ConstLanelet goal_lane;
  Pose goal_pose = planner_data_->route_handler->getGoalPose();

  lanelet::Lanelet closest_shoulder_lanelet;

  lanelet::utils::query::getClosestLanelet(
    planner_data_->route_handler->getShoulderLanelets(), planner_data_->self_pose->pose,
    &closest_shoulder_lanelet);

  Pose refined_goal_pose = lanelet::utils::getClosestCenterPose(closest_shoulder_lanelet, goal_pose.position);

  return refined_goal_pose;
}

Pose PullOverModule::researchGoal(){
  const float forward_serach_length = 20;
  const float backward_search_length = 20;
  const auto common_param = occupancy_grid_map_.getParam();

  auto goal_pose = getRefinedGoal();
  const Pose current_pose = planner_data_->self_pose->pose;

  // Ignore the areas behind the ego
  const Pose goal2current = inverseTransformPose(current_pose, goal_pose);
  float dx = std::max(-backward_search_length, static_cast<float>(goal2current.position.x));

  bool prev_is_collided = false;
  pull_over_areas_.clear();
  const Pose goal_pose_local = global2local(occupancy_grid_map_.getMap(), goal_pose);
  Pose start_pose = calcOffsetPose(goal_pose, dx, 0, 0);
  // Serch non collision areas around the goal
  while (true) {
    bool is_last_search = (dx >= forward_serach_length);
    Pose serach_pose = calcOffsetPose(goal_pose_local, dx, 0, 0);
    bool is_collided = occupancy_grid_map_.detectCollision(
      pose2index(occupancy_grid_map_.getMap(), serach_pose, common_param.theta_size), false);
    if (!prev_is_collided && is_collided || is_last_search) {
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
      pull_over_areas_.push_back(PullOverArea{
        start_pose, end_pose, calcDistance2d(calcAveragePose(start_pose, end_pose), goal_pose)});
    }
    if (is_last_search) break;

    if ((prev_is_collided && !is_collided)) {
      start_pose = calcOffsetPose(goal_pose, dx, 0, 0);
    }
    prev_is_collided = is_collided;
    dx += 0.05;
  }

  // If init goal does not collide, add it to candidates.
  goal_candidates_.clear();
  if (!occupancy_grid_map_.detectCollision(
        pose2index(occupancy_grid_map_.getMap(), goal_pose_local, common_param.theta_size), false)) {
    goal_candidates_.push_back(goal_pose);
  }

  // Sort center poses of areas with distance from init goal, and add them to candidates.
  std::sort(pull_over_areas_.begin(), pull_over_areas_.end());
  MarkerArray marker_array;
  // std::cerr << "areas size: " << pull_over_areas_.size() << std::endl;
  for (int32_t i = 0; i < pull_over_areas_.size(); i++) {
    marker_array.markers.push_back(createParkingAreaMarker(
      pull_over_areas_.at(i).start_pose, pull_over_areas_.at(i).end_pose, i));
    // std::cerr << "distance: " << pull_over_areas_.at(i).distance_from_goal << std::endl;
    goal_candidates_.push_back(pull_over_areas_.at(i).center_pose);
  }

  parking_area_pub_->publish(marker_array);

  return goal_candidates_.front();
}

BT::NodeStatus PullOverModule::updateState()
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  // RCLCPP_DEBUG(getLogger(), "PULL_OVER updateState");

  // if (hasFinishedPullOver()) {
  //   current_state_ = BT::NodeStatus::SUCCESS;
  //   return current_state_;
  // }
  current_state_ = BT::NodeStatus::RUNNING;
  return current_state_;
}

BehaviorModuleOutput PullOverModule::plan()
{
  // RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  const Pose goal_pose = researchGoal();
  goal_pose_.pose = goal_pose;
  goal_pose_.header =  planner_data_->route_handler->getRouteHeader();
  goal_pose_publisher_->publish(goal_pose_);

  updatePullOverStatus();

  PathWithLaneId path;
  if (status_.is_safe) {
    path = status_.pull_over_path.path;
  } else {
    parallel_parking_planner_.setParams(planner_data_);
    parallel_parking_planner_.plan(goal_pose);
    path = parallel_parking_planner_.getCurrentPath();

    Cl_publisher_->publish(parallel_parking_planner_.Cl_);
    Cr_publisher_->publish(parallel_parking_planner_.Cr_);
    path_pose_array_pub_->publish(parallel_parking_planner_.path_pose_array_);
    start_pose_publisher_->publish(parallel_parking_planner_.start_pose_);
  }

  // Generate drivable area
  // {
  //   lanelet::ConstLanelets lanes;
  //   lanes.push_back(current_lane);
  //   lanes.push_back(goal_lane);
  //   path.drivable_area = util::generateDrivableArea(
  //     lanes, common_parameters.drivable_area_resolution, common_parameters.vehicle_length,
  //     planner_data_);
  // }

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(path);
  // output.path = std::make_shared<PathWithLaneId>(status_.pull_over_path.path);

  return output;
}

PathWithLaneId PullOverModule::planCandidate() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  // Get lane change lanes
  const auto current_lanes = getCurrentLanes();
  const auto pull_over_lanes = getPullOverLanes(current_lanes);

  // Find lane change path
  bool found_valid_path, found_safe_path;
  PullOverPath selected_path;
  std::tie(found_valid_path, found_safe_path) =
    getSafePath(pull_over_lanes, check_distance_, selected_path);
  selected_path.path.header = planner_data_->route_handler->getRouteHeader();

  return selected_path.path;
}

BehaviorModuleOutput PullOverModule::planWaitingApproval()
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  BehaviorModuleOutput out;
  out.path = std::make_shared<PathWithLaneId>(getReferencePath());
  out.path_candidate = std::make_shared<PathWithLaneId>(planCandidate());
  return out;
}

void PullOverModule::setParameters(const PullOverParameters & parameters)
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  parameters_ = parameters;
}

void PullOverModule::updatePullOverStatus()
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  const auto & route_handler = planner_data_->route_handler;
  const auto common_parameters = planner_data_->parameters;

  const auto current_lanes = getCurrentLanes();
  status_.current_lanes = current_lanes;

  lanelet::ConstLanelet target_shoulder_lane;

  if (route_handler->getPullOverTarget(
        route_handler->getShoulderLanelets(), &target_shoulder_lane)) {
    route_handler->setPullOverGoalPose(
      target_shoulder_lane, common_parameters.vehicle_width, parameters_.margin_from_boundary);
  } else {
    RCLCPP_ERROR(getLogger(), "failed to get shoulder lane!!!");
  }

  // Get pull_over lanes
  const auto pull_over_lanes = getPullOverLanes(current_lanes);
  status_.pull_over_lanes = pull_over_lanes;

  // Find pull_over path
  bool found_valid_path, found_safe_path;
  PullOverPath selected_path;
  std::tie(found_valid_path, found_safe_path) =
    getSafePath(pull_over_lanes, check_distance_, selected_path);

  // Update status
  status_.is_safe = found_safe_path;
  status_.pull_over_path = selected_path;

  status_.lane_follow_lane_ids = util::getIds(current_lanes);
  status_.pull_over_lane_ids = util::getIds(pull_over_lanes);

  // Generate drivable area
  {
    lanelet::ConstLanelets lanes;
    lanes.insert(lanes.end(), current_lanes.begin(), current_lanes.end());
    lanes.insert(lanes.end(), pull_over_lanes.begin(), pull_over_lanes.end());

    const double resolution = common_parameters.drivable_area_resolution;
    status_.pull_over_path.path.drivable_area = util::generateDrivableArea(
      lanes, resolution, common_parameters.vehicle_length, planner_data_);
  }

  const auto current_pose = planner_data_->self_pose->pose;
  const auto arclength_start =
    lanelet::utils::getArcCoordinates(status_.pull_over_lanes, current_pose);
  status_.start_distance = arclength_start.length;

  status_.pull_over_path.path.header = planner_data_->route_handler->getRouteHeader();

  parallel_parking_planner_.clear();
  // とりあえずここでparral parkingのpathを生成する。
  // Pose goal_pose = researchGoal();
  // parallel_parking_planner_.setParams(planner_data_);
  // parallel_parking_planner_.plan(goal_pose);

  // Generate drivable area
  // {
  //   lanelet::ConstLanelets lanes;
  //   lanes.push_back(current_lane);
  //   lanes.push_back(goal_lane);
  //   path.drivable_area = util::generateDrivableArea(
  //     lanes, common_parameters.drivable_area_resolution, common_parameters.vehicle_length,
  //     planner_data_);
  // }
  // path.header = planner_data_->route_handler->getRouteHeader();

}

PathWithLaneId PullOverModule::getReferencePath() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  PathWithLaneId reference_path;

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto common_parameters = planner_data_->parameters;

  // Set header
  reference_path.header = route_handler->getRouteHeader();

  const auto current_lanes = getCurrentLanes();

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

lanelet::ConstLanelets PullOverModule::getCurrentLanes() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto common_parameters = planner_data_->parameters;

  lanelet::ConstLanelet current_lane;
  if (!route_handler->getClosestLaneletWithinRoute(current_pose, &current_lane)) {
    RCLCPP_ERROR(getLogger(), "failed to find closest lanelet within route!!!");
    return {};  // TODO(Horibe) what should be returned?
  }

  // For current_lanes with desired length
  return route_handler->getLaneletSequence(
    current_lane, current_pose, common_parameters.backward_path_length,
    common_parameters.forward_path_length);
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
  const lanelet::ConstLanelets & pull_over_lanes, const double check_distance,
  PullOverPath & safe_path) const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  std::vector<PullOverPath> valid_paths;

  const auto & route_handler = planner_data_->route_handler;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto current_twist = planner_data_->self_odometry->twist.twist;
  const auto common_parameters = planner_data_->parameters;

  const auto current_lanes = getCurrentLanes();
  if (!isLongEnough(current_lanes)) {
    return std::make_pair(false, false);
  }

  std::cerr <<  __func__ << " pull_over_lanes size " << pull_over_lanes.size() << std::endl;
  if (!pull_over_lanes.empty()) {
    // find candidate paths
    const auto pull_over_paths = pull_over_utils::getPullOverPaths(
      *route_handler, current_lanes, pull_over_lanes, current_pose, current_twist,
      common_parameters, parameters_);

    // get lanes used for detection
    lanelet::ConstLanelets check_lanes;
    std::cerr <<  __func__ << " pull_over_paths size " << pull_over_paths.size() << std::endl;
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
      current_pose, route_handler->isInGoalRouteSection(current_lanes.back()),
      route_handler->getGoalPose());

    std::cerr <<  __func__ << " valid_paths size " << valid_paths.size() << std::endl;
    if (valid_paths.empty()) {
      return std::make_pair(false, false);
    }
    // select safe path
    bool found_safe_path = pull_over_utils::selectSafePath(
      valid_paths, current_lanes, check_lanes, planner_data_->dynamic_object, current_pose,
      current_twist, common_parameters.vehicle_width, parameters_, occupancy_grid_map_, &safe_path);
    return std::make_pair(true, found_safe_path);
  }
  return std::make_pair(false, false);
}

bool PullOverModule::isLongEnough(const lanelet::ConstLanelets & lanelets) const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  PathShifter path_shifter;
  const double maximum_jerk = parameters_.maximum_lateral_jerk;
  const double pull_over_velocity = parameters_.minimum_pull_over_velocity;
  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
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
  // const double distance_to_goal = util::getSignedDistance(current_pose, goal_pose, lanelets);
  const double distance_to_goal = std::abs(util::getSignedDistance(current_pose, goal_pose, lanelets));

  return distance_to_goal > pull_over_total_distance_min;
}

bool PullOverModule::isSafe() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  return status_.is_safe;
}

bool PullOverModule::isNearEndOfLane() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  const auto current_pose = planner_data_->self_pose->pose;
  const auto common_parameters = planner_data_->parameters;
  const double threshold = 5 + common_parameters.minimum_pull_over_length;

  return std::max(0.0, util::getDistanceToEndOfLane(current_pose, status_.current_lanes)) <
         threshold;
}

bool PullOverModule::isCurrentSpeedLow() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  const auto current_twist = planner_data_->self_odometry->twist.twist;
  const double threshold_kmph = 10;
  return util::l2Norm(current_twist.linear) < threshold_kmph * 1000 / 3600;
}

bool PullOverModule::hasFinishedPullOver() const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
  // check ego car is close enough to goal pose
  const auto current_pose = planner_data_->self_pose->pose;
  const auto goal_pose = planner_data_->route_handler->getGoalPose();
  const auto arclength_current =
    lanelet::utils::getArcCoordinates(status_.pull_over_lanes, current_pose);
  const auto arclength_goal = lanelet::utils::getArcCoordinates(status_.pull_over_lanes, goal_pose);
  const bool car_is_on_goal =
    (arclength_goal.length - arclength_current.length < parameters_.pull_over_finish_judge_buffer)
      ? true
      : false;

  // check ego car is stopping
  const double ego_vel = util::l2Norm(planner_data_->self_odometry->twist.twist.linear);
  const bool car_is_stopping = (ego_vel == 0.0) ? true : false;

  lanelet::Lanelet closest_shoulder_lanelet;

  if (
    lanelet::utils::query::getClosestLanelet(
      planner_data_->route_handler->getShoulderLanelets(), planner_data_->self_pose->pose,
      &closest_shoulder_lanelet) &&
    car_is_on_goal && car_is_stopping) {
    const auto road_lanes = getCurrentLanes();

    // check if goal pose is in shoulder lane and distance is long enough for pull out
    // if (isLongEnough(road_lanes)) {
    //   return true;
    // }
  }

  return false;
}

std::pair<HazardLightsCommand, double> PullOverModule::getHazard(
  const lanelet::ConstLanelets & target_lanes, const Pose & current_pose, const Pose & goal_pose,
  const double & velocity, const double & hazard_on_threshold_dis,
  const double & hazard_on_threshold_vel, const double & base_link2front) const
{
  RCLCPP_ERROR(getLogger(), "(%s):", __func__);
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

}  // namespace behavior_path_planner
