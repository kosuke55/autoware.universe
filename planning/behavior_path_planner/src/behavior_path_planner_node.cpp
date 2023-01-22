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

#include "behavior_path_planner/behavior_path_planner_node.hpp"

#include "behavior_path_planner/debug_utilities.hpp"
#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/avoidance/manager.hpp"
#include "behavior_path_planner/scene_module/avoidance_by_lc/manager.hpp"
#include "behavior_path_planner/scene_module/lane_change/manager.hpp"
#include "behavior_path_planner/scene_module/pull_out/manager.hpp"
#include "behavior_path_planner/scene_module/pull_over/manager.hpp"
#include "behavior_path_planner/scene_module/side_shift/manager.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <motion_velocity_smoother/smoother/analytical_jerk_constrained_smoother/analytical_jerk_constrained_smoother.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{
rclcpp::SubscriptionOptions createSubscriptionOptions(rclcpp::Node * node_ptr)
{
  rclcpp::CallbackGroup::SharedPtr callback_group =
    node_ptr->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  auto sub_opt = rclcpp::SubscriptionOptions();
  sub_opt.callback_group = callback_group;

  return sub_opt;
}
}  // namespace

namespace behavior_path_planner
{
using tier4_planning_msgs::msg::PathChangeModuleId;
using vehicle_info_util::VehicleInfoUtil;

BehaviorPathPlannerNode::BehaviorPathPlannerNode(const rclcpp::NodeOptions & node_options)
: Node("behavior_path_planner", node_options)
{
  using std::placeholders::_1;
  using std::chrono_literals::operator""ms;
  using motion_velocity_smoother::AnalyticalJerkConstrainedSmoother;

  // data_manager
  {
    planner_data_ = std::make_shared<PlannerData>();
    planner_data_->parameters = getCommonParam();
    planner_data_->smoother = std::make_shared<AnalyticalJerkConstrainedSmoother>(*this);
  }

  // publisher
  path_publisher_ = create_publisher<PathWithLaneId>("~/output/path", 1);
  // path_candidate_publisher_ = create_publisher<Path>("~/output/path_candidate", 1);
  turn_signal_publisher_ =
    create_publisher<TurnIndicatorsCommand>("~/output/turn_indicators_cmd", 1);
  hazard_signal_publisher_ = create_publisher<HazardLightsCommand>("~/output/hazard_lights_cmd", 1);
  debug_drivable_area_publisher_ = create_publisher<OccupancyGrid>("~/debug/drivable_area", 1);
  // debug_path_publisher_ = create_publisher<Path>("~/debug/path_for_visualize", 1);
  // debug_avoidance_msg_array_publisher_ =
  //   create_publisher<AvoidanceDebugMsgArray>("~/debug/avoidance_debug_message_array", 1);

  if (planner_data_->parameters.visualize_drivable_area_for_shared_linestrings_lanelet) {
    debug_drivable_area_lanelets_publisher_ =
      create_publisher<MarkerArray>("~/drivable_area_boundary", 1);
  }

  // subscriber
  velocity_subscriber_ = create_subscription<Odometry>(
    "~/input/odometry", 1, std::bind(&BehaviorPathPlannerNode::onVelocity, this, _1),
    createSubscriptionOptions(this));
  acceleration_subscriber_ = create_subscription<AccelWithCovarianceStamped>(
    "~/input/accel", 1, std::bind(&BehaviorPathPlannerNode::onAcceleration, this, _1),
    createSubscriptionOptions(this));
  perception_subscriber_ = create_subscription<PredictedObjects>(
    "~/input/perception", 1, std::bind(&BehaviorPathPlannerNode::onPerception, this, _1),
    createSubscriptionOptions(this));
  // todo: change to ~/input
  occupancy_grid_subscriber_ = create_subscription<OccupancyGrid>(
    "/perception/occupancy_grid_map/map", 1,
    std::bind(&BehaviorPathPlannerNode::onOccupancyGrid, this, _1),
    createSubscriptionOptions(this));
  scenario_subscriber_ = create_subscription<Scenario>(
    "~/input/scenario", 1,
    [this](const Scenario::ConstSharedPtr msg) {
      current_scenario_ = std::make_shared<Scenario>(*msg);
    },
    createSubscriptionOptions(this));
  external_approval_subscriber_ = create_subscription<ApprovalMsg>(
    "~/input/external_approval", 1,
    std::bind(&BehaviorPathPlannerNode::onExternalApproval, this, _1),
    createSubscriptionOptions(this));
  force_approval_subscriber_ = create_subscription<PathChangeModule>(
    "~/input/force_approval", 1, std::bind(&BehaviorPathPlannerNode::onForceApproval, this, _1),
    createSubscriptionOptions(this));
  velocity_limit_subscriber_ = this->create_subscription<VelocityLimit>(
    "~/input/external_velocity_limit_mps", rclcpp::QoS{1}.transient_local(),
    std::bind(&BehaviorPathPlannerNode::onVelocityLimit, this, _1));
  lateral_offset_subscriber_ = this->create_subscription<LateralOffset>(
    "~/input/lateral_offset", 1, std::bind(&BehaviorPathPlannerNode::onLateralOffset, this, _1));
  autoware_state_subscriber_ = this->create_subscription<AutowareState>(
    "/autoware/state", 1, std::bind(&BehaviorPathPlannerNode::onState, this, _1));

  // route_handler
  auto qos_transient_local = rclcpp::QoS{1}.transient_local();
  vector_map_subscriber_ = create_subscription<HADMapBin>(
    "~/input/vector_map", qos_transient_local, std::bind(&BehaviorPathPlannerNode::onMap, this, _1),
    createSubscriptionOptions(this));
  route_subscriber_ = create_subscription<HADMapRoute>(
    "~/input/route", qos_transient_local, std::bind(&BehaviorPathPlannerNode::onRoute, this, _1),
    createSubscriptionOptions(this));

  // behavior tree manager
  {
    const std::string path_candidate_name_space = "/planning/path_candidate/";
    const std::string path_reference_name_space = "/planning/path_reference/";
    mutex_pm_.lock();

    const auto & p = planner_data_->parameters;
    planner_manager_ = std::make_shared<PlannerManager>(*this, p.verbose);

    if (p.launch_pull_over) {
      auto manager = std::make_shared<PullOverModuleManager>(
        this, "pull_over", 1, p.priority_pull_over, p.enable_simultaneous_execution_pull_over);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "pull_over", create_publisher<Path>(path_candidate_name_space + "pull_over", 1));
      path_reference_publishers_.emplace(
        "pull_over", create_publisher<Path>(path_reference_name_space + "pull_over", 1));
    }

    if (p.launch_avoidance_by_lc) {
      auto manager = std::make_shared<AvoidanceByLCModuleManager>(
        this, "avoidance_by_lc", 1, p.priority_avoidance_by_lc,
        p.enable_simultaneous_execution_avoidance_by_lc);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "avoidance_by_lc",
        create_publisher<Path>(path_candidate_name_space + "avoidance_by_lc", 1));
      path_reference_publishers_.emplace(
        "avoidance_by_lc",
        create_publisher<Path>(path_reference_name_space + "avoidance_by_lc", 1));
    }

    if (p.launch_avoidance) {
      auto manager = std::make_shared<AvoidanceModuleManager>(
        this, "avoidance", 1, p.priority_avoidance, p.enable_simultaneous_execution_avoidance);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "avoidance", create_publisher<Path>(path_candidate_name_space + "avoidance", 1));
      path_reference_publishers_.emplace(
        "avoidance", create_publisher<Path>(path_reference_name_space + "avoidance", 1));
    }

    if (p.launch_lane_change) {
      auto manager = std::make_shared<LaneChangeModuleManager>(
        this, "lane_change", 1, p.priority_lane_change,
        p.enable_simultaneous_execution_lane_change);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "lane_change", create_publisher<Path>(path_candidate_name_space + "lane_change", 1));
      path_reference_publishers_.emplace(
        "lane_change", create_publisher<Path>(path_reference_name_space + "lane_change", 1));
    }

    if (p.launch_side_shift) {
      auto manager = std::make_shared<SideShiftModuleManager>(
        this, "side_shift", 1, p.priority_side_shift, p.enable_simultaneous_execution_side_shift);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "side_shift", create_publisher<Path>(path_candidate_name_space + "side_shift", 1));
      path_reference_publishers_.emplace(
        "side_shift", create_publisher<Path>(path_reference_name_space + "side_shift", 1));
    }

    if (p.launch_pull_out) {
      auto manager = std::make_shared<PullOutModuleManager>(
        this, "pull_out", 1, p.priority_pull_out, p.enable_simultaneous_execution_pull_out);
      planner_manager_->registerSceneModuleManager(manager);
      path_candidate_publishers_.emplace(
        "pull_out", create_publisher<Path>(path_candidate_name_space + "pull_out", 1));
      path_reference_publishers_.emplace(
        "pull_out", create_publisher<Path>(path_reference_name_space + "pull_out", 1));
    }

    mutex_pm_.unlock();
  }

  // turn signal decider
  {
    double intersection_search_distance{declare_parameter("intersection_search_distance", 30.0)};
    turn_signal_decider_.setParameters(
      planner_data_->parameters.base_link2front, intersection_search_distance);
  }

  waitForData();

  // Start timer. This must be done after all data (e.g. vehicle pose, velocity) are ready.
  {
    const auto planning_hz = declare_parameter("planning_hz", 10.0);
    const auto period_ns = rclcpp::Rate(planning_hz).period();
    timer_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&BehaviorPathPlannerNode::run, this));
  }
}

BehaviorPathPlannerParameters BehaviorPathPlannerNode::getCommonParam()
{
  BehaviorPathPlannerParameters p{};

  // vehicle info
  const auto vehicle_info = VehicleInfoUtil(*this).getVehicleInfo();
  p.vehicle_info = vehicle_info;
  p.vehicle_width = vehicle_info.vehicle_width_m;
  p.vehicle_length = vehicle_info.vehicle_length_m;
  p.wheel_tread = vehicle_info.wheel_tread_m;
  p.wheel_base = vehicle_info.wheel_base_m;
  p.front_overhang = vehicle_info.front_overhang_m;
  p.rear_overhang = vehicle_info.rear_overhang_m;
  p.left_over_hang = vehicle_info.left_overhang_m;
  p.right_over_hang = vehicle_info.right_overhang_m;
  p.base_link2front = vehicle_info.max_longitudinal_offset_m;
  p.base_link2rear = p.rear_overhang;

  // NOTE: backward_path_length is used not only calculating path length but also calculating the
  // size of a drivable area.
  //       The drivable area has to cover not the base link but the vehicle itself. Therefore
  //       rear_overhang must be added to backward_path_length. In addition, because of the
  //       calculation of the drivable area in the obstacle_avoidance_planner package, the drivable
  //       area has to be a little longer than the backward_path_length parameter by adding
  //       min_backward_offset.
  constexpr double min_backward_offset = 1.0;
  const double backward_offset = vehicle_info.rear_overhang_m + min_backward_offset;

  p.launch_pull_over = declare_parameter<bool>("launch_pull_over");
  p.launch_pull_out = declare_parameter<bool>("launch_pull_out");
  p.launch_avoidance_by_lc = declare_parameter<bool>("launch_avoidance_by_lc");
  p.launch_avoidance = declare_parameter<bool>("launch_avoidance");
  p.launch_lane_change = declare_parameter<bool>("launch_lane_change");
  p.launch_side_shift = declare_parameter<bool>("launch_side_shift");

  p.priority_pull_over = declare_parameter<int>("priority_pull_over");
  p.priority_avoidance_by_lc = declare_parameter<int>("priority_avoidance_by_lc");
  p.priority_avoidance = declare_parameter<int>("priority_avoidance");
  p.priority_lane_change = declare_parameter<int>("priority_lane_change");
  p.priority_pull_out = declare_parameter<int>("priority_pull_out");
  p.priority_side_shift = declare_parameter<int>("priority_side_shift");

  p.enable_simultaneous_execution_avoidance_by_lc =
    declare_parameter<bool>("enable_simultaneous_execution_avoidance_by_lc");
  p.enable_simultaneous_execution_avoidance =
    declare_parameter<bool>("enable_simultaneous_execution_avoidance");
  p.enable_simultaneous_execution_lane_change =
    declare_parameter<bool>("enable_simultaneous_execution_lane_change");
  p.enable_simultaneous_execution_side_shift =
    declare_parameter<bool>("enable_simultaneous_execution_side_shift");
  p.enable_simultaneous_execution_pull_over =
    declare_parameter<bool>("enable_simultaneous_execution_pull_over");
  p.enable_simultaneous_execution_pull_out =
    declare_parameter<bool>("enable_simultaneous_execution_pull_out");

  p.verbose = declare_parameter<bool>("verbose");

  p.shift_request_time_limit = declare_parameter<double>("shift_request_time_limit");

  // ROS parameters
  p.backward_path_length_output =
    declare_parameter("backward_path_length_output", 5.0) + backward_offset;
  p.backward_path_length = declare_parameter("backward_path_length", 5.0) + backward_offset;
  p.forward_path_length = declare_parameter("forward_path_length", 100.0);
  p.backward_length_buffer_for_end_of_lane =
    declare_parameter("backward_length_buffer_for_end_of_lane", 5.0);
  p.backward_length_buffer_for_end_of_pull_over =
    declare_parameter("backward_length_buffer_for_end_of_pull_over", 5.0);
  p.backward_length_buffer_for_end_of_pull_out =
    declare_parameter("backward_length_buffer_for_end_of_pull_out", 5.0);
  p.minimum_lane_change_length = declare_parameter("minimum_lane_change_length", 8.0);
  p.minimum_pull_over_length = declare_parameter("minimum_pull_over_length", 15.0);
  p.drivable_area_resolution = declare_parameter<double>("drivable_area_resolution");
  p.drivable_lane_forward_length = declare_parameter<double>("drivable_lane_forward_length");
  p.drivable_lane_backward_length = declare_parameter<double>("drivable_lane_backward_length");
  p.drivable_lane_margin = declare_parameter<double>("drivable_lane_margin");
  p.drivable_area_margin = declare_parameter<double>("drivable_area_margin");
  p.refine_goal_search_radius_range = declare_parameter("refine_goal_search_radius_range", 7.5);
  p.turn_light_on_threshold_dis_lat = declare_parameter("turn_light_on_threshold_dis_lat", 0.3);
  p.turn_light_on_threshold_dis_long = declare_parameter("turn_light_on_threshold_dis_long", 10.0);
  p.turn_light_on_threshold_time = declare_parameter("turn_light_on_threshold_time", 3.0);
  p.visualize_drivable_area_for_shared_linestrings_lanelet =
    declare_parameter("visualize_drivable_area_for_shared_linestrings_lanelet", true);

  return p;
}

void BehaviorPathPlannerNode::waitForData()
{
  // wait until mandatory data is ready
  while (!current_scenario_ && rclcpp::ok()) {
    RCLCPP_INFO_SKIPFIRST_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for scenario topic");
    rclcpp::spin_some(this->get_node_base_interface());
    rclcpp::Rate(100).sleep();
  }

  mutex_pd_.lock();  // for planner_data_
  while (!planner_data_->route_handler->isHandlerReady() && rclcpp::ok()) {
    mutex_pd_.unlock();
    RCLCPP_INFO_SKIPFIRST_THROTTLE(
      get_logger(), *get_clock(), 5000, "waiting for route to be ready");
    rclcpp::spin_some(this->get_node_base_interface());
    rclcpp::Rate(100).sleep();
    mutex_pd_.lock();
  }

  while (rclcpp::ok()) {
    if (
      planner_data_->dynamic_object && planner_data_->self_odometry &&
      planner_data_->self_acceleration) {
      break;
    }

    mutex_pd_.unlock();
    RCLCPP_INFO_SKIPFIRST_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "waiting for vehicle pose, vehicle_velocity, and obstacles");
    rclcpp::spin_some(this->get_node_base_interface());
    rclcpp::Rate(100).sleep();
    mutex_pd_.lock();
  }

  self_pose_listener_.waitForFirstPose();
  planner_data_->self_pose = self_pose_listener_.getCurrentPose();
  mutex_pd_.unlock();
}

void BehaviorPathPlannerNode::run()
{
  RCLCPP_DEBUG(get_logger(), "----- BehaviorPathPlannerNode start -----");
  mutex_pm_.lock();  // for planner_manager_
  mutex_pd_.lock();  // for planner_data_

  // behavior_path_planner runs only in LANE DRIVING scenario.
  if (current_scenario_->current_scenario != Scenario::LANEDRIVING) {
    mutex_pm_.unlock();  // for planner_manager_
    mutex_pd_.unlock();  // for planner_data_
    return;
  }

  // update planner data
  planner_data_->self_pose = self_pose_listener_.getCurrentPose();

  const auto planner_data = planner_data_;
  // run behavior planner
  const auto output = planner_manager_->run(planner_data);

  // path handling
  const auto path = getPath(output, planner_data);
  const auto module_statuses = planner_manager_->getSceneModuleStatus();

  // update planner data
  planner_data_->prev_output_path = path;
  mutex_pd_.unlock();

  publishPathCandidate(planner_manager_->getSceneModuleManagers());
  publishPathReference(planner_manager_->getSceneModuleManagers());

  PathWithLaneId clipped_path;
  if (skipSmoothGoalConnection(module_statuses)) {
    clipped_path = *path;
  } else {
    clipped_path = modifyPathForSmoothGoalConnection(*path);
  }
  clipPathLength(clipped_path);
  if (!clipped_path.points.empty()) {
    path_publisher_->publish(clipped_path);
  } else {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "behavior path output is empty! Stop publish.");
  }

  // path_candidate_publisher_->publish(util::toPath(*path_candidate));

  // debug_path_publisher_->publish(util::toPath(path));
  debug_drivable_area_publisher_->publish(path->drivable_area);

  // for turn signal
  {
    TurnIndicatorsCommand turn_signal;
    HazardLightsCommand hazard_signal;
    if (output.turn_signal_info.hazard_signal.command == HazardLightsCommand::ENABLE) {
      turn_signal.command = TurnIndicatorsCommand::DISABLE;
      hazard_signal.command = output.turn_signal_info.hazard_signal.command;
    } else {
      turn_signal = turn_signal_decider_.getTurnSignal(
        *path, planner_data->self_pose->pose, *(planner_data->route_handler),
        output.turn_signal_info.turn_signal, output.turn_signal_info.signal_distance);
      hazard_signal.command = HazardLightsCommand::DISABLE;
    }
    turn_signal.stamp = get_clock()->now();
    hazard_signal.stamp = get_clock()->now();
    turn_signal_publisher_->publish(turn_signal);
    hazard_signal_publisher_->publish(hazard_signal);
  }

  // for debug
  // debug_avoidance_msg_array_publisher_->publish(bt_manager_->getAvoidanceDebugMsgArray());
  // publishDebugMarker(bt_manager_->getDebugMarkers());

  if (planner_data->parameters.visualize_drivable_area_for_shared_linestrings_lanelet) {
    const auto drivable_area_lines = marker_utils::createFurthestLineStringMarkerArray(
      util::getDrivableAreaForAllSharedLinestringLanelets(planner_data));
    debug_drivable_area_lanelets_publisher_->publish(drivable_area_lines);
  }

  planner_manager_->print();
  planner_manager_->publishDebugMarker();

  mutex_pm_.unlock();
  RCLCPP_DEBUG(get_logger(), "----- behavior path planner end -----\n\n");
}

PathWithLaneId::SharedPtr BehaviorPathPlannerNode::getPath(
  const BehaviorModuleOutput & bt_output, const std::shared_ptr<PlannerData> planner_data)
{
  // TODO(Horibe) do some error handling when path is not available.

  auto path = bt_output.path ? bt_output.path : planner_data->prev_output_path;
  path->header = planner_data->route_handler->getRouteHeader();
  path->header.stamp = this->now();

  const auto & forward = planner_data_->parameters.forward_path_length;
  const auto & backward = planner_data_->parameters.backward_path_length_output;
  const auto & ego_pose = planner_data_->self_pose->pose;
  util::clipPathLength(*path, ego_pose, forward, backward);

  RCLCPP_DEBUG(
    get_logger(), "BehaviorTreeManager: output is %s.", bt_output.path ? "FOUND" : "NOT FOUND");
  return path;
}

bool BehaviorPathPlannerNode::skipSmoothGoalConnection(
  const std::vector<std::shared_ptr<SceneModuleStatus>> & statuses) const
{
  const auto target_module = "pull_over";

  for (auto & status : statuses) {
    if (status->is_waiting_approval || status->status == ModuleStatus::RUNNING) {
      if (target_module == status->module_name) {
        return true;
      }
    }
  }
  return false;
}

void BehaviorPathPlannerNode::onVelocity(const Odometry::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->self_odometry = msg;
}
void BehaviorPathPlannerNode::onVelocityLimit(const VelocityLimit::ConstSharedPtr msg)
{
  planner_data_->velocity_limit = *msg;
}
void BehaviorPathPlannerNode::onAcceleration(const AccelWithCovarianceStamped::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->self_acceleration = msg;
}
void BehaviorPathPlannerNode::onPerception(const PredictedObjects::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->dynamic_object = msg;
}
void BehaviorPathPlannerNode::onOccupancyGrid(const OccupancyGrid::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->occupancy_grid = msg;
}
void BehaviorPathPlannerNode::onExternalApproval(const ApprovalMsg::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->approval.is_approved.data = msg->approval;
  // TODO(wep21): Replace msg stamp after {stamp: now} is implemented in ros2 topic pub
  planner_data_->approval.is_approved.stamp = this->now();
}
void BehaviorPathPlannerNode::onForceApproval(const PathChangeModule::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  auto getModuleName = [](PathChangeModuleId module) {
    if (module.type == PathChangeModuleId::FORCE_LANE_CHANGE) {
      return "ForceLaneChange";
    } else {
      return "NONE";
    }
  };
  planner_data_->approval.is_force_approved.module_name = getModuleName(msg->module);
  planner_data_->approval.is_force_approved.stamp = msg->header.stamp;
}
void BehaviorPathPlannerNode::onMap(const HADMapBin::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);
  planner_data_->route_handler->setMap(*msg);
}
void BehaviorPathPlannerNode::onRoute(const HADMapRoute::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock_pd(mutex_pd_);
  std::lock_guard<std::mutex> lock_pm(mutex_pm_);
  const bool is_first_time = !(planner_data_->route_handler->isHandlerReady());

  planner_data_->route_handler->setRoute(*msg);

  // Reset behavior tree when new route is received,
  // so that the each modules do not have to care about the "route jump".
  if (!is_first_time) {
    RCLCPP_DEBUG(get_logger(), "new route is received. reset behavior tree.");
    planner_manager_->reset();
  }
}

void BehaviorPathPlannerNode::onLateralOffset(const LateralOffset::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_pd_);

  if (!planner_data_->lateral_offset) {
    planner_data_->lateral_offset = msg;
    return;
  }

  const auto & new_offset = msg->lateral_offset;
  const auto & old_offset = planner_data_->lateral_offset->lateral_offset;

  // offset is not changed.
  if (std::abs(old_offset - new_offset) < 1e-4) {
    return;
  }

  const auto latest_update_time = planner_data_->lateral_offset->stamp;

  // new offset is requested.
  if (isReadyForNextRequest(
        latest_update_time, planner_data_->parameters.shift_request_time_limit)) {
    planner_data_->lateral_offset = msg;
  }
}

bool BehaviorPathPlannerNode::isReadyForNextRequest(
  const rclcpp::Time & latest_update_time, const double & min_request_time_sec,
  bool override_requests) const
{
  const auto interval_from_last_request_sec = this->now() - latest_update_time;

  if (interval_from_last_request_sec.seconds() >= min_request_time_sec && !override_requests) {
    return true;
  }

  return false;
}

void BehaviorPathPlannerNode::onState(const AutowareState::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock_pd(mutex_pd_);
  std::lock_guard<std::mutex> lock_pm(mutex_pm_);

  if (!planner_data_->state) {
    planner_data_->state = msg;
    return;
  }

  if (planner_data_->state->state != AutowareState::DRIVING) {
    planner_data_->state = msg;
    return;
  }

  if (msg->state == AutowareState::DRIVING) {
    planner_data_->state = msg;
    return;
  }

  planner_manager_->reset();
  planner_data_->state = msg;
}

void BehaviorPathPlannerNode::clipPathLength(PathWithLaneId & path) const
{
  const auto ego_pose = planner_data_->self_pose->pose;
  const double forward = planner_data_->parameters.forward_path_length;
  const double backward = planner_data_->parameters.backward_path_length;

  util::clipPathLength(path, ego_pose, forward, backward);
}

PathWithLaneId BehaviorPathPlannerNode::modifyPathForSmoothGoalConnection(
  const PathWithLaneId & path) const
{
  const auto goal = planner_data_->route_handler->getGoalPose();
  const auto is_approved = planner_data_->approval.is_approved.data;
  auto goal_lane_id = planner_data_->route_handler->getGoalLaneId();

  Pose refined_goal{};
  {
    lanelet::ConstLanelet goal_lanelet;
    lanelet::ConstLanelet pull_over_lane;
    geometry_msgs::msg::Pose pull_over_goal;
    if (
      is_approved && planner_data_->route_handler->getPullOverTarget(
                       planner_data_->route_handler->getShoulderLanelets(), &pull_over_lane)) {
      refined_goal = planner_data_->route_handler->getPullOverGoalPose();
      goal_lane_id = pull_over_lane.id();
    } else if (planner_data_->route_handler->getGoalLanelet(&goal_lanelet)) {
      refined_goal = util::refineGoal(goal, goal_lanelet);
    } else {
      refined_goal = goal;
    }
  }

  auto refined_path = util::refinePathForGoal(
    planner_data_->parameters.refine_goal_search_radius_range, M_PI * 0.5, path, refined_goal,
    goal_lane_id);
  refined_path.header.frame_id = "map";
  refined_path.header.stamp = this->now();

  return refined_path;
}

void BehaviorPathPlannerNode::publishPathCandidate(
  const std::vector<std::shared_ptr<SceneModuleManagerInterface>> & managers)
{
  for (auto & manager : managers) {
    if (path_candidate_publishers_.count(manager->getModuleName()) == 0) {
      continue;
    }

    if (manager->getSceneModules().empty()) {
      path_candidate_publishers_.at(manager->getModuleName())
        ->publish(convertToPath(nullptr, false));
      continue;
    }

    for (auto & module : manager->getSceneModules()) {
      path_candidate_publishers_.at(module->name())
        ->publish(convertToPath(module->getPathCandidate(), module->isExecutionReady()));
    }
  }
}

void BehaviorPathPlannerNode::publishPathReference(
  const std::vector<std::shared_ptr<SceneModuleManagerInterface>> & managers)
{
  for (auto & manager : managers) {
    if (path_reference_publishers_.count(manager->getModuleName()) == 0) {
      continue;
    }

    if (manager->getSceneModules().empty()) {
      path_reference_publishers_.at(manager->getModuleName())
        ->publish(convertToPath(nullptr, false));
      continue;
    }

    for (auto & module : manager->getSceneModules()) {
      path_reference_publishers_.at(module->name())
        ->publish(convertToPath(module->getPathReference(), true));
    }
  }
}

Path BehaviorPathPlannerNode::convertToPath(
  const std::shared_ptr<PathWithLaneId> & path_candidate_ptr, const bool is_ready)
{
  Path output;
  output.header = planner_data_->route_handler->getRouteHeader();
  output.header.stamp = this->now();

  if (!path_candidate_ptr) {
    return output;
  }

  output = util::toPath(*path_candidate_ptr);
  // header is replaced by the input one, so it is substituted again
  output.header = planner_data_->route_handler->getRouteHeader();
  output.header.stamp = this->now();

  if (!is_ready) {
    for (auto & point : output.points) {
      point.longitudinal_velocity_mps = 0.0;
    }
  }

  return output;
}
}  // namespace behavior_path_planner

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(behavior_path_planner::BehaviorPathPlannerNode)
