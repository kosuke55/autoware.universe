// Copyright 2021 The Autoware Foundation
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

#include "trajectory_follower/mpc_lateral_controller.hpp"

#include "tf2_ros/create_timer_ros.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware
{
namespace motion
{
namespace control
{
namespace trajectory_follower
{
namespace
{
using namespace std::literals::chrono_literals;

template <typename T>
void update_param(
  const std::vector<rclcpp::Parameter> & parameters, const std::string & name, T & value)
{
  auto it = std::find_if(
    parameters.cbegin(), parameters.cend(),
    [&name](const rclcpp::Parameter & parameter) { return parameter.get_name() == name; });
  if (it != parameters.cend()) {
    value = static_cast<T>(it->template get_value<T>());
  }
}
}  // namespace

MpcLateralController::MpcLateralController(rclcpp::Node * node) : node_{node}
{
  using std::placeholders::_1;

  m_mpc.m_ctrl_period = node_->get_parameter("ctrl_period").as_double();
  m_enable_path_smoothing = node_->declare_parameter<bool8_t>("enable_path_smoothing");
  m_path_filter_moving_ave_num = node_->declare_parameter<int64_t>("path_filter_moving_ave_num");
  m_curvature_smoothing_num_traj =
    node_->declare_parameter<int64_t>("curvature_smoothing_num_traj");
  m_curvature_smoothing_num_ref_steer =
    node_->declare_parameter<int64_t>("curvature_smoothing_num_ref_steer");
  m_traj_resample_dist = node_->declare_parameter<float64_t>("traj_resample_dist");
  m_mpc.m_admissible_position_error =
    node_->declare_parameter<float64_t>("admissible_position_error");
  m_mpc.m_admissible_yaw_error_rad =
    node_->declare_parameter<float64_t>("admissible_yaw_error_rad");
  m_mpc.m_use_steer_prediction = node_->declare_parameter<bool8_t>("use_steer_prediction");
  m_mpc.m_param.steer_tau = node_->declare_parameter<float64_t>("vehicle_model_steer_tau");

  /* stop state parameters */
  m_stop_state_entry_ego_speed = node_->declare_parameter<float64_t>("stop_state_entry_ego_speed");
  m_stop_state_entry_target_speed =
    node_->declare_parameter<float64_t>("stop_state_entry_target_speed");

  /* mpc parameters */
  const float64_t steer_lim_deg = node_->declare_parameter<float64_t>("steer_lim_deg");
  const float64_t steer_rate_lim_dps = node_->declare_parameter<float64_t>("steer_rate_lim_dps");
  constexpr float64_t deg2rad = static_cast<float64_t>(autoware::common::types::PI) / 180.0;
  m_mpc.m_steer_lim = steer_lim_deg * deg2rad;
  m_mpc.m_steer_rate_lim = steer_rate_lim_dps * deg2rad;
  const float64_t wheelbase =
    vehicle_info_util::VehicleInfoUtil(*node_).getVehicleInfo().wheel_base_m;

  /* vehicle model setup */
  const std::string vehicle_model_type =
    node_->declare_parameter<std::string>("vehicle_model_type");
  std::shared_ptr<trajectory_follower::VehicleModelInterface> vehicle_model_ptr;
  if (vehicle_model_type == "kinematics") {
    vehicle_model_ptr = std::make_shared<trajectory_follower::KinematicsBicycleModel>(
      wheelbase, m_mpc.m_steer_lim, m_mpc.m_param.steer_tau);
  } else if (vehicle_model_type == "kinematics_no_delay") {
    vehicle_model_ptr = std::make_shared<trajectory_follower::KinematicsBicycleModelNoDelay>(
      wheelbase, m_mpc.m_steer_lim);
  } else if (vehicle_model_type == "dynamics") {
    const float64_t mass_fl = node_->declare_parameter<float64_t>("vehicle.mass_fl");
    const float64_t mass_fr = node_->declare_parameter<float64_t>("vehicle.mass_fr");
    const float64_t mass_rl = node_->declare_parameter<float64_t>("vehicle.mass_rl");
    const float64_t mass_rr = node_->declare_parameter<float64_t>("vehicle.mass_rr");
    const float64_t cf = node_->declare_parameter<float64_t>("vehicle.cf");
    const float64_t cr = node_->declare_parameter<float64_t>("vehicle.cr");

    // vehicle_model_ptr is only assigned in ctor, so parameter value have to be passed at init time
    // // NOLINT
    vehicle_model_ptr = std::make_shared<trajectory_follower::DynamicsBicycleModel>(
      wheelbase, mass_fl, mass_fr, mass_rl, mass_rr, cf, cr);
  } else {
    RCLCPP_ERROR(node_->get_logger(), "vehicle_model_type is undefined");
  }

  /* QP solver setup */
  const std::string qp_solver_type = node_->declare_parameter<std::string>("qp_solver_type");
  std::shared_ptr<trajectory_follower::QPSolverInterface> qpsolver_ptr;
  if (qp_solver_type == "unconstraint_fast") {
    qpsolver_ptr = std::make_shared<trajectory_follower::QPSolverEigenLeastSquareLLT>();
  } else if (qp_solver_type == "osqp") {
    qpsolver_ptr = std::make_shared<trajectory_follower::QPSolverOSQP>(node_->get_logger());
  } else {
    RCLCPP_ERROR(node_->get_logger(), "qp_solver_type is undefined");
  }

  /* delay compensation */
  {
    const float64_t delay_tmp = node_->declare_parameter<float64_t>("input_delay");
    const float64_t delay_step = std::round(delay_tmp / m_mpc.m_ctrl_period);
    m_mpc.m_param.input_delay = delay_step * m_mpc.m_ctrl_period;
    m_mpc.m_input_buffer = std::deque<float64_t>(static_cast<size_t>(delay_step), 0.0);
  }

  /* initialize lowpass filter */
  {
    const float64_t steering_lpf_cutoff_hz =
      node_->declare_parameter<float64_t>("steering_lpf_cutoff_hz");
    const float64_t error_deriv_lpf_cutoff_hz =
      node_->declare_parameter<float64_t>("error_deriv_lpf_cutoff_hz");
    m_mpc.initializeLowPassFilters(steering_lpf_cutoff_hz, error_deriv_lpf_cutoff_hz);
  }

  /* set up ros system */
  // initTimer(m_mpc.m_ctrl_period);

  m_pub_ctrl_cmd =
    node_->create_publisher<autoware_auto_control_msgs::msg::AckermannLateralCommand>(
      "~/output/lateral_control_cmd", 1);
  m_pub_predicted_traj = node_->create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/output/predicted_trajectory", 1);
  m_pub_diagnostic =
    node_->create_publisher<autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic>(
      "~/output/lateral_diagnostic", 1);
  m_sub_ref_path = node_->create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/input/current_trajectory", rclcpp::QoS{1},
    std::bind(&MpcLateralController::onTrajectory, this, _1));
  m_sub_steering = node_->create_subscription<autoware_auto_vehicle_msgs::msg::SteeringReport>(
    "~/input/current_steering", rclcpp::QoS{1},
    std::bind(&MpcLateralController::onSteering, this, _1));
  m_sub_odometry = node_->create_subscription<nav_msgs::msg::Odometry>(
    "~/input/current_odometry", rclcpp::QoS{1},
    std::bind(&MpcLateralController::onOdometry, this, _1));

  // TODO(Frederik.Beaujean) ctor is too long, should factor out parameter declarations
  declareMPCparameters();

  /* get parameter updates */
  m_set_param_res = node_->add_on_set_parameters_callback(
    std::bind(&MpcLateralController::paramCallback, this, _1));

  m_mpc.setQPSolver(qpsolver_ptr);
  m_mpc.setVehicleModel(vehicle_model_ptr, vehicle_model_type);

  m_mpc.setLogger(node_->get_logger());
  m_mpc.setClock(node_->get_clock());
}

MpcLateralController::~MpcLateralController()
{
  autoware_auto_control_msgs::msg::AckermannLateralCommand stop_cmd = getStopControlCommand();
  createCtrlCmdMsg(stop_cmd);  // todo
}

LateralOutput MpcLateralController::run()
{
  if (!checkData() || !updateCurrentPose()) {
    LateralOutput output{};
    return output;  // todo
  }

  autoware_auto_control_msgs::msg::AckermannLateralCommand ctrl_cmd;
  autoware_auto_planning_msgs::msg::Trajectory predicted_traj;
  autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic diagnostic;

  if (!m_is_ctrl_cmd_prev_initialized) {
    m_ctrl_cmd_prev = getInitialControlCommand();
    m_is_ctrl_cmd_prev_initialized = true;
  }

  const bool8_t is_mpc_solved = m_mpc.calculateMPC(
    *m_current_steering_ptr, m_current_odometry_ptr->twist.twist.linear.x, m_current_pose_ptr->pose,
    ctrl_cmd, predicted_traj, diagnostic);

  if (isStoppedState()) {
    // Reset input buffer
    for (auto & value : m_mpc.m_input_buffer) {
      value = m_ctrl_cmd_prev.steering_tire_angle;
    }
    // Use previous command value as previous raw steer command
    m_mpc.m_raw_steer_cmd_prev = m_ctrl_cmd_prev.steering_tire_angle;

    const auto cmd_msg = createCtrlCmdMsg(m_ctrl_cmd_prev);
    publishPredictedTraj(predicted_traj);
    publishDiagnostic(diagnostic);
    LateralOutput output{cmd_msg};
    return output;
  }

  if (!is_mpc_solved) {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000 /*ms*/,
      "MPC is not solved. publish 0 velocity.");
    ctrl_cmd = getStopControlCommand();
  }

  m_ctrl_cmd_prev = ctrl_cmd;
  const auto cmd_msg = createCtrlCmdMsg(ctrl_cmd);
  publishPredictedTraj(predicted_traj);
  publishDiagnostic(diagnostic);
  LateralOutput output{cmd_msg};
  return output;
}

bool8_t MpcLateralController::checkData() const
{
  if (!m_mpc.hasVehicleModel()) {
    RCLCPP_DEBUG(node_->get_logger(), "MPC does not have a vehicle model");
    return false;
  }
  if (!m_mpc.hasQPSolver()) {
    RCLCPP_DEBUG(node_->get_logger(), "MPC does not have a QP solver");
    return false;
  }

  if (!m_current_odometry_ptr) {
    RCLCPP_DEBUG(
      node_->get_logger(), "waiting data. current_velocity = %d",
      m_current_odometry_ptr != nullptr);
    return false;
  }

  if (!m_current_steering_ptr) {
    RCLCPP_DEBUG(
      node_->get_logger(), "waiting data. current_steering = %d",
      m_current_steering_ptr != nullptr);
    return false;
  }

  if (m_mpc.m_ref_traj.size() == 0) {
    RCLCPP_DEBUG(node_->get_logger(), "trajectory size is zero.");
    return false;
  }

  return true;
}

void MpcLateralController::onTrajectory(
  const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
  m_current_trajectory_ptr = msg;

  if (!m_current_pose_ptr && !updateCurrentPose()) {
    RCLCPP_DEBUG(node_->get_logger(), "Current pose is not received yet.");
    return;
  }

  if (msg->points.size() < 3) {
    RCLCPP_DEBUG(node_->get_logger(), "received path size is < 3, not enough.");
    return;
  }

  if (!isValidTrajectory(*msg)) {
    RCLCPP_ERROR(node_->get_logger(), "Trajectory is invalid!! stop computing.");
    return;
  }

  m_mpc.setReferenceTrajectory(
    *msg, m_traj_resample_dist, m_enable_path_smoothing, m_path_filter_moving_ave_num,
    m_curvature_smoothing_num_traj, m_curvature_smoothing_num_ref_steer, m_current_pose_ptr);
}

bool8_t MpcLateralController::updateCurrentPose()
{
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = m_tf_buffer.lookupTransform(
      m_current_trajectory_ptr->header.frame_id, "base_link", tf2::TimePointZero);
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN_SKIPFIRST_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000 /*ms*/, ex.what());
    RCLCPP_WARN_SKIPFIRST_THROTTLE(
      node_->get_logger(), *node_->get_clock(), 5000 /*ms*/,
      m_tf_buffer.allFramesAsString().c_str());
    return false;
  }

  geometry_msgs::msg::PoseStamped ps;
  ps.header = transform.header;
  ps.pose.position.x = transform.transform.translation.x;
  ps.pose.position.y = transform.transform.translation.y;
  ps.pose.position.z = transform.transform.translation.z;
  ps.pose.orientation = transform.transform.rotation;
  m_current_pose_ptr = std::make_shared<geometry_msgs::msg::PoseStamped>(ps);
  return true;
}

void MpcLateralController::onOdometry(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  m_current_odometry_ptr = msg;
}

void MpcLateralController::onSteering(
  const autoware_auto_vehicle_msgs::msg::SteeringReport::SharedPtr msg)
{
  m_current_steering_ptr = msg;
}

autoware_auto_control_msgs::msg::AckermannLateralCommand
MpcLateralController::getStopControlCommand() const
{
  autoware_auto_control_msgs::msg::AckermannLateralCommand cmd;
  cmd.steering_tire_angle = static_cast<decltype(cmd.steering_tire_angle)>(m_steer_cmd_prev);
  cmd.steering_tire_rotation_rate = 0.0;
  return cmd;
}

autoware_auto_control_msgs::msg::AckermannLateralCommand
MpcLateralController::getInitialControlCommand() const
{
  autoware_auto_control_msgs::msg::AckermannLateralCommand cmd;
  cmd.steering_tire_angle = m_current_steering_ptr->steering_tire_angle;
  cmd.steering_tire_rotation_rate = 0.0;
  return cmd;
}

bool8_t MpcLateralController::isStoppedState() const
{
  // Note: This function used to take into account the distance to the stop line
  // for the stop state judgement. However, it has been removed since the steering
  // control was turned off when approaching/exceeding the stop line on a curve or
  // emergency stop situation and it caused large tracking error.
  const int64_t nearest = trajectory_follower::MPCUtils::calcNearestIndex(
    *m_current_trajectory_ptr, m_current_pose_ptr->pose);

  // If the nearest index is not found, return false
  if (nearest < 0) {
    return false;
  }

  const float64_t current_vel = m_current_odometry_ptr->twist.twist.linear.x;
  const float64_t target_vel =
    m_current_trajectory_ptr->points.at(static_cast<size_t>(nearest)).longitudinal_velocity_mps;
  if (
    std::fabs(current_vel) < m_stop_state_entry_ego_speed &&
    std::fabs(target_vel) < m_stop_state_entry_target_speed) {
    return true;
  } else {
    return false;
  }
}

autoware_auto_control_msgs::msg::AckermannLateralCommand MpcLateralController::createCtrlCmdMsg(
  autoware_auto_control_msgs::msg::AckermannLateralCommand ctrl_cmd)
{
  ctrl_cmd.stamp = node_->now();
  // m_pub_ctrl_cmd->publish(ctrl_cmd);
  m_steer_cmd_prev = ctrl_cmd.steering_tire_angle;
  return ctrl_cmd;
}

void MpcLateralController::publishPredictedTraj(
  autoware_auto_planning_msgs::msg::Trajectory & predicted_traj) const
{
  predicted_traj.header.stamp = node_->now();
  predicted_traj.header.frame_id = m_current_trajectory_ptr->header.frame_id;
  m_pub_predicted_traj->publish(predicted_traj);
}

void MpcLateralController::publishDiagnostic(
  autoware_auto_system_msgs::msg::Float32MultiArrayDiagnostic & diagnostic) const
{
  diagnostic.diag_header.data_stamp = node_->now();
  diagnostic.diag_header.name = std::string("linear-MPC lateral controller");
  m_pub_diagnostic->publish(diagnostic);
}

// void MpcLateralController::initTimer(float64_t period_s)
// {
//   const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
//     std::chrono::duration<float64_t>(period_s));
//   m_timer = rclcpp::create_timer(
//     this, node_->get_clock(), period_ns, std::bind(&MpcLateralController::onTimer, this));
// }

void MpcLateralController::declareMPCparameters()
{
  m_mpc.m_param.prediction_horizon = node_->declare_parameter<int64_t>("mpc_prediction_horizon");
  m_mpc.m_param.prediction_dt = node_->declare_parameter<float64_t>("mpc_prediction_dt");
  m_mpc.m_param.weight_lat_error = node_->declare_parameter<float64_t>("mpc_weight_lat_error");
  m_mpc.m_param.weight_heading_error =
    node_->declare_parameter<float64_t>("mpc_weight_heading_error");
  m_mpc.m_param.weight_heading_error_squared_vel =
    node_->declare_parameter<float64_t>("mpc_weight_heading_error_squared_vel");
  m_mpc.m_param.weight_steering_input =
    node_->declare_parameter<float64_t>("mpc_weight_steering_input");
  m_mpc.m_param.weight_steering_input_squared_vel =
    node_->declare_parameter<float64_t>("mpc_weight_steering_input_squared_vel");
  m_mpc.m_param.weight_lat_jerk = node_->declare_parameter<float64_t>("mpc_weight_lat_jerk");
  m_mpc.m_param.weight_steer_rate = node_->declare_parameter<float64_t>("mpc_weight_steer_rate");
  m_mpc.m_param.weight_steer_acc = node_->declare_parameter<float64_t>("mpc_weight_steer_acc");
  m_mpc.m_param.low_curvature_weight_lat_error =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_lat_error");
  m_mpc.m_param.low_curvature_weight_heading_error =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_heading_error");
  m_mpc.m_param.low_curvature_weight_heading_error_squared_vel =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_heading_error_squared_vel");
  m_mpc.m_param.low_curvature_weight_steering_input =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_steering_input");
  m_mpc.m_param.low_curvature_weight_steering_input_squared_vel =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_steering_input_squared_vel");
  m_mpc.m_param.low_curvature_weight_lat_jerk =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_lat_jerk");
  m_mpc.m_param.low_curvature_weight_steer_rate =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_steer_rate");
  m_mpc.m_param.low_curvature_weight_steer_acc =
    node_->declare_parameter<float64_t>("mpc_low_curvature_weight_steer_acc");
  m_mpc.m_param.low_curvature_thresh_curvature =
    node_->declare_parameter<float64_t>("mpc_low_curvature_thresh_curvature");
  m_mpc.m_param.weight_terminal_lat_error =
    node_->declare_parameter<float64_t>("mpc_weight_terminal_lat_error");
  m_mpc.m_param.weight_terminal_heading_error =
    node_->declare_parameter<float64_t>("mpc_weight_terminal_heading_error");
  m_mpc.m_param.zero_ff_steer_deg = node_->declare_parameter<float64_t>("mpc_zero_ff_steer_deg");
  m_mpc.m_param.acceleration_limit = node_->declare_parameter<float64_t>("mpc_acceleration_limit");
  m_mpc.m_param.velocity_time_constant =
    node_->declare_parameter<float64_t>("mpc_velocity_time_constant");
}

rcl_interfaces::msg::SetParametersResult MpcLateralController::paramCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  // strong exception safety wrt MPCParam
  trajectory_follower::MPCParam param = m_mpc.m_param;
  try {
    update_param(parameters, "mpc_prediction_horizon", param.prediction_horizon);
    update_param(parameters, "mpc_prediction_dt", param.prediction_dt);
    update_param(parameters, "mpc_weight_lat_error", param.weight_lat_error);
    update_param(parameters, "mpc_weight_heading_error", param.weight_heading_error);
    update_param(
      parameters, "mpc_weight_heading_error_squared_vel", param.weight_heading_error_squared_vel);
    update_param(parameters, "mpc_weight_steering_input", param.weight_steering_input);
    update_param(
      parameters, "mpc_weight_steering_input_squared_vel", param.weight_steering_input_squared_vel);
    update_param(parameters, "mpc_weight_lat_jerk", param.weight_lat_jerk);
    update_param(parameters, "mpc_weight_steer_rate", param.weight_steer_rate);
    update_param(parameters, "mpc_weight_steer_acc", param.weight_steer_acc);
    update_param(
      parameters, "mpc_low_curvature_weight_lat_error", param.low_curvature_weight_lat_error);
    update_param(
      parameters, "mpc_low_curvature_weight_heading_error",
      param.low_curvature_weight_heading_error);
    update_param(
      parameters, "mpc_low_curvature_weight_heading_error_squared_vel",
      param.low_curvature_weight_heading_error_squared_vel);
    update_param(
      parameters, "mpc_low_curvature_weight_steering_input",
      param.low_curvature_weight_steering_input);
    update_param(
      parameters, "mpc_low_curvature_weight_steering_input_squared_vel",
      param.low_curvature_weight_steering_input_squared_vel);
    update_param(
      parameters, "mpc_low_curvature_weight_lat_jerk", param.low_curvature_weight_lat_jerk);
    update_param(
      parameters, "mpc_low_curvature_weight_steer_rate", param.low_curvature_weight_steer_rate);
    update_param(
      parameters, "mpc_low_curvature_weight_steer_acc", param.low_curvature_weight_steer_acc);
    update_param(
      parameters, "mpc_low_curvature_thresh_curvature", param.low_curvature_thresh_curvature);
    update_param(parameters, "mpc_weight_terminal_lat_error", param.weight_terminal_lat_error);
    update_param(
      parameters, "mpc_weight_terminal_heading_error", param.weight_terminal_heading_error);
    update_param(parameters, "mpc_zero_ff_steer_deg", param.zero_ff_steer_deg);
    update_param(parameters, "mpc_acceleration_limit", param.acceleration_limit);
    update_param(parameters, "mpc_velocity_time_constant", param.velocity_time_constant);

    // initialize input buffer
    update_param(parameters, "input_delay", param.input_delay);
    const float64_t delay_step = std::round(param.input_delay / m_mpc.m_ctrl_period);
    const float64_t delay = delay_step * m_mpc.m_ctrl_period;
    if (param.input_delay != delay) {
      param.input_delay = delay;
      m_mpc.m_input_buffer = std::deque<float64_t>(static_cast<size_t>(delay_step), 0.0);
    }

    // transaction succeeds, now assign values
    m_mpc.m_param = param;
  } catch (const rclcpp::exceptions::InvalidParameterTypeException & e) {
    result.successful = false;
    result.reason = e.what();
  }

  return result;
}

bool8_t MpcLateralController::isValidTrajectory(
  const autoware_auto_planning_msgs::msg::Trajectory & traj) const
{
  for (const auto & p : traj.points) {
    if (
      !isfinite(p.pose.position.x) || !isfinite(p.pose.position.y) ||
      !isfinite(p.pose.orientation.w) || !isfinite(p.pose.orientation.x) ||
      !isfinite(p.pose.orientation.y) || !isfinite(p.pose.orientation.z) ||
      !isfinite(p.longitudinal_velocity_mps) || !isfinite(p.lateral_velocity_mps) ||
      !isfinite(p.lateral_velocity_mps) || !isfinite(p.heading_rate_rps) ||
      !isfinite(p.front_wheel_angle_rad) || !isfinite(p.rear_wheel_angle_rad)) {
      return false;
    }
  }
  return true;
}

}  // namespace trajectory_follower
}  // namespace control
}  // namespace motion
}  // namespace autoware
