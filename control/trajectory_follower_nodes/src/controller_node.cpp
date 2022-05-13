// Copyright 2021 Tier IV, Inc. All rights reserved.
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

#include "trajectory_follower_nodes/controller_node.hpp"

#include "motion_common/motion_common.hpp"
#include "motion_common/trajectory_common.hpp"
#include "time_utils/time_utils.hpp"
#include "trajectory_follower/mpc_lateral_controller.hpp"
#include "trajectory_follower/pid_longitudinal_controller.hpp"

#include <algorithm>
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
namespace trajectory_follower_nodes
{
Controller::Controller(const rclcpp::NodeOptions & node_options) : Node("controller", node_options)
{
  using std::placeholders::_1;

  const double m_ctrl_period = declare_parameter<float64_t>("ctrl_period", 0.015);

  lateral_controller_ = std::make_shared<trajectory_follower::MpcLateralController>(this);
  longitudinal_controller_ = std::make_shared<trajectory_follower::PidLongitudinalController>(this);

  m_control_cmd_pub_ = create_publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>(
    "~/output/control_cmd", rclcpp::QoS{1}.transient_local());

  // Timer
  {
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<float64_t>(m_ctrl_period));
    m_timer_control_ = rclcpp::create_timer(
      this, get_clock(), period_ns, std::bind(&Controller::callbackTimerControl, this));
  }
}

void Controller::callbackTimerControl()
{
  const auto longitudinal_output = longitudinal_controller_->run();
  const auto lateral_output = lateral_controller_->run();

  longitudinal_controller_->sync(lateral_output.sync_data);
  lateral_controller_->sync(longitudinal_output.sync_data);

  autoware_auto_control_msgs::msg::AckermannControlCommand out;
  out.stamp = this->now();
  out.lateral = lateral_output.control_cmd;
  out.longitudinal = longitudinal_output.control_cmd;
  m_control_cmd_pub_->publish(out);
}

}  // namespace trajectory_follower_nodes
}  // namespace control
}  // namespace motion
}  // namespace autoware

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::motion::control::trajectory_follower_nodes::Controller)
