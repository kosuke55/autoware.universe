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

#ifndef PERCEPTION_EVALUATOR__PLANNING_EVALUATOR_NODE_HPP_
#define PERCEPTION_EVALUATOR__PLANNING_EVALUATOR_NODE_HPP_

#include "perception_evaluator/metrics_calculator.hpp"
#include "perception_evaluator/parameters.hpp"
#include "perception_evaluator/stat.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "autoware_auto_perception_msgs/msg/predicted_objects.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace perception_diagnostics
{
using autoware_auto_perception_msgs::msg::ObjectClassification;
using autoware_auto_perception_msgs::msg::PredictedObjects;
using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using nav_msgs::msg::Odometry;

using MarkerArray = visualization_msgs::msg::MarkerArray;

/**
 * @brief Node for perception evaluation
 */
class PerceptionEvaluatorNode : public rclcpp::Node
{
public:
  explicit PerceptionEvaluatorNode(const rclcpp::NodeOptions & node_options);
  ~PerceptionEvaluatorNode();

  /**
   * @brief callback on receiving a dynamic objects array
   * @param [in] objects_msg received dynamic object array message
   */
  void onObjects(const PredictedObjects::ConstSharedPtr objects_msg);

  DiagnosticStatus generateDiagnosticStatus(
    const Metric & metric, const Stat<double> & metric_stat) const;

private:
  // Timer
  rclcpp::TimerBase::SharedPtr timer_;
  void initTimer(double period_s);
  void onTimer();

  // Subscribers and publishers
  rclcpp::Subscription<PredictedObjects>::SharedPtr objects_sub_;
  rclcpp::Publisher<DiagnosticArray>::SharedPtr metrics_pub_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_marker_;

  // TF
  std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

  // Parameters
  std::shared_ptr<Parameters> parameters_;
  void initParameter();
  rcl_interfaces::msg::SetParametersResult onParameter(
    const std::vector<rclcpp::Parameter> & parameters);
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  // Parameters
  std::string output_file_str_;
  std::string ego_frame_str_;

  // Metrics Calculator
  MetricsCalculator metrics_calculator_;
  std::deque<rclcpp::Time> stamps_;
  std::array<std::deque<Stat<double>>, static_cast<size_t>(Metric::SIZE)> metric_stats_;

  // Debug
  void publishDebugMarker();
};
}  // namespace perception_diagnostics

#endif  // PERCEPTION_EVALUATOR__PLANNING_EVALUATOR_NODE_HPP_
