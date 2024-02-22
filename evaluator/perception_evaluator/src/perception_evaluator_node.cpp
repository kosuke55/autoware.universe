// Copyright 2024 Tier IV, Inc.
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

#include "perception_evaluator/perception_evaluator_node.hpp"

#include "boost/lexical_cast.hpp"

#include <glog/logging.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
namespace {
std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens; std::string token; int p = 0;
  for (char c : s) {
    if (c == '(') p++; else if (c == ')') p--;
    if (c == delimiter && p == 0) { tokens.push_back(token); token.clear(); } else token += c;
  }
  if (!token.empty()) tokens.push_back(token);
  return tokens;
}

template <typename T> void view(const std::string &n, T e) { std::cerr << n << ": " << e << ", "; }
template <typename T> void view(const std::string &n, const std::vector<T> &v) { std::cerr << n << ":"; for (const auto &e : v) std::cerr << " " << e; std::cerr << ", "; }
template <typename First, typename... Rest> void view_multi(const std::vector<std::string> &n, First f, Rest... r) { view(n[0], f); if constexpr (sizeof...(r) > 0) view_multi(std::vector<std::string>(n.begin() + 1, n.end()), r...); }

template <typename... Args> void debug_helper(const char *f, int l, const char *n, Args... a) {
  std::cerr << f << ": " << l << ", "; auto nl = split(n, ',');
  for (auto &nn : nl) { nn.erase(nn.begin(), std::find_if(nn.begin(), nn.end(), [](int ch) { return !std::isspace(ch); })); nn.erase(std::find_if(nn.rbegin(), nn.rend(), [](int ch) { return !std::isspace(ch); }).base(), nn.end()); }
  view_multi(nl, a...); std::cerr << std::endl;
}

#define debug(...) debug_helper(__func__, __LINE__, #__VA_ARGS__, __VA_ARGS__)
#define line() { std::cerr << "(" << __FILE__ << ") " << __func__ << ": " << __LINE__ << std::endl; }
} // namespace
// clang-format on

namespace perception_diagnostics
{
PerceptionEvaluatorNode::PerceptionEvaluatorNode(const rclcpp::NodeOptions & node_options)
: Node("perception_evaluator", node_options)
{
  using std::placeholders::_1;

  google::InitGoogleLogging("map_based_prediction_node");
  google::InstallFailureSignalHandler();

  objects_sub_ = create_subscription<PredictedObjects>(
    "~/input/objects", 1, std::bind(&PerceptionEvaluatorNode::onObjects, this, _1));

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Parameters
  output_file_str_ = declare_parameter<std::string>("output_file");
  ego_frame_str_ = declare_parameter<std::string>("ego_frame");

  // List of metrics to calculate and publish
  metrics_pub_ = create_publisher<DiagnosticArray>("~/metrics", 1);
  for (const std::string & selected_metric :
       declare_parameter<std::vector<std::string>>("selected_metrics")) {
    const Metric metric = str_to_metric.at(selected_metric);
    metrics_.push_back(metric);
  }

  // Timer
  initTimer(/*period_s=*/0.1);
}

PerceptionEvaluatorNode::~PerceptionEvaluatorNode()
{
  if (!output_file_str_.empty()) {
    // column width is the maximum size we might print + 1 for the space between columns
    // Write data using format
    std::ofstream f(output_file_str_);
    f << std::fixed << std::left;
    // header
    f << "#Stamp(ns)";
    for (Metric metric : metrics_) {
      f << " " << metric_descriptions.at(metric);
      f << " . .";  // extra "columns" to align columns headers
    }
    f << std::endl;
    f << "#.";
    for (Metric metric : metrics_) {
      (void)metric;
      f << " min max mean";
    }
    f << std::endl;
    // data
    for (size_t i = 0; i < stamps_.size(); ++i) {
      f << stamps_[i].nanoseconds();
      for (Metric metric : metrics_) {
        const auto & stat = metric_stats_[static_cast<size_t>(metric)][i];
        f << " " << stat;
      }
      f << std::endl;
    }
    f.close();
  }
}

void PerceptionEvaluatorNode::initTimer(double period_s)
{
  const auto period_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(period_s));
  timer_ = rclcpp::create_timer(
    this, get_clock(), period_ns, std::bind(&PerceptionEvaluatorNode::onTimer, this));
}

void PerceptionEvaluatorNode::onTimer()
{
  DiagnosticArray metrics_msg;
  for (Metric metric : metrics_) {
    const auto metric_stat = metrics_calculator_.calculate(Metric(metric));
    if (!metric_stat.has_value()) {
      continue;
    }

    metric_stats_[static_cast<size_t>(metric)].push_back(*metric_stat);
    if (metric_stat->count() > 0) {
      metrics_msg.status.push_back(generateDiagnosticStatus(metric, *metric_stat));
    }
  }

  if (!metrics_msg.status.empty()) {
    metrics_msg.header.stamp = now();
    metrics_pub_->publish(metrics_msg);
  }
}

DiagnosticStatus PerceptionEvaluatorNode::generateDiagnosticStatus(
  const Metric & metric, const Stat<double> & metric_stat) const
{
  DiagnosticStatus status;
  status.level = status.OK;
  status.name = metric_to_str.at(metric);

  diagnostic_msgs::msg::KeyValue key_value;
  key_value.key = "min";
  key_value.value = std::to_string(metric_stat.min());
  status.values.push_back(key_value);
  key_value.key = "max";
  key_value.value = std::to_string(metric_stat.max());
  status.values.push_back(key_value);
  key_value.key = "mean";
  key_value.value = std::to_string(metric_stat.mean());
  status.values.push_back(key_value);

  return status;
}

void PerceptionEvaluatorNode::onObjects(const PredictedObjects::ConstSharedPtr objects_msg)
{
  metrics_calculator_.setPredictedObjects(*objects_msg);
}
}  // namespace perception_diagnostics

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(perception_diagnostics::PerceptionEvaluatorNode)
