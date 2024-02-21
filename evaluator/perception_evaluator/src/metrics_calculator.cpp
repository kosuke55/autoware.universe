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

#include "perception_evaluator/metrics_calculator.hpp"

#include "motion_utils/trajectory/trajectory.hpp"
#include "perception_evaluator/metrics/deviation_metrics.hpp"
#include "tier4_autoware_utils/geometry/geometry.hpp"

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
std::optional<Stat<double>> MetricsCalculator::calculate(const Metric metric) const
{
  line();
  // tmp implementation
  if (stamp_and_objects_map_.empty()) {
    line();
    return {};
  }
  line();
  if(stamp_and_objects_map_.rbegin()->second.objects.empty()) {
    line();
    return {};
  }
  const auto object = stamp_and_objects_map_.rbegin()->second.objects.front();
  line();
  const auto object_pose = object.kinematics.initial_pose_with_covariance.pose;
  if (object.kinematics.predicted_paths.empty()) {
    line();
    return {};
  }
  const auto predicted_path = object.kinematics.predicted_paths.front();

  line();

  // Functions to calculate pose metrics
  const auto history_path = generateHistoryPath();
  switch (metric) {
    case Metric::lateral_deviation:
      line();
      return metrics::calcLateralDeviation(history_path, object_pose);
    case Metric::yaw_deviation:
      line();
      return metrics::calcYawDeviation(history_path, object_pose);
    case Metric::predicted_path_deviation:
      line();
      return metrics::calcPredictedPathDeviation(history_path, predicted_path);
    default:
      line();
      return {};
  }
}

void MetricsCalculator::setPredictedObjects(const PredictedObjects & objects)
{
  // pop if the size exceeds the history length
  if (stamp_and_objects_map_.size() > history_length_) {
    stamp_and_objects_map_.erase(stamp_and_objects_map_.begin());
  }

  // store the predicted objects
  stamp_and_objects_map_.emplace(objects.header.stamp, objects);
}

std::vector<Pose> MetricsCalculator::generateHistoryPath() const
{
  std::vector<Pose> history_path;
  for (const auto & stamp_and_objects : stamp_and_objects_map_) {
    const auto & objects = stamp_and_objects.second;
    for (const auto & object : objects.objects) {
      history_path.push_back(object.kinematics.initial_pose_with_covariance.pose);
    }
  }

  const auto filtered_path = averageFilterPath(history_path, 5);

  return filtered_path;
}

std::vector<Pose> MetricsCalculator::averageFilterPath(
  const std::vector<Pose> & path, const size_t window_size) const
{
  if (path.empty() || window_size <= 1) {
    return path;  // Early return for edge cases
  }

  std::vector<Pose> filtered_path;
  filtered_path.reserve(path.size());  // Reserve space to avoid reallocations

  const int half_window = static_cast<int>(window_size) / 2;

  // Calculate the moving average for positions
  for (size_t i = 0; i < path.size(); ++i) {
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    size_t valid_points = 0;  // Correctly initialize and use as counter

    for (int j = std::max(static_cast<int>(i) - half_window, 0);
         j <= std::min(static_cast<int>(i) + half_window, static_cast<int>(path.size()) - 1); ++j) {
      sum_x += path[j].position.x;
      sum_y += path[j].position.y;
      sum_z += path[j].position.z;
      ++valid_points;
    }

    Pose average_pose;
    if (valid_points > 0) {  // Prevent division by zero
      average_pose.position.x = sum_x / valid_points;
      average_pose.position.y = sum_y / valid_points;
      average_pose.position.z = sum_z / valid_points;
    }

    // Placeholder for orientation to ensure structure integrity
    average_pose.orientation = geometry_msgs::msg::Quaternion{};
    filtered_path.push_back(average_pose);
  }

  // Calculate yaw and convert to quaternion after averaging positions
  for (size_t i = 0; i < filtered_path.size(); ++i) {
    if (i < filtered_path.size() - 1) {
      const double yaw = tier4_autoware_utils::calcAzimuthAngle(
        filtered_path[i].position, filtered_path[i + 1].position);
      filtered_path[i].orientation = tier4_autoware_utils::createQuaternionFromYaw(yaw);
    } else if (filtered_path.size() > 1) {
      // For the last point, use the orientation of the second-to-last point
      filtered_path[i].orientation = filtered_path[i - 1].orientation;
    }
  }

  return filtered_path;
}

}  // namespace perception_diagnostics
