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

#include "perception_evaluator/metrics/deviation_metrics.hpp"

#include "tier4_autoware_utils/geometry/geometry.hpp"
#include "tier4_autoware_utils/geometry/pose_deviation.hpp"

#include <motion_utils/trajectory/trajectory.hpp>
#include <tier4_autoware_utils/geometry/geometry.hpp>

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
namespace metrics
{

double calcLateralDeviation(const std::vector<Pose> & ref_path, const Pose & target_pose)
{
  if (ref_path.empty()) {
    return 0.0;
  }

  const size_t nearest_index = motion_utils::findNearestIndex(ref_path, target_pose.position);
  return std::abs(
    tier4_autoware_utils::calcLateralDeviation(ref_path[nearest_index], target_pose.position));
}

double calcYawDeviation(const std::vector<Pose> & ref_path, const Pose & target_pose)
{
  if (ref_path.empty()) {
    return 0.0;
  }

  const size_t nearest_index = motion_utils::findNearestIndex(ref_path, target_pose.position);
  return std::abs(tier4_autoware_utils::calcYawDeviation(ref_path[nearest_index], target_pose));
}

std::vector<double> calcPredictedPathDeviation(
  const std::vector<Pose> & ref_path, const PredictedPath & pred_path)
{
  std::vector<double> deviations;

  if (ref_path.empty() || pred_path.path.empty()) {
    return {};
  }
  for (const Pose & p : pred_path.path) {
    const size_t nearest_index = motion_utils::findNearestIndex(ref_path, p.position);
    deviations.push_back(
      tier4_autoware_utils::calcDistance2d(ref_path[nearest_index].position, p.position));
  }

  return deviations;
}
}  // namespace metrics
}  // namespace perception_diagnostics
