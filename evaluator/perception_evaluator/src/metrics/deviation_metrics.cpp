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


namespace perception_diagnostics
{
namespace metrics
{

Stat<double> calcLateralDeviation(const std::vector<Pose> & ref_path, const Pose & target_pose)
{
  Stat<double> stat{};

  if (ref_path.empty()) {
    return stat;
  }

  const size_t nearest_index = motion_utils::findNearestIndex(ref_path, target_pose.position);
  stat.add(
    tier4_autoware_utils::calcLateralDeviation(ref_path[nearest_index], target_pose.position));

  return stat;
}

Stat<double> calcYawDeviation(const std::vector<Pose> & ref_path, const Pose & target_pose)
{
  Stat<double> stat{};

  if (ref_path.empty()) {
    return stat;
  }

  const size_t nearest_index = motion_utils::findNearestIndex(ref_path, target_pose.position);
  stat.add(tier4_autoware_utils::calcYawDeviation(ref_path[nearest_index], target_pose));

  return stat;
}

Stat<double> calcPredictedPathDeviation(
  const std::vector<Pose> & ref_path, const PredictedPath & pred_path)
{
  Stat<double> stat{};

  if (ref_path.empty() || pred_path.path.empty()) {
    return stat;
  }

  for (const Pose & p : pred_path.path) {
    const size_t nearest_index = motion_utils::findNearestIndex(ref_path, p.position);
    stat.add(tier4_autoware_utils::calcDistance2d(ref_path[nearest_index].position, p.position));
  }

  return stat;
}
}  // namespace metrics
}  // namespace perception_diagnostics
