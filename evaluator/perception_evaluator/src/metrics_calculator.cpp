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
namespace perception_diagnostics
{
std::optional<Stat<double>> MetricsCalculator::calculate(
  const Metric metric, const PredictedObjects & objects) const
{
  const std::vector<Pose> history_path{};

  // tmp implementation
  const auto object = objects.objects.front();
  const auto object_pose = object.kinematics.initial_pose_with_covariance.pose;
  const auto predicted_path = object.kinematics.predicted_paths.front();

  // Functions to calculate pose metrics
  // tmp implementation
  switch (metric) {
    case Metric::lateral_deviation:
      return metrics::calcLateralDeviation(history_path, object_pose);
    case Metric::yaw_deviation:
      return metrics::calcYawDeviation(history_path, object_pose);
    case Metric::predicted_path_deviation:
      return metrics::calcPredictedPathDeviation(history_path, predicted_path);
    default:
      return {};
  }
}

void MetricsCalculator::setPredictedObjects(const PredictedObjects & objects)
{
  dynamic_objects_ = objects;
  dynamic_objects_history_.push_back(objects);
}
}  // namespace perception_diagnostics
