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

#ifndef PERCEPTION_EVALUATOR__METRICS_CALCULATOR_HPP_
#define PERCEPTION_EVALUATOR__METRICS_CALCULATOR_HPP_

#include "perception_evaluator/metrics/metric.hpp"
#include "perception_evaluator/parameters.hpp"
#include "perception_evaluator/stat.hpp"

#include "autoware_auto_perception_msgs/msg/predicted_objects.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include <optional>

namespace perception_diagnostics
{
using autoware_auto_perception_msgs::msg::PredictedObjects;
using geometry_msgs::msg::Point;
using geometry_msgs::msg::Pose;

class MetricsCalculator
{
public:
  Parameters parameters;

  MetricsCalculator() = default;

  /**
   * @brief calculate
   * @param [in] metric Metric enum value
   * @return string describing the requested metric
   */
  std::optional<Stat<double>> calculate(
    const Metric metric, const PredictedObjects & objects) const;

  /**
   * @brief set the dynamic objects used to calculate obstacle metrics
   * @param [in] traj input previous trajectory
   */
  void setPredictedObjects(const PredictedObjects & objects);

private:
  PredictedObjects dynamic_objects_;
  std::vector<PredictedObjects> dynamic_objects_history_;
};  // class MetricsCalculator

}  // namespace perception_diagnostics

#endif  // PERCEPTION_EVALUATOR__METRICS_CALCULATOR_HPP_
