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

#ifndef PERCEPTION_EVALUATOR__METRICS__OBSTACLE_METRICS_HPP_
#define PERCEPTION_EVALUATOR__METRICS__OBSTACLE_METRICS_HPP_

#include "perception_evaluator/stat.hpp"

#include "autoware_auto_perception_msgs/msg/predicted_objects.hpp"
#include "autoware_auto_perception_msgs/msg/trajectory.hpp"

namespace perception_diagnostics
{
namespace metrics
{
using autoware_auto_perception_msgs::msg::PredictedObjects;
using autoware_auto_perception_msgs::msg::Trajectory;

/**
 * @brief calculate the distance to the closest obstacle at each point of the trajectory
 * @param [in] obstacles obstacles
 * @param [in] traj trajectory
 * @return calculated statistics
 */
Stat<double> calcDistanceToObstacle(const PredictedObjects & obstacles, const Trajectory & traj);

/**
 * @brief calculate the time to collision of the trajectory with the given obstacles
 * Assume that "now" corresponds to the first trajectory point
 * @param [in] obstacles obstacles
 * @param [in] traj trajectory
 * @return calculated statistics
 */
Stat<double> calcTimeToCollision(
  const PredictedObjects & obstacles, const Trajectory & traj, const double distance_threshold);

}  // namespace metrics
}  // namespace perception_diagnostics

#endif  // PERCEPTION_EVALUATOR__METRICS__OBSTACLE_METRICS_HPP_