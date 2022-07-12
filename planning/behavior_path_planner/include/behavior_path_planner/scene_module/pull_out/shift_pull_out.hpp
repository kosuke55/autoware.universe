// Copyright 2022 TIER IV, Inc.
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

#ifndef BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__SHIFT_PULL_OUT_HPP_
#define BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__SHIFT_PULL_OUT_HPP_

#include "behavior_path_planner/scene_module/pull_out/pull_out_base.hpp"

#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>

namespace behavior_path_planner
{
class ShiftPullOut : public PullOutBase
{
public:
  explicit ShiftPullOut(rclcpp::Node & node, const PullOutParameters & parameters);

  boost::optional<PathWithLaneId> plan(Pose start_pose, Pose goal_pose) override;
};
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__SHIFT_PULL_OUT_HPP_