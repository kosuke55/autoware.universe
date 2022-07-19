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

#ifndef BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_MODULE_HPP_
#define BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_MODULE_HPP_

#include "behavior_path_planner/scene_module/pull_out/pull_out_parameters.hpp"
#include "behavior_path_planner/scene_module/pull_out/pull_out_path.hpp"
#include "behavior_path_planner/scene_module/pull_out/shift_pull_out.hpp"
#include "behavior_path_planner/scene_module/scene_module_interface.hpp"
#include "behavior_path_planner/scene_module/utils/path_shifter.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <vehicle_info_util/vehicle_info.hpp>

#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>

#include <tf2/utils.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace behavior_path_planner
{
struct PullOutStatus
{
  PathWithLaneId lane_follow_path;
  PullOutPath pull_out_path;
  PullOutPath retreat_path;
  PullOutPath straight_back_path;
  lanelet::ConstLanelets current_lanes;
  lanelet::ConstLanelets pull_out_lanes;
  std::vector<uint64_t> lane_follow_lane_ids;
  std::vector<uint64_t> pull_out_lane_ids;
  bool is_safe;
  double start_distance;
  bool back_finished;
  bool is_retreat_path_valid;
  Pose backed_pose;
};

class PullOutModule : public SceneModuleInterface
{
public:
  PullOutModule(
    const std::string & name, rclcpp::Node & node, const PullOutParameters & parameters);

  BehaviorModuleOutput run() override;

  bool isExecutionRequested() const override;
  bool isExecutionReady() const override;
  BT::NodeStatus updateState() override;
  BehaviorModuleOutput plan() override;
  BehaviorModuleOutput planWaitingApproval() override;
  CandidateOutput planCandidate() const override;
  void onEntry() override;
  void onExit() override;

  void setParameters(const PullOutParameters & parameters);

private:
  std::shared_ptr<PullOutBase> pull_out_planner_;
  PullOutParameters parameters_;
  PullOutStatus status_;

  double pull_out_lane_length_ = 200.0;
  double check_distance_ = 100.0;
  std::vector<Pose> backed_pose_candidates_;
  PoseStamped backed_pose_;

  rclcpp::Publisher<PoseStamped>::SharedPtr backed_pose_pub_;

  PathWithLaneId getReferencePath() const;
  lanelet::ConstLanelets getCurrentLanes() const;
  // lanelet::ConstLanelets getPullOutLanes(const lanelet::ConstLanelets & current_lanes) const;
  std::pair<bool, bool> getSafePath(
    const lanelet::ConstLanelets & pull_out_lanes, const double check_distance,
    PullOutPath & safe_path) const;
  std::pair<bool, bool> getSafeRetreatPath(
    const lanelet::ConstLanelets & pull_out_lanes, const double check_distance,
    RetreatPath & safe_backed_path, double & back_distance) const;

  std::vector<Pose> searchBackedPoses();

  bool getBackDistance(
    const lanelet::ConstLanelets & pullover_lanes, const double check_distance,
    PullOutPath & safe_path, double & back_distance) const;

  // turn signal
  TurnSignalInfo calcTurnSignalInfo(const ShiftPoint & shift_point) const;

  void updatePullOutStatus();
  bool isInLane(
    const lanelet::ConstLanelet & candidate_lanelet,
    const tier4_autoware_utils::LinearRing2d & vehicle_footprint) const;
  bool isLongEnough(const lanelet::ConstLanelets & lanelets) const;
  bool isSafe() const;
  bool isNearEndOfLane() const;
  bool isCurrentSpeedLow() const;
  bool hasFinishedPullOut() const;
  bool hasFinishedBack() const;
  vehicle_info_util::VehicleInfo getVehicleInfo(
    const BehaviorPathPlannerParameters & parameters) const;
};
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_MODULE_HPP_
