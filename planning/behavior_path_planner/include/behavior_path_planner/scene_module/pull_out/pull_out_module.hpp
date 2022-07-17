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

#include <lane_departure_checker/lane_departure_checker.hpp>
#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <vehicle_info_util/vehicle_info.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <autoware_auto_planning_msgs/msg/path_with_lane_id.hpp>

#include <tf2/utils.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace behavior_path_planner
{
using geometry_msgs::msg::PoseArray;
using lane_departure_checker::LaneDepartureChecker;

struct PullOutStatus
{
  PathWithLaneId lane_follow_path;
  PullOutPath pull_out_path;
  PullOutPath retreat_path;
  PathWithLaneId backward_path;
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
  vehicle_info_util::VehicleInfo vehicle_info_;
  std::unique_ptr<rclcpp::Time> last_back_finished_time_;

  rclcpp::Publisher<PoseStamped>::SharedPtr backed_pose_pub_;
  rclcpp::Publisher<PoseArray>::SharedPtr full_path_pose_array_pub_;
  rclcpp::Clock::SharedPtr clock_;

  PathWithLaneId getReferencePath() const;
  lanelet::ConstLanelets getCurrentLanes() const;
  PathWithLaneId getFullPath() const;
  std::vector<Pose> searchBackedPoses();

  std::shared_ptr<LaneDepartureChecker> lane_departure_checker_;

  // turn signal
  TurnSignalInfo calcTurnSignalInfo(const Pose start_pose, const Pose end_pose) const;

  void updatePullOutStatus();
  bool isInLane(
    const lanelet::ConstLanelet & candidate_lanelet,
    const tier4_autoware_utils::LinearRing2d & vehicle_footprint) const;
  bool isLongEnough(const lanelet::ConstLanelets & lanelets) const;
  bool hasFinishedPullOut() const;
  void checkBackFinished();

  void publishDebugData() const;
};
}  // namespace behavior_path_planner

#endif  // BEHAVIOR_PATH_PLANNER__SCENE_MODULE__PULL_OUT__PULL_OUT_MODULE_HPP_
