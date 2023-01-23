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

#include "behavior_path_planner/planner_manager.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/utilities.hpp>

#include <boost/format.hpp>

#include <memory>
#include <string>

namespace behavior_path_planner
{

namespace
{
static UUID generateUUID()
{
  // Generate random number
  UUID uuid;
  std::mt19937 gen(std::random_device{}());
  std::independent_bits_engine<std::mt19937, 8, uint8_t> bit_eng(gen);
  std::generate(uuid.uuid.begin(), uuid.uuid.end(), bit_eng);

  return uuid;
}
}  // namespace

PlannerManager::PlannerManager(rclcpp::Node & node, const bool verbose)
: logger_(node.get_logger().get_child("planner_manager")),
  clock_(*node.get_clock()),
  verbose_{verbose}
{
}

BehaviorModuleOutput PlannerManager::run(const std::shared_ptr<PlannerData> & data)
{
  stop_watch_.tic("processing_time");

  if (!start_lanelet_) {
    updateStartLanelet(data);
  }

  std::for_each(scene_manager_ptrs_.begin(), scene_manager_ptrs_.end(), [&data](const auto & m) {
    m->setData(data);
  });

  while (rclcpp::ok()) {
    /**
     * STEP1: get approved modules' output
     */
    const auto approved_path_data = update(data);

    /**
     * STEP2: check modules that need to be launched
     */
    const auto candidate_module_id = getCandidateModuleID(approved_path_data);

    /**
     * STEP3: if there is no module that need to be launched, return approved modues' output
     */
    if (!candidate_module_id) {
      candidate_module_id_ = boost::none;
      processing_time_ = stop_watch_.toc("processing_time", true);
      return approved_path_data;
    }

    /**
     * STEP4: if there is module that should be launched, execute the module
     */
    const auto result = run(candidate_module_id.get(), approved_path_data);

    /**
     * STEP5: if the candidate module's modification is NOT approved yet, return the result.
     * NOTE: the result is output of the candidate module, but the output path don't contains path
     * shape modification that needs approval. On the other hand, it could include velocity profile
     * modification.
     */
    if (isWaitingApproval(candidate_module_id.get())) {
      candidate_module_id_ = candidate_module_id.get();
      processing_time_ = stop_watch_.toc("processing_time", true);
      return result;
    }

    /**
     * STEP6: if the candidate module is approved, push the module into approved_modules_
     */
    candidate_module_id_ = boost::none;
    addApprovedModule(candidate_module_id.get());
  }

  processing_time_ = stop_watch_.toc("processing_time", true);

  return {};
}

boost::optional<ModuleID> PlannerManager::getCandidateModuleID(
  const BehaviorModuleOutput & previous_module_output) const
{
  std::vector<ModuleID> request_modules{};

  bool lc_running = false;
  bool avoid_running = false;
  for (const auto & m : approved_modules_) {
    const auto & manager = m.first;
    if (manager->getModuleName() == "lane_change") {
      lc_running = isRunning(m);
    }
    if (manager->getModuleName() == "avoidance") {
      avoid_running = isRunning(m);
    }
  }

  const auto block_simlutaneous_execution = [this]() {
    for (const auto & m : approved_modules_) {
      const auto & manager = m.first;
      if (!manager->isSimultaneousExecutable()) {
        return true;
      }
    }
    return false;
  }();

  // pickup execution requested modules
  for (const auto & m : scene_manager_ptrs_) {
    // generate temporary uuid
    const auto uuid = generateUUID();

    // already exist the modules that don't support simultaneous execution.
    if (block_simlutaneous_execution) {
      continue;
    }

    // the manager's module doesn't support simultaneous execution.
    if (!approved_modules_.empty() && !m->isSimultaneousExecutable()) {
      continue;
    }

    if ((lc_running || avoid_running) && m->getModuleName() == "avoidance_by_lc") {
      continue;
    }

    /**
     * CASE1: there is no candidate module
     */
    if (!candidate_module_id_) {
      if (!m->canLaunchNewModule()) {
        continue;
      }

      if (m->isExecutionRequested(previous_module_output)) {
        // launch new candidate module
        request_modules.emplace_back(m, uuid);
      }

      continue;
    }

    // candidate module already exist
    const auto & manager = candidate_module_id_.get().first;
    const auto & name = manager->getModuleName();

    /**
     * CASE2: same name module already launched as candidate
     */
    if (name == m->getModuleName()) {
      // if (isExecutionRequested(candidate_module_id_.get())) {
      if (isRunning(candidate_module_id_.get())) {
        // keep current candidate module
        request_modules.push_back(candidate_module_id_.get());
      } else {
        // candidate module is no longer needed
        deleteExpiredModules(candidate_module_id_.get());
      }

      continue;
    }

    /**
     * CASE3: different name module already launched as candidate
     */

    // don't launch new module as candidate
    if (!m->canLaunchNewModule()) {
      continue;
    }

    if (!m->isExecutionRequested(previous_module_output)) {
      continue;
    }

    request_modules.emplace_back(m, uuid);
  }

  // select one module to run
  const auto high_priority_module = selectHighestPriorityModule(request_modules);

  if (!high_priority_module) {
    return {};
  }

  // post process
  {
    const auto & manager = high_priority_module.get().first;
    const auto & uuid = high_priority_module.get().second;

    // if the selected module is NOT registered in manager, registered the module
    if (!manager->exist(uuid)) {
      manager->registerNewModule(previous_module_output, uuid);
    }

    // if the current candidate module is NOT selected as high priority module, delete the candidate
    // module from manager
    for (const auto & m : request_modules) {
      if (m.first->getModuleName() != manager->getModuleName() && m.first->exist(m.second)) {
        deleteExpiredModules(m);
      }
    }
  }

  return high_priority_module;
}

boost::optional<ModuleID> PlannerManager::selectHighestPriorityModule(
  std::vector<ModuleID> & request_modules) const
{
  if (request_modules.empty()) {
    return {};
  }

  std::sort(request_modules.begin(), request_modules.end(), [](auto a, auto b) {
    return a.first->getPriority() < b.first->getPriority();
  });

  return request_modules.front();
}

BehaviorModuleOutput PlannerManager::update(const std::shared_ptr<PlannerData> & data)
{
  BehaviorModuleOutput output = getReferencePath(data);

  bool remove_after_module = false;

  for (auto itr = approved_modules_.begin(); itr != approved_modules_.end();) {
    // if one of the approved_modules_ is waiting approval, remove all approved module from the
    // waiting approval module.
    if (remove_after_module) {
      itr = approved_modules_.erase(itr);
      continue;
    }

    // execute approved module
    const auto result = run(*itr, output);

    // if (!isExecutionRequested(*itr)) {
    if (!isRunning(*itr)) {
      // option: remove expired module in the order in which they are activated.
      if (itr == approved_modules_.begin()) {
        const auto & manager = itr->first;
        const auto & name = manager->getModuleName();

        // update date on which lane the vhicle is driving
        if (name == "lane_change" || name == "avoidance_by_lc") {
          updateStartLanelet(data);
        }

        // unregister the expired module from manager
        deleteExpiredModules(*itr);
        itr = approved_modules_.erase(itr);
        output = result;
        continue;
      }
    }

    // if one of the approved_modules_ is waiting approval, the module is handled as candidate
    // module again.
    if (isWaitingApproval(*itr)) {
      candidate_module_id_ = *itr;
      itr = approved_modules_.erase(itr);
      remove_after_module = true;
      continue;
    }

    output = result;
    itr++;
  }

  return output;
}

BehaviorModuleOutput PlannerManager::getReferencePath(
  const std::shared_ptr<PlannerData> & data) const
{
  PathWithLaneId reference_path{};

  const auto & route_handler = data->route_handler;
  const auto & pose = data->self_pose->pose;
  const auto p = data->parameters;

  // Set header
  reference_path.header = route_handler->getRouteHeader();

  // const auto current_lanes = data->route_handler->getLaneletSequence(
  //   start_lanelet_.get(), pose, p.backward_path_length, p.forward_path_length);
  // const auto drivable_lanes = util::generateDrivableLanes(current_lanes);

  // calculate path with backward margin to avoid end points' instability by spline interpolation
  constexpr double extra_margin = 10.0;
  const double backward_length = p.backward_path_length + extra_margin;
  // const double backward_length =
  //   std::max(p.backward_path_length, p.backward_path_length + extra_margin);

  const auto current_lanes_with_backward_margin = route_handler->getLaneletSequence(
    start_lanelet_.get(), pose, backward_length, std::numeric_limits<double>::max());
  // start_lanelet_.get(), pose, backward_length, p.forward_path_length);

  reference_path = util::getCenterLinePath(
    *route_handler, current_lanes_with_backward_margin, pose, backward_length,
    p.forward_path_length, p);

  // clip backward length
  // const size_t current_seg_idx = motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
  //   reference_path.points, pose, 3.0, 1.0);
  // util::clipPathLength(reference_path, current_seg_idx, p.forward_path_length, backward_length);

  // const auto expanded_lanes = util::expandLanelets(drivable_lanes, 0.0, 0.0);
  // util::generateDrivableArea(reference_path, expanded_lanes, p.vehicle_length, data);
  reference_path.drivable_area = util::generateDrivableArea(
    reference_path, current_lanes_with_backward_margin, p.drivable_area_resolution,
    p.vehicle_length, data);

  BehaviorModuleOutput output;
  output.path = std::make_shared<PathWithLaneId>(reference_path);
  output.reference_path = std::make_shared<PathWithLaneId>(reference_path);

  return output;
}

}  // namespace behavior_path_planner
