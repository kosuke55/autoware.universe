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

#include "behavior_path_planner/scene_module/avoidance/avoidance_module.hpp"

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/avoidance/avoidance_utils.hpp"
#include "behavior_path_planner/scene_module/avoidance/debug.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <opencv2/opencv.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

// set as macro so that calling function name will be printed.
// debug print is heavy. turn on only when debugging.
#define DEBUG_PRINT(...) \
  RCLCPP_DEBUG_EXPRESSION(getLogger(), parameters_->print_debug_info, __VA_ARGS__)
#define printShiftPoints(p, msg) DEBUG_PRINT("[%s] %s", msg, toStrInfo(p).c_str())

namespace behavior_path_planner
{
using motion_utils::calcLongitudinalOffsetPoint;
using motion_utils::calcLongitudinalOffsetPose;
using motion_utils::calcSignedArcLength;
using motion_utils::findNearestIndex;
using motion_utils::findNearestSegmentIndex;
using motion_utils::insertTargetPoint;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcInterpolatedPose;
using tier4_autoware_utils::calcLateralDeviation;
using tier4_autoware_utils::calcLongitudinalDeviation;
using tier4_autoware_utils::createPoint;
using tier4_autoware_utils::getPoint;
using tier4_autoware_utils::getPose;
using tier4_autoware_utils::normalizeRadian;
using tier4_autoware_utils::pi;
using tier4_planning_msgs::msg::AvoidanceDebugFactor;

AvoidanceModule::AvoidanceModule(
  const std::string & name, rclcpp::Node & node, std::shared_ptr<AvoidanceParameters> parameters,
  std::shared_ptr<RTCInterface> & rtc_interface_left,
  std::shared_ptr<RTCInterface> & rtc_interface_right)
: SceneModuleInterface{name, node},
  parameters_{parameters},
  rtc_interface_left_{rtc_interface_left},
  rtc_interface_right_{rtc_interface_right},
  uuid_left_{generateUUID()},
  uuid_right_{generateUUID()}
{
  using std::placeholders::_1;
}

bool AvoidanceModule::isExecutionRequested() const
{
  DEBUG_PRINT("AVOIDANCE isExecutionRequested");

  // if (current_state_ == ModuleStatus::RUNNING) {
  //   return true;
  // }
  // auto avoid_data = calcAvoidancePlanningData(debug_data_);

  // fillAvoidancePath(avoid_data, debug_data_);

  if (parameters_->publish_debug_marker) {
    setDebugData(avoidance_data_, path_shifter_, debug_data_);
    // setDebugData(avoid_data, path_shifter_, debug_data_);
  }

  if (current_state_ == ModuleStatus::IDLE) {
    if (!avoidance_data_.avoid_required) {
      return false;
    }
    // if (avoidance_data_.unapproved_new_sp.empty()) {
    //   return false;
    // }
  }

  const bool has_base_offset = std::abs(path_shifter_.getBaseOffset()) > 0.01;
  return !avoidance_data_.target_objects.empty() || has_base_offset;
}

bool AvoidanceModule::isExecutionReady() const
{
  DEBUG_PRINT("AVOIDANCE isExecutionReady");

  // {
  //   DebugData debug;
  //   static_cast<void>(calcAvoidancePlanningData(debug));
  // }

  if (current_state_ == ModuleStatus::RUNNING) {
    return true;
  }

  return true;
}

ModuleStatus AvoidanceModule::updateState()
{
  const auto is_plan_running = isAvoidancePlanRunning();

  // DebugData debug;
  // const auto avoid_data = calcAvoidancePlanningData(debug);
  // const bool has_avoidance_target = !avoid_data.target_objects.empty();
  const bool has_avoidance_target = !avoidance_data_.target_objects.empty();
  if (!is_plan_running && !has_avoidance_target) {
    current_state_ = ModuleStatus::SUCCESS;
  } else {
    current_state_ = ModuleStatus::RUNNING;
  }

  DEBUG_PRINT(
    "is_plan_running = %d, has_avoidance_target = %d", is_plan_running, has_avoidance_target);

  return current_state_;
}

bool AvoidanceModule::isAvoidancePlanRunning() const
{
  const bool has_base_offset = std::abs(path_shifter_.getBaseOffset()) > 0.01;
  const bool has_shift_point = (path_shifter_.getShiftPointsSize() > 0);
  return has_base_offset || has_shift_point;
}

AvoidancePlanningData AvoidanceModule::calcAvoidancePlanningData(DebugData & debug) const
{
  AvoidancePlanningData data;

  // reference pose
  const auto reference_pose = getUnshiftedEgoPose(prev_output_);
  data.reference_pose = reference_pose.pose;

  // // center line path (output of this function must have size > 1)
  // const auto center_path = calcCenterLinePath(planner_data_, reference_pose);
  // debug.center_line = center_path;
  // if (center_path.points.size() < 2) {
  //   RCLCPP_WARN_THROTTLE(
  //     getLogger(), *clock_, 5000, "calcCenterLinePath() must return path which size > 1");
  //   return data;
  // }

  // // reference path
  // data.reference_path =
  //   util::resamplePathWithSpline(center_path, parameters_->resample_interval_for_planning);
  // if (data.reference_path.points.size() < 2) {
  //   // if the resampled path has only 1 point, use original path.
  //   data.reference_path = center_path;
  // }
  data.reference_path = util::resamplePathWithSpline(
    *previous_module_output_.path, parameters_->resample_interval_for_planning);

  const size_t nearest_segment_index =
    findNearestSegmentIndex(data.reference_path.points, data.reference_pose.position);
  data.ego_closest_path_index =
    std::min(nearest_segment_index + 1, data.reference_path.points.size() - 1);

  // arclength from ego pose (used in many functions)
  data.arclength_from_ego = util::calcPathArcLengthArray(
    data.reference_path, 0, data.reference_path.points.size(),
    calcSignedArcLength(data.reference_path.points, getEgoPosition(), 0));

  // lanelet info
  data.current_lanelets = calcLaneAroundPose(
    planner_data_, reference_pose.pose, parameters_->adjacent_lane_check_backward_distance);
  // planner_data_, reference_pose.pose, planner_data_->parameters.backward_path_length);

  // target objects for avoidance
  fillAvoidanceTargetObjects(data, debug);

  DEBUG_PRINT("target object size = %lu", data.target_objects.size());

  return data;
}

void AvoidanceModule::fillAvoidanceTargetObjects(
  AvoidancePlanningData & avoidance_data, DebugData & debug) const
{
  using boost::geometry::return_centroid;
  using boost::geometry::within;
  using lanelet::geometry::distance2d;
  using lanelet::geometry::toArcCoordinates;
  using lanelet::utils::getId;
  using lanelet::utils::to2D;

  const auto & path_points = avoidance_data.reference_path.points;
  const auto & ego_pos = getEgoPosition();

  // detection area filter
  // when expanding lanelets, right_offset must be minus.
  // This is because y axis is positive on the left.
  const auto expanded_lanelets = getTargetLanelets(
    planner_data_, avoidance_data.current_lanelets, parameters_->detection_area_left_expand_dist,
    parameters_->detection_area_right_expand_dist * (-1.0));

  PredictedObjects far_objects;
  const auto lane_filtered_objects =
    util::filterObjectsByLanelets(*planner_data_->dynamic_object, expanded_lanelets, far_objects);

  for (const auto & object : far_objects.objects) {
    ObjectData ignore_object;
    ignore_object.object = object;
    ignore_object.reason = "OutOfTargetArea";
    avoidance_data.ignore_objects.push_back(ignore_object);
  }

  DEBUG_PRINT("dynamic_objects size = %lu", planner_data_->dynamic_object->objects.size());

  // for goal
  const auto & rh = planner_data_->route_handler;
  const auto dist_to_goal =
    rh->isInGoalRouteSection(expanded_lanelets.back())
      ? calcSignedArcLength(path_points, ego_pos, rh->getGoalPose().position)
      : std::numeric_limits<double>::max();

  lanelet::ConstLineStrings3d debug_linestring;
  debug_linestring.clear();
  // for filtered objects
  ObjectDataArray target_objects;
  std::vector<AvoidanceDebugMsg> avoidance_debug_msg_array;
  for (const auto & object : lane_filtered_objects.objects) {
    const auto & object_pose = object.kinematics.initial_pose_with_covariance.pose;

    AvoidanceDebugMsg avoidance_debug_msg;
    const auto avoidance_debug_array_false_and_push_back =
      [&avoidance_debug_msg, &avoidance_debug_msg_array, &avoidance_data](
        const std::string & failed_reason, ObjectData & object) {
        avoidance_debug_msg.allow_avoidance = false;
        avoidance_debug_msg.failed_reason = failed_reason;
        avoidance_debug_msg_array.push_back(avoidance_debug_msg);
        object.reason = failed_reason;
        avoidance_data.ignore_objects.push_back(object);
      };

    ObjectData object_data;
    object_data.object = object;
    avoidance_debug_msg.object_id = getUuidStr(object_data);

    if (!isTargetObjectType(object)) {
      avoidance_debug_array_false_and_push_back(
        AvoidanceDebugFactor::OBJECT_IS_NOT_TYPE, object_data);
      continue;
    }

    const auto object_closest_index = findNearestIndex(path_points, object_pose.position);
    const auto object_closest_pose = path_points.at(object_closest_index).point.pose;

    // Create envelope polygon.
    fillObjectEnvelopePolygon(object_closest_pose, object_data);

    // calc object centroid.
    object_data.centroid = return_centroid<Point2d>(object_data.envelope_poly);

    // calc object moving time.
    fillObjectMovingTime(object_data);

    // calc longitudinal distance from ego to closest target object footprint point.
    fillLongitudinalAndLengthByClosestEnvelopeFootprint(
      avoidance_data.reference_path, ego_pos, object_data);
    avoidance_debug_msg.longitudinal_distance = object_data.longitudinal;

    // Calc lateral deviation from path to target object.
    object_data.lateral = calcLateralDeviation(object_closest_pose, object_pose.position);
    avoidance_debug_msg.lateral_distance_from_centerline = object_data.lateral;

    // Find the footprint point closest to the path, set to object_data.overhang_distance.
    object_data.overhang_dist = calcEnvelopeOverhangDistance(
      object_data, object_closest_pose, object_data.overhang_pose.position);

    // Check whether the the ego should avoid the object.
    const auto & vehicle_width = planner_data_->parameters.vehicle_width;
    const auto passable_buffer = 0.5 * vehicle_width + parameters_->lateral_passable_safety_buffer;
    object_data.avoid_required =
      (isOnRight(object_data) && std::abs(object_data.overhang_dist) < passable_buffer) ||
      (!isOnRight(object_data) && object_data.overhang_dist < 0.5 * passable_buffer);

    if (object_data.move_time > parameters_->threshold_time_object_is_moving) {
      avoidance_debug_array_false_and_push_back("MovingObject", object_data);
      continue;
    }

    // object is behind ego or too far.
    if (object_data.longitudinal < -parameters_->object_check_backward_distance) {
      avoidance_debug_array_false_and_push_back(
        AvoidanceDebugFactor::OBJECT_IS_BEHIND_THRESHOLD, object_data);
      continue;
    }
    if (object_data.longitudinal > parameters_->object_check_forward_distance) {
      avoidance_debug_array_false_and_push_back(
        AvoidanceDebugFactor::OBJECT_IS_IN_FRONT_THRESHOLD, object_data);
      continue;
    }

    // Target object is behind the path goal -> ignore.
    if (object_data.longitudinal > dist_to_goal) {
      avoidance_debug_array_false_and_push_back(
        AvoidanceDebugFactor::OBJECT_BEHIND_PATH_GOAL, object_data);
      continue;
    }

    lanelet::ConstLanelet overhang_lanelet;
    if (!rh->getClosestLaneletWithinRoute(object_closest_pose, &overhang_lanelet)) {
      continue;
    }

    if (overhang_lanelet.id()) {
      object_data.overhang_lanelet = overhang_lanelet;
      lanelet::BasicPoint3d overhang_basic_pose(
        object_data.overhang_pose.position.x, object_data.overhang_pose.position.y,
        object_data.overhang_pose.position.z);
      const bool get_left =
        isOnRight(object_data) && parameters_->enable_avoidance_over_same_direction;
      const bool get_right =
        !isOnRight(object_data) && parameters_->enable_avoidance_over_same_direction;

      const auto target_lines = rh->getFurthestLinestring(
        overhang_lanelet, get_right, get_left,
        parameters_->enable_avoidance_over_opposite_direction);

      if (isOnRight(object_data)) {
        object_data.to_road_shoulder_distance =
          distance2d(to2D(overhang_basic_pose), to2D(target_lines.back().basicLineString()));
        debug_linestring.push_back(target_lines.back());
      } else {
        object_data.to_road_shoulder_distance =
          distance2d(to2D(overhang_basic_pose), to2D(target_lines.front().basicLineString()));
        debug_linestring.push_back(target_lines.front());
      }
    }

    object_data.to_intersection_distance =
      util::getDistanceToNextIntersection(object_pose, avoidance_data.current_lanelets);
    if (
      object_data.stop_time > parameters_->threshold_time_force_avoidance &&
      parameters_->enable_avoidance_all_parking_vehicle) {
      if (
        object_data.to_intersection_distance >
        parameters_->threshold_intersection_force_avoidance) {
        object_data.last_seen = clock_->now();
        avoidance_data.target_objects.push_back(object_data);
        continue;
      }
    }

    avoidance_debug_msg.lateral_distance_from_centerline = object_data.lateral;
    if (std::abs(object_data.lateral) < parameters_->threshold_distance_object_is_on_center) {
      avoidance_debug_array_false_and_push_back(
        AvoidanceDebugFactor::TOO_NEAR_TO_CENTERLINE, object_data);
      continue;
    }

    DEBUG_PRINT(
      "set object_data: longitudinal = %f, lateral = %f, largest_overhang = %f,"
      "to_road_shoulder_distance = %f",
      object_data.longitudinal, object_data.lateral, object_data.overhang_dist,
      object_data.to_road_shoulder_distance);

    const auto object_norm_yaw = normalizeRadian(tf2::getYaw(object_pose.orientation));
    const auto path_norm_yaw = normalizeRadian(tf2::getYaw(object_closest_pose.orientation));
    object_data.relative_yaw = normalizeRadian(object_norm_yaw - path_norm_yaw);

    const auto merge_object =
      std::abs(object_data.relative_yaw) > parameters_->object_check_yaw &&
      std::abs(object_data.relative_yaw) < pi - parameters_->object_check_yaw;
    const auto merge_ego_lane =
      (object_data.overhang_dist < parameters_->object_check_overhang && !isOnRight(object_data)) ||
      (object_data.overhang_dist > -parameters_->object_check_overhang && isOnRight(object_data));

    // object is merging into the ego lane
    if (merge_object && merge_ego_lane) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::MERGING_OBJECT, object_data);
      continue;
    }

    lanelet::ConstLanelet object_closest_lanelet;
    const auto lanelet_map = rh->getLaneletMapPtr();
    if (!lanelet::utils::query::getClosestLanelet(
          lanelet::utils::query::laneletLayer(lanelet_map), object_pose, &object_closest_lanelet)) {
      continue;
    }

    lanelet::BasicPoint2d object_centroid(object_data.centroid.x(), object_data.centroid.y());

    /**
     * Is not object in adjacent lane?
     *   - Yes -> Is parking object?
     *     - Yes -> the object is avoidance target.
     *     - No -> ignore this object.
     *   - No -> the object is avoidance target no matter whether it is parking object or not.
     */
    const auto is_in_ego_lane =
      within(object_centroid, overhang_lanelet.polygon2d().basicPolygon());
    if (is_in_ego_lane) {
      /**
       * TODO(Satoshi Ota) use intersection area
       * under the assumption that there is no parking vehicle inside intersection,
       * ignore all objects that is in the ego lane as not parking objects.
       */
      std::string turn_direction = overhang_lanelet.attributeOr("turn_direction", "else");
      if (turn_direction == "right" || turn_direction == "left" || turn_direction == "straight") {
        avoidance_debug_array_false_and_push_back(
          AvoidanceDebugFactor::NOT_PARKING_OBJECT, object_data);
        continue;
      }

      const auto centerline_pose =
        lanelet::utils::getClosestCenterPose(object_closest_lanelet, object_pose.position);
      lanelet::BasicPoint3d centerline_point(
        centerline_pose.position.x, centerline_pose.position.y, centerline_pose.position.z);

      // ============================================ <- most_left_lanelet.leftBound()
      // y              road shouler
      // ^ ------------------------------------------
      // |   x                                +
      // +---> --- object closest lanelet --- o ----- <- object_closest_lanelet.centerline()
      //
      // --------------------------------------------
      // +: object position
      // o: nearest point on centerline

      const auto most_left_lanelet = rh->getMostLeftLanelet(object_closest_lanelet);
      const auto center_to_left_boundary =
        distance2d(to2D(most_left_lanelet.leftBound().basicLineString()), to2D(centerline_point));
      double object_shiftable_distance = center_to_left_boundary - 0.5 * object.shape.dimensions.y;

      const lanelet::Attribute sub_type =
        most_left_lanelet.attribute(lanelet::AttributeName::Subtype);
      if (sub_type.value() != "road_shoulder") {
        object_shiftable_distance += parameters_->minimum_road_shoulder_width;
      }

      const auto arc_coordinates = toArcCoordinates(
        to2D(object_closest_lanelet.centerline().basicLineString()), object_centroid);
      object_data.offset_ratio = arc_coordinates.distance / object_shiftable_distance;

      const auto is_parking_object =
        object_data.offset_ratio > parameters_->object_check_road_shoulder_ratio;
      if (!is_parking_object) {
        avoidance_debug_array_false_and_push_back(
          AvoidanceDebugFactor::NOT_PARKING_OBJECT, object_data);
        continue;
      }
    }

    object_data.last_seen = clock_->now();

    // set data
    avoidance_data.target_objects.push_back(object_data);
  }

  // debug
  {
    updateAvoidanceDebugData(avoidance_debug_msg_array);
    debug_avoidance_msg_array_ptr_ =
      std::make_shared<AvoidanceDebugMsgArray>(debug_data_.avoidance_debug_msg_array);
    debug.farthest_linestring_from_overhang =
      std::make_shared<lanelet::ConstLineStrings3d>(debug_linestring);
    debug.current_lanelets =
      std::make_shared<lanelet::ConstLanelets>(avoidance_data.current_lanelets);
    debug.expanded_lanelets = std::make_shared<lanelet::ConstLanelets>(expanded_lanelets);
  }
}

/**
 * updateRegisteredRawShiftPoints
 *
 *  - update path index of the registered objects
 *  - remove old objects whose end point is behind ego pose.
 */
void AvoidanceModule::updateRegisteredRawShiftPoints()
{
  fillAdditionalInfoFromPoint(registered_raw_shift_points_);

  AvoidPointArray avoid_points;
  const int margin = 0;
  const auto deadline = static_cast<size_t>(
    std::max(static_cast<int>(avoidance_data_.ego_closest_path_index) - margin, 0));

  for (const auto & ap : registered_raw_shift_points_) {
    if (ap.end_idx > deadline) {
      avoid_points.push_back(ap);
    }
  }

  DEBUG_PRINT(
    "ego_closest_path_index = %lu, registered_raw_shift_points_ size: %lu -> %lu",
    avoidance_data_.ego_closest_path_index, registered_raw_shift_points_.size(),
    avoid_points.size());

  printShiftPoints(registered_raw_shift_points_, "registered_raw_shift_points_ (before)");
  printShiftPoints(avoid_points, "registered_raw_shift_points_ (after)");

  registered_raw_shift_points_ = avoid_points;
  debug_data_.registered_raw_shift = registered_raw_shift_points_;
}

AvoidPointArray AvoidanceModule::applyPreProcessToRawShiftPoints(
  AvoidPointArray & current_raw_shift_points, DebugData & debug) const
{
  stop_watch_.tic(__func__);

  /**
   * Use all registered points. For the current points, if the similar one of the current
   * points are already registered, will not use it.
   * TODO(Horibe): enrich this logic to be able to consider the removal of the registered
   *               shift, because it cannot handle the case like "we don't have to avoid
   *               the object anymore".
   */
  auto total_raw_shift_points =
    combineRawShiftPointsWithUniqueCheck(registered_raw_shift_points_, current_raw_shift_points);

  printShiftPoints(current_raw_shift_points, "current_raw_shift_points");
  printShiftPoints(registered_raw_shift_points_, "registered_raw_shift_points");
  printShiftPoints(total_raw_shift_points, "total_raw_shift_points");

  /*
   * Add return-to-center shift point from the last shift point, if needed.
   * If there is no shift points, set return-to center shift from ego.
   */
  // TODO(Horibe) Here, the return point is calculated considering the prepare distance,
  // but there is an issue that sometimes this prepare distance is erased by the trimSimilarGrad,
  // and it suddenly tries to return from ego. Then steer rotates aggressively.
  // It is temporally solved by changing the threshold of trimSimilarGrad, but it needs to be
  // fixed in a proper way.
  // Maybe after merge, all shift points before the prepare distance can be deleted.
  addReturnShiftPointFromEgo(total_raw_shift_points, current_raw_shift_points);
  printShiftPoints(total_raw_shift_points, "total_raw_shift_points_with_extra_return_shift");

  /**
   * On each path point, compute shift length with considering the raw shift points.
   * Then create a merged shift points by finding the change point of the gradient of shifting.
   *  - take maximum shift length if there is duplicate shift point
   *  - take sum if there are shifts for opposite direction (right and left)
   *  - shift length is interpolated linearly.
   * Note: Because this function just foolishly extracts points, it includes
   *       insignificant small (useless) shift points, which should be removed in post-process.
   */
  auto merged_shift_points = mergeShiftPoints(total_raw_shift_points, debug);
  debug.merged = merged_shift_points;

  /*
   * Remove unnecessary shift points
   *  - Quantize the shift length to reduce the shift point noise
   *  - Change the shift length to the previous one if the deviation is small.
   *  - Combine shift points that have almost same gradient
   *  - Remove unnecessary return shift (back to the center line).
   */
  auto shift_points = trimShiftPoint(merged_shift_points, debug);
  DEBUG_PRINT("final shift point size = %lu", shift_points.size());

  RCLCPP_INFO_EXPRESSION(
    getLogger(), parameters_->print_processing_time, "- %s: %f ms", __func__,
    stop_watch_.toc(__func__, true));

  return shift_points;
}

void AvoidanceModule::registerRawShiftPoints(const AvoidPointArray & future)
{
  if (future.empty()) {
    RCLCPP_ERROR(getLogger(), "future is empty! return.");
    return;
  }

  const auto old_size = registered_raw_shift_points_.size();

  const auto future_with_info = fillAdditionalInfo(future);
  printShiftPoints(future_with_info, "future_with_info");
  printShiftPoints(registered_raw_shift_points_, "registered_raw_shift_points_");
  printShiftPoints(current_raw_shift_points_, "current_raw_shift_points_");

  const auto isAlreadyRegistered = [this](const auto id) {
    const auto & r = registered_raw_shift_points_;
    return std::any_of(r.begin(), r.end(), [id](const auto & r_sp) { return r_sp.id == id; });
  };

  const auto getAvoidPointByID = [this](const auto id) {
    for (const auto & sp : current_raw_shift_points_) {
      if (sp.id == id) {
        return sp;
      }
    }
    return AvoidPoint{};
  };

  for (const auto & ap : future_with_info) {
    if (ap.parent_ids.empty()) {
      RCLCPP_ERROR(getLogger(), "avoid point for path_shifter must have parent_id.");
    }
    for (const auto parent_id : ap.parent_ids) {
      if (!isAlreadyRegistered(parent_id)) {
        registered_raw_shift_points_.push_back(getAvoidPointByID(parent_id));
      }
    }
  }

  DEBUG_PRINT("registered object size: %lu -> %lu", old_size, registered_raw_shift_points_.size());
}

double AvoidanceModule::getShiftLength(
  const ObjectData & object, const bool & is_object_on_right, const double & avoid_margin) const
{
  const auto shift_length =
    behavior_path_planner::calcShiftLength(is_object_on_right, object.overhang_dist, avoid_margin);
  return is_object_on_right ? std::min(shift_length, getLeftShiftBound())
                            : std::max(shift_length, getRightShiftBound());
}

double AvoidanceModule::getShiftLength(
  const ObjectData & object, const double avoidance_velocity) const
{
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;

  const auto variable = getLateralMarginFromVelocity(std::abs(avoidance_velocity));
  const auto constant = parameters_->lateral_collision_safety_buffer + 0.5 * vehicle_width;

  const auto shift_length = getShiftLength(object, isOnRight(object), variable + constant);
  const auto ego_shift = getCurrentShift();

  return shift_length - ego_shift;
}

/**
 * calcRawShiftPointsFromObjects
 *
 * Calculate the shift points (start/end point, shift length) from the object lateral
 * and longitudinal positions in the Frenet coordinate. The jerk limit is also considered here.
 */
AvoidPointArray AvoidanceModule::calcRawShiftPointsFromObjects(
  const double avoidance_velocity, const AvoidancePlanningData & data, DebugData & debug) const
{
  stop_watch_.tic(__func__);

  const auto & p = parameters_;

  {
    debug_avoidance_initializer_for_shift_point_.clear();
    debug.unavoidable_objects.clear();
  }

  const auto prepare_distance = getNominalPrepareDistance();

  // To be consistent with changes in the ego position, the current shift length is considered.
  const auto current_ego_shift = getCurrentShift();
  // implement lane detection here.

  // buffer params
  const auto & lon_collision_safety_buffer_front = p->longitudinal_collision_safety_buffer_front;
  const auto & lon_collision_safety_buffer_back = p->longitudinal_collision_safety_buffer_back;
  const auto & lat_collision_safety_buffer = p->lateral_collision_safety_buffer;
  const auto & road_shoulder_safety_margin = p->road_shoulder_safety_margin;

  // vehicle params
  const auto & vehicle_width = planner_data_->parameters.vehicle_width;
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto & base_link2rear = planner_data_->parameters.base_link2rear;

  AvoidPointArray avoid_points;
  std::vector<AvoidanceDebugMsg> avoidance_debug_msg_array;
  avoidance_debug_msg_array.reserve(data.target_objects.size());
  for (auto o : data.target_objects) {
    AvoidanceDebugMsg avoidance_debug_msg;
    const auto avoidance_debug_array_false_and_push_back =
      [&avoidance_debug_msg, &avoidance_debug_msg_array](const std::string & failed_reason) {
        avoidance_debug_msg.allow_avoidance = false;
        avoidance_debug_msg.failed_reason = failed_reason;
        avoidance_debug_msg_array.push_back(avoidance_debug_msg);
      };

    const auto max_allowable_lateral_distance =
      o.to_road_shoulder_distance - road_shoulder_safety_margin - 0.5 * vehicle_width;

    avoidance_debug_msg.object_id = getUuidStr(o);
    avoidance_debug_msg.longitudinal_distance = o.longitudinal;
    avoidance_debug_msg.lateral_distance_from_centerline = o.lateral;
    avoidance_debug_msg.to_furthest_linestring_distance = o.to_road_shoulder_distance;
    avoidance_debug_msg.max_shift_length = max_allowable_lateral_distance;

    // CALCULATE SHIFT LENGTH
    //
    // leftbound          centerline         rightbound
    // |                      ^ x                     |
    // |   +-------+          |   D5                  |
    // |   |       |<-------------------------------->|
    // |   |       |   D1     |                       |
    // |   |  obj  |<-------->|                       |
    // |   |       |   D2     |   D3    +-------+     |
    // |   |       |<-----><----------->|       |     |
    // |   +-------+          |   D4    |       |  D6 |
    // |           |          |<------->|  ego  |<--->|
    // |           |          |         |       |     |
    // |           |          |         |       |     |
    // |           |          |   D8    +-------+     |
    // |           |<---------------------->|<->|     |
    // |                      |               D7      |
    // |          y  <--------+                       |
    //
    // D1: overhang_dist (signed value)
    // D2: lat_collision_safety_buffer (unsigned value)
    // D3: lateral_margin (unsigned CHANGEABLE value)
    // D2 + D3 + D7: lateral_avoid_margin (unsigned value)
    // D4 + D7: shift_length (signed value)
    // D5: to_road_shoulder_distance (unsigned value)
    // D6: road_shoulder_safety_margin (unsigned value)
    // D7: vehicle_width (unsigned value)
    // D8: max_allowable_lateral_distance (unsigned value)

    const auto lateral_avoid_margin = getLateralMarginFromVelocity(std::abs(avoidance_velocity)) +
                                      lat_collision_safety_buffer + 0.5 * vehicle_width;

    if (max_allowable_lateral_distance <= lateral_avoid_margin) {
      avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::INSUFFICIENT_LATERAL_MARGIN);
      o.reason = AvoidanceDebugFactor::INSUFFICIENT_LATERAL_MARGIN;
      debug.unavoidable_objects.push_back(o);
      continue;
    }

    const auto is_object_on_right = isOnRight(o);
    const auto shift_length = getShiftLength(o, is_object_on_right, lateral_avoid_margin);
    if (isSameDirectionShift(is_object_on_right, shift_length)) {
      avoidance_debug_array_false_and_push_back("IgnoreSameDirectionShift");
      o.reason = "IgnoreSameDirectionShift";
      debug.unavoidable_objects.push_back(o);
      continue;
    }

    const auto avoiding_shift = shift_length - current_ego_shift;
    const auto return_shift = shift_length;

    // CALCULATE START/END LONGITUDINAL
    //
    // ================================= leftbound ========================================
    //
    //     |<--------------------------------------------------------------------------->|
    //     |<----------------------------------------->|              D10                |
    //   D7| D8                 D9                     |                                 |
    //  |<>|<-->|                                      |                        end    path
    //  +-------+             start(avoid)             +-----+               (return)   end
    //  |  ego  |        x<---->x                      | obj |                   x       x
    //  +-------+           D1  |                      +-----+                   |
    //                          |                                                |
    //                          |      D2          D3     D4   D5        D6      |
    //                          |<------------>x<----->x<--->x<-->x<------------>|
    //                                        end(avoid)        start(return)
    //
    // ================================= rightbound =======================================
    //
    // D1: prepare_distance
    // D2: avoidance_distance[avoid]
    // D3: longitudinal_avoid_margin_front
    // D4: object.length
    // D5: longitudinal_avoid_margin_back
    // D6: avoiding_distance[return]
    // D7: base_link2rear
    // D8: base_link2front
    // D9: object.longitudinal
    // D10: arclength_from_ego.back()
    // D9 - D3 - D2: start_longitudinal[avoid]
    // D9 - D3: end_longitudinal[avoid]
    // D9 - D1 - D3: remaining_distance[avoid]
    // D9 + D4 + D5: start_longitudinal[return]
    // D9 + D4 + D5 + D6: end_longitudinal[return]
    // D10 - D5 - end_margin(1.0m): remaining_distance[return]

    const auto longitudinal_avoid_margin_front =
      lon_collision_safety_buffer_front + base_link2front;
    const auto longitudinal_avoid_margin_back = lon_collision_safety_buffer_back + base_link2rear;

    // use absolute dist for return-to-center, relative dist from current for avoiding.
    const auto nominal_avoid_distance = getNominalAvoidanceDistance(avoiding_shift);
    // const auto nominal_avoid_distance =
    //   getNominalAvoidanceDistance(avoiding_shift, avoidance_velocity);
    const auto nominal_return_distance = getNominalAvoidanceDistance(return_shift);

    /**
     * Is there enough distance from ego to object for avoidance?
     *   - Yes -> use the nominal distance.
     *   - No -> check if it is possible to avoid within maximum jerk limit.
     *     - Yes -> use the stronger jerk.
     *     - No -> ignore this object. Expected behavior is that the vehicle will stop in front
     *             of the obstacle, then start avoidance.
     */
    const bool has_enough_distance = o.longitudinal > (prepare_distance + nominal_avoid_distance +
                                                       longitudinal_avoid_margin_front);
    const auto remaining_distance =
      o.longitudinal - prepare_distance - longitudinal_avoid_margin_front;
    if (!has_enough_distance) {
      if (remaining_distance <= 0.0) {
        // TODO(Horibe) Even if there is no enough distance for avoidance shift, the
        // return-to-center shift must be considered for each object if the current_shift
        // is not zero.
        avoidance_debug_array_false_and_push_back(
          AvoidanceDebugFactor::REMAINING_DISTANCE_LESS_THAN_ZERO);
        if (!data.avoiding_now) {
          o.reason = AvoidanceDebugFactor::REMAINING_DISTANCE_LESS_THAN_ZERO;
          debug.unavoidable_objects.push_back(o);
        }
        continue;
      }

      // This is the case of exceeding the jerk limit. Use the sharp avoidance ego speed.
      const auto required_jerk = path_shifter_.calcJerkFromLatLonDistance(
        avoiding_shift, remaining_distance, getSharpAvoidanceEgoSpeed());
      avoidance_debug_msg.required_jerk = required_jerk;
      avoidance_debug_msg.maximum_jerk = p->max_lateral_jerk;
      if (required_jerk > p->max_lateral_jerk) {
        avoidance_debug_array_false_and_push_back(AvoidanceDebugFactor::TOO_LARGE_JERK);
        if (!data.avoiding_now) {
          o.reason = AvoidanceDebugFactor::TOO_LARGE_JERK;
          debug.unavoidable_objects.push_back(o);
        }
        continue;
      }
    }
    const auto avoiding_distance =
      has_enough_distance ? nominal_avoid_distance : remaining_distance;

    DEBUG_PRINT(
      "nominal_lateral_jerk = %f, getNominalAvoidanceEgoSpeed() = %f, prepare_distance = %f, "
      "has_enough_distance = %d",
      p->nominal_lateral_jerk, getNominalAvoidanceEgoSpeed(), prepare_distance,
      has_enough_distance);

    AvoidPoint ap_avoid;
    ap_avoid.length = shift_length;
    ap_avoid.start_length = current_ego_shift;
    ap_avoid.end_longitudinal = o.longitudinal - longitudinal_avoid_margin_front;
    ap_avoid.start_longitudinal =
      o.longitudinal - avoiding_distance - longitudinal_avoid_margin_front;
    ap_avoid.id = getOriginalShiftPointUniqueId();
    ap_avoid.object = o;
    avoid_points.push_back(ap_avoid);

    // The end_margin also has the purpose of preventing the return path from NOT being
    // triggered at the end point.
    const auto end_margin = 1.0;
    const auto return_remaining_distance =
      std::max(avoidance_data_.arclength_from_ego.back() - o.longitudinal - end_margin, 0.0);

    AvoidPoint ap_return;
    ap_return.length = 0.0;
    ap_return.start_length = shift_length;
    ap_return.start_longitudinal = o.longitudinal + o.length + longitudinal_avoid_margin_back;
    ap_return.end_longitudinal = o.longitudinal + o.length + longitudinal_avoid_margin_back +
                                 std::min(nominal_return_distance, return_remaining_distance);
    ap_return.id = getOriginalShiftPointUniqueId();
    ap_return.object = o;
    avoid_points.push_back(ap_return);

    DEBUG_PRINT(
      "object is set: avoid_shift = %f, return_shift = %f, dist = (avoidStart: %3.3f, avoidEnd: "
      "%3.3f, returnEnd: %3.3f), avoiding_dist = (nom:%f, res:%f), lateral_avoid_margin = %f, "
      "return_dist "
      "= %f",
      avoiding_shift, return_shift, ap_avoid.start_longitudinal, ap_avoid.end_longitudinal,
      ap_return.end_longitudinal, nominal_avoid_distance, avoiding_distance, lateral_avoid_margin,
      nominal_return_distance);
    avoidance_debug_msg.allow_avoidance = true;
    avoidance_debug_msg_array.push_back(avoidance_debug_msg);
  }

  debug_avoidance_initializer_for_shift_point_ = std::move(avoidance_debug_msg_array);
  debug_avoidance_initializer_for_shift_point_time_ = clock_->now();
  fillAdditionalInfoFromLongitudinal(avoid_points);

  RCLCPP_INFO_EXPRESSION(
    getLogger(), p->print_processing_time, "- %s: %f ms", __func__,
    stop_watch_.toc(__func__, true));

  return avoid_points;
}

AvoidPointArray AvoidanceModule::fillAdditionalInfo(const AvoidPointArray & shift_points) const
{
  if (shift_points.empty()) {
    return shift_points;
  }

  auto out_points = shift_points;

  const auto & path = avoidance_data_.reference_path;
  const auto arclength = avoidance_data_.arclength_from_ego;

  // calc longitudinal
  for (auto & sp : out_points) {
    sp.start_idx = findNearestIndex(path.points, sp.start.position);
    sp.start_longitudinal = arclength.at(sp.start_idx);
    sp.end_idx = findNearestIndex(path.points, sp.end.position);
    sp.end_longitudinal = arclength.at(sp.end_idx);
  }

  // sort by longitudinal
  std::sort(out_points.begin(), out_points.end(), [](auto a, auto b) {
    return a.end_longitudinal < b.end_longitudinal;
  });

  // calc relative lateral length
  out_points.front().start_length = getCurrentBaseShift();
  for (size_t i = 1; i < shift_points.size(); ++i) {
    out_points.at(i).start_length = shift_points.at(i - 1).length;
  }

  return out_points;
}
AvoidPoint AvoidanceModule::fillAdditionalInfo(const AvoidPoint & shift_point) const
{
  const auto ret = fillAdditionalInfo(AvoidPointArray{shift_point});
  return ret.front();
}

void AvoidanceModule::fillAdditionalInfoFromPoint(AvoidPointArray & shift_points) const
{
  if (shift_points.empty()) {
    return;
  }

  const auto & path = avoidance_data_.reference_path;
  const auto arclength = util::calcPathArcLengthArray(path);
  const auto dist_path_front_to_ego =
    calcSignedArcLength(path.points, 0, avoidance_data_.ego_closest_path_index);

  // calc longitudinal
  for (auto & sp : shift_points) {
    sp.start_idx = findNearestIndex(path.points, sp.start.position);
    sp.start_longitudinal = arclength.at(sp.start_idx) - dist_path_front_to_ego;
    sp.end_idx = findNearestIndex(path.points, sp.end.position);
    sp.end_longitudinal = arclength.at(sp.end_idx) - dist_path_front_to_ego;
  }
}

void AvoidanceModule::fillAdditionalInfoFromLongitudinal(AvoidPointArray & shift_points) const
{
  const auto & path = avoidance_data_.reference_path;
  const auto arclength = util::calcPathArcLengthArray(path);
  const auto path_front_to_ego =
    calcSignedArcLength(path.points, 0, avoidance_data_.ego_closest_path_index);

  for (auto & sp : shift_points) {
    sp.start_idx = findPathIndexFromArclength(arclength, sp.start_longitudinal + path_front_to_ego);
    sp.start = path.points.at(sp.start_idx).point.pose;
    sp.end_idx = findPathIndexFromArclength(arclength, sp.end_longitudinal + path_front_to_ego);
    sp.end = path.points.at(sp.end_idx).point.pose;
  }
}
/*
 * combineRawShiftPointsWithUniqueCheck
 *
 * Combine points A into B. If shift_point of A which has same object_id and
 * similar shape is already in B, it will not be added into B.
 */
AvoidPointArray AvoidanceModule::combineRawShiftPointsWithUniqueCheck(
  const AvoidPointArray & base_points, const AvoidPointArray & added_points) const
{
  // TODO(Horibe) parametrize
  const auto isSimilar = [](const AvoidPoint & a, const AvoidPoint & b) {
    using tier4_autoware_utils::calcDistance2d;
    if (calcDistance2d(a.start, b.start) > 1.0) {
      return false;
    }
    if (calcDistance2d(a.end, b.end) > 1.0) {
      return false;
    }
    if (std::abs(a.length - b.length) > 0.5) {
      return false;
    }
    return true;
  };
  const auto hasSameObjectId = [](const auto & a, const auto & b) {
    return a.object.object.object_id == b.object.object.object_id;
  };

  auto combined = base_points;  // initialized
  for (const auto & o : added_points) {
    bool skip = false;

    for (const auto & b : base_points) {
      if (hasSameObjectId(o, b) && isSimilar(o, b)) {
        skip = true;
        break;
      }
    }
    if (!skip) {
      combined.push_back(o);
    }
  }

  return combined;
}

void AvoidanceModule::generateTotalShiftLine(
  const AvoidPointArray & avoid_points, ShiftLineData & shift_line_data) const
{
  const auto & path = avoidance_data_.reference_path;
  const auto & arcs = avoidance_data_.arclength_from_ego;
  const auto N = path.points.size();

  auto & sl = shift_line_data;

  sl.shift_line = std::vector<double>(N, 0.0);
  sl.shift_line_grad = std::vector<double>(N, 0.0);

  sl.pos_shift_line = std::vector<double>(N, 0.0);
  sl.neg_shift_line = std::vector<double>(N, 0.0);

  sl.pos_shift_line_grad = std::vector<double>(N, 0.0);
  sl.neg_shift_line_grad = std::vector<double>(N, 0.0);

  // debug
  sl.shift_line_history = std::vector<std::vector<double>>(avoid_points.size(), sl.shift_line);

  // take minmax for same directional shift length
  for (size_t j = 0; j < avoid_points.size(); ++j) {
    const auto & ap = avoid_points.at(j);
    for (size_t i = 0; i < N; ++i) {
      // calc current interpolated shift
      const auto i_shift = lerpShiftLengthOnArc(arcs.at(i), ap);

      // update maximum shift for positive direction
      if (i_shift > sl.pos_shift_line.at(i)) {
        sl.pos_shift_line.at(i) = i_shift;
        sl.pos_shift_line_grad.at(i) = ap.getGradient();
      }

      // update minumum shift for negative direction
      if (i_shift < sl.neg_shift_line.at(i)) {
        sl.neg_shift_line.at(i) = i_shift;
        sl.neg_shift_line_grad.at(i) = ap.getGradient();
      }

      // store for debug print
      sl.shift_line_history.at(j).at(i) = i_shift;
    }
  }

  // Merge shift length of opposite directions.
  for (size_t i = 0; i < N; ++i) {
    sl.shift_line.at(i) = sl.pos_shift_line.at(i) + sl.neg_shift_line.at(i);
    sl.shift_line_grad.at(i) = sl.pos_shift_line_grad.at(i) + sl.neg_shift_line_grad.at(i);
  }

  // overwrite shift with current_ego_shift until ego pose.
  for (size_t i = 0; i <= avoidance_data_.ego_closest_path_index; ++i) {
    sl.shift_line.at(i) = sl.shift_line.at(avoidance_data_.ego_closest_path_index);
    sl.shift_line_grad.at(i) = 0.0;
  }

  // If the shift point does not have an associated object,
  // use previous value.
  for (size_t i = 1; i < N; ++i) {
    bool has_object = false;
    for (const auto & ap : avoid_points) {
      if (ap.start_idx < i && i < ap.end_idx) {
        has_object = true;
        break;
      }
    }
    if (!has_object) {
      sl.shift_line.at(i) = sl.shift_line.at(i - 1);
    }
  }
  sl.shift_line_history.push_back(sl.shift_line);
}

AvoidPointArray AvoidanceModule::extractShiftPointsFromLine(ShiftLineData & shift_line_data) const
{
  const auto & path = avoidance_data_.reference_path;
  const auto & arcs = avoidance_data_.arclength_from_ego;
  const auto N = path.points.size();

  auto & sl = shift_line_data;

  const auto getBwdGrad = [&](const size_t i) {
    if (i == 0) {
      return sl.shift_line_grad.at(i);
    }
    const double ds = arcs.at(i) - arcs.at(i - 1);
    if (ds < 1.0e-5) {
      return sl.shift_line_grad.at(i);
    }  // use theoretical value when ds is too small.
    return (sl.shift_line.at(i) - sl.shift_line.at(i - 1)) / ds;
  };

  const auto getFwdGrad = [&](const size_t i) {
    if (i == arcs.size() - 1) {
      return sl.shift_line_grad.at(i);
    }
    const double ds = arcs.at(i + 1) - arcs.at(i);
    if (ds < 1.0e-5) {
      return sl.shift_line_grad.at(i);
    }  // use theoretical value when ds is too small.
    return (sl.shift_line.at(i + 1) - sl.shift_line.at(i)) / ds;
  };

  // calculate forward and backward gradient of the shift length.
  // This will be used for grad-change-point check.
  sl.forward_grad = std::vector<double>(N, 0.0);
  sl.backward_grad = std::vector<double>(N, 0.0);
  for (size_t i = 0; i < N - 1; ++i) {
    sl.forward_grad.at(i) = getFwdGrad(i);
    sl.backward_grad.at(i) = getBwdGrad(i);
  }

  AvoidPointArray merged_avoid_points;
  AvoidPoint ap{};
  bool found_first_start = false;
  constexpr auto CREATE_SHIFT_GRAD_THR = 0.001;
  constexpr auto IS_ALREADY_SHIFTING_THR = 0.001;
  for (size_t i = avoidance_data_.ego_closest_path_index; i < N - 1; ++i) {
    const auto & p = path.points.at(i).point.pose;
    const auto shift = sl.shift_line.at(i);

    // If the vehicle is already on the avoidance (checked by the first point has shift),
    // set a start point at the first path point.
    if (!found_first_start && std::abs(shift) > IS_ALREADY_SHIFTING_THR) {
      setStartData(ap, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      found_first_start = true;
      DEBUG_PRINT("shift (= %f) is not zero at i = %lu. set start shift here.", shift, i);
    }

    // find the point where the gradient of the shift is changed
    const bool set_shift_point_flag =
      std::abs(sl.forward_grad.at(i) - sl.backward_grad.at(i)) > CREATE_SHIFT_GRAD_THR;

    if (!set_shift_point_flag) {
      continue;
    }

    if (!found_first_start) {
      setStartData(ap, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      found_first_start = true;
      DEBUG_PRINT("grad change detected. start at i = %lu", i);
    } else {
      setEndData(ap, shift, p, i, arcs.at(i));
      ap.id = getOriginalShiftPointUniqueId();
      merged_avoid_points.push_back(ap);
      setStartData(ap, 0.0, p, i, arcs.at(i));  // start length is overwritten later.
      DEBUG_PRINT("end and start point found at i = %lu", i);
    }
  }
  return merged_avoid_points;
}

AvoidPointArray AvoidanceModule::mergeShiftPoints(
  const AvoidPointArray & raw_shift_points, DebugData & debug) const
{
  // Generate shift line by merging raw_shift_points.
  ShiftLineData shift_line_data;
  generateTotalShiftLine(raw_shift_points, shift_line_data);

  // Re-generate shift points by detecting gradient-change point of the shift line.
  auto merged_shift_points = extractShiftPointsFromLine(shift_line_data);

  // set parent id
  for (auto & ap : merged_shift_points) {
    ap.parent_ids = calcParentIds(raw_shift_points, ap);
  }

  // sort by distance from ego.
  alignShiftPointsOrder(merged_shift_points);

  // debug visualize
  {
    debug.pos_shift = shift_line_data.pos_shift_line;
    debug.neg_shift = shift_line_data.neg_shift_line;
    debug.total_shift = shift_line_data.shift_line;
  }

  // debug print
  {
    const auto & arc = avoidance_data_.arclength_from_ego;
    const auto & closest = avoidance_data_.ego_closest_path_index;
    const auto & sl = shift_line_data.shift_line;
    const auto & sg = shift_line_data.shift_line_grad;
    const auto & fg = shift_line_data.forward_grad;
    const auto & bg = shift_line_data.backward_grad;
    using std::setw;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "\n[idx, arc, shift (for each shift points, filtered | total), grad (ideal, bwd, fwd)]: "
          "closest = "
       << closest << ", raw_shift_points size = " << raw_shift_points.size() << std::endl;
    for (size_t i = 0; i < arc.size(); ++i) {
      ss << "i = " << i << " | arc: " << arc.at(i) << " | shift: (";
      for (const auto & p : shift_line_data.shift_line_history) {
        ss << setw(5) << p.at(i) << ", ";
      }
      ss << "| total: " << setw(5) << sl.at(i) << ") | grad: (" << sg.at(i) << ", " << fg.at(i)
         << ", " << bg.at(i) << ")" << std::endl;
    }
    DEBUG_PRINT("%s", ss.str().c_str());
  }

  printShiftPoints(merged_shift_points, "merged_shift_points");

  return merged_shift_points;
}

std::vector<size_t> AvoidanceModule::calcParentIds(
  const AvoidPointArray & parent_candidates, const AvoidPoint & child) const
{
  // Get the ID of the original AP whose transition area overlaps with the given AP,
  // and set it to the parent id.
  std::set<uint64_t> ids;
  for (const auto & ap : parent_candidates) {
    const auto p_s = ap.start_longitudinal;
    const auto p_e = ap.end_longitudinal;
    const auto has_overlap = !(p_e < child.start_longitudinal || child.end_longitudinal < p_s);

    if (!has_overlap) {
      continue;
    }

    // Id the shift is overlapped, insert the shift point. Additionally, the shift which refers
    // to the same object id (created by the same object) will be set.
    //
    // Why? : think that there are two shifts, avoiding and .
    // If you register only the avoiding shift, the return-to-center shift will not be generated
    // when you get too close to or over the obstacle. The return-shift can be handled with
    // addReturnShift(), but it maybe reasonable to register the return-to-center shift for the
    // object at the same time as registering the avoidance shift to remove the complexity of the
    // addReturnShift().
    for (const auto & ap_local : parent_candidates) {
      if (ap_local.object.object.object_id == ap.object.object.object_id) {
        ids.insert(ap_local.id);
      }
    }
  }
  return std::vector<size_t>(ids.begin(), ids.end());
}

/*
 * Remove unnecessary avoid points
 * - Combine avoid points that have almost same gradient
 * - Quantize the shift length to reduce the shift point noise
 * - Change the shift length to the previous one if the deviation is small.
 * - Remove unnecessary return shift (back to the center line).
 */
AvoidPointArray AvoidanceModule::trimShiftPoint(
  const AvoidPointArray & shift_points, DebugData & debug) const
{
  if (shift_points.empty()) {
    return shift_points;
  }

  AvoidPointArray sp_array_trimmed = shift_points;

  // sort shift points from front to back.
  alignShiftPointsOrder(sp_array_trimmed);

  // - Combine avoid points that have almost same gradient.
  // this is to remove the noise.
  {
    const auto CHANGE_SHIFT_THRESHOLD_FOR_NOISE = 0.1;
    trimSimilarGradShiftPoint(sp_array_trimmed, CHANGE_SHIFT_THRESHOLD_FOR_NOISE);
    debug.trim_similar_grad_shift = sp_array_trimmed;
    printShiftPoints(sp_array_trimmed, "after trim_similar_grad_shift");
  }

  // - Quantize the shift length to reduce the shift point noise
  // This is to remove the noise coming from detection accuracy, interpolation, resampling, etc.
  {
    // constexpr double QUANTIZATION_DISTANCE = 0.2;
    constexpr double QUANTIZATION_DISTANCE = 0.05;
    quantizeShiftPoint(sp_array_trimmed, QUANTIZATION_DISTANCE);
    printShiftPoints(sp_array_trimmed, "after sp_array_trimmed");
    debug.quantized = sp_array_trimmed;
  }

  // - Change the shift length to the previous one if the deviation is small.
  {
    // constexpr double SHIFT_DIFF_THRES = 0.5;
    // trimSmallShiftPoint(sp_array_trimmed, SHIFT_DIFF_THRES);
    debug.trim_small_shift = sp_array_trimmed;
    printShiftPoints(sp_array_trimmed, "after trim_small_shift");
  }

  // - Combine avoid points that have almost same gradient (again)
  {
    const auto CHANGE_SHIFT_THRESHOLD = 0.2;
    trimSimilarGradShiftPoint(sp_array_trimmed, CHANGE_SHIFT_THRESHOLD);
    debug.trim_similar_grad_shift_second = sp_array_trimmed;
    printShiftPoints(sp_array_trimmed, "after trim_similar_grad_shift_second");
  }

  // - trimTooSharpShift
  // Check if it is not too sharp for the return-to-center shift point.
  // If the shift is sharp, it is combined with the next shift point until it gets non-sharp.
  {
    trimSharpReturn(sp_array_trimmed);
    debug.trim_too_sharp_shift = sp_array_trimmed;
    printShiftPoints(sp_array_trimmed, "after trimSharpReturn");
  }

  return sp_array_trimmed;
}

void AvoidanceModule::alignShiftPointsOrder(
  AvoidPointArray & shift_points, const bool recalculate_start_length) const
{
  if (shift_points.empty()) {
    return;
  }

  // sort shift points from front to back.
  std::sort(shift_points.begin(), shift_points.end(), [](auto a, auto b) {
    return a.end_longitudinal < b.end_longitudinal;
  });

  // calc relative length
  // NOTE: the input shift point must not have conflict range. Otherwise relative
  // length value will be broken.
  if (recalculate_start_length) {
    shift_points.front().start_length = getCurrentLinearShift();
    for (size_t i = 1; i < shift_points.size(); ++i) {
      shift_points.at(i).start_length = shift_points.at(i - 1).length;
    }
  }
}

void AvoidanceModule::quantizeShiftPoint(
  AvoidPointArray & shift_points, const double interval) const
{
  if (interval < 1.0e-5) {
    return;  // no need to process
  }

  for (auto & sp : shift_points) {
    sp.length = std::round(sp.length / interval) * interval;
  }

  alignShiftPointsOrder(shift_points);
}

void AvoidanceModule::trimSmallShiftPoint(
  AvoidPointArray & shift_points, const double shift_diff_thres) const
{
  AvoidPointArray shift_points_orig = shift_points;
  shift_points.clear();

  shift_points.push_back(shift_points_orig.front());  // Take the first one anyway (think later)

  for (size_t i = 1; i < shift_points_orig.size(); ++i) {
    auto sp_now = shift_points_orig.at(i);
    const auto sp_prev = shift_points.back();
    const auto shift_diff = sp_now.length - sp_prev.length;

    auto sp_modified = sp_now;

    // remove the shift point if the length is almost same as the previous one.
    if (std::abs(shift_diff) < shift_diff_thres) {
      sp_modified.length = sp_prev.length;
      sp_modified.start_length = sp_prev.length;
      DEBUG_PRINT(
        "i = %lu, relative shift = %f is small. set with relative shift = 0.", i, shift_diff);
    } else {
      DEBUG_PRINT("i = %lu, shift = %f is large. take this one normally.", i, shift_diff);
    }

    shift_points.push_back(sp_modified);
  }

  alignShiftPointsOrder(shift_points);

  DEBUG_PRINT("size %lu -> %lu", shift_points_orig.size(), shift_points.size());
}

void AvoidanceModule::trimSimilarGradShiftPoint(
  AvoidPointArray & avoid_points, const double change_shift_dist_threshold) const
{
  AvoidPointArray avoid_points_orig = avoid_points;
  avoid_points.clear();

  avoid_points.push_back(avoid_points_orig.front());  // Take the first one anyway (think later)

  // Save the points being merged. When merging consecutively, also check previously merged points.
  AvoidPointArray being_merged_points;

  for (size_t i = 1; i < avoid_points_orig.size(); ++i) {
    const auto ap_now = avoid_points_orig.at(i);
    const auto ap_prev = avoid_points.back();

    being_merged_points.push_back(ap_prev);  // This point is about to be merged.

    auto combined_ap = ap_prev;
    setEndData(combined_ap, ap_now.length, ap_now.end, ap_now.end_idx, ap_now.end_longitudinal);
    combined_ap.parent_ids = concatParentIds(combined_ap.parent_ids, ap_prev.parent_ids);

    const auto has_large_length_change = [&]() {
      for (const auto & original : being_merged_points) {
        const auto longitudinal = original.end_longitudinal - combined_ap.start_longitudinal;
        const auto new_length = combined_ap.getGradient() * longitudinal + combined_ap.start_length;
        const bool has_large_change =
          std::abs(new_length - original.length) > change_shift_dist_threshold;

        DEBUG_PRINT(
          "original.length: %f, original.end_longitudinal: %f, combined_ap.start_longitudinal: "
          "%f, combined_ap.Gradient: %f, new_length: %f, has_large_change: %d",
          original.length, original.end_longitudinal, combined_ap.start_longitudinal,
          combined_ap.getGradient(), new_length, has_large_change);

        if (std::abs(new_length - original.length) > change_shift_dist_threshold) {
          return true;
        }
      }
      return false;
    }();

    if (has_large_length_change) {
      // If this point is merged with the previous points, it makes a large changes.
      // Do not merge this.
      avoid_points.push_back(ap_now);
      being_merged_points.clear();
      DEBUG_PRINT("use this point. has_large_length_change = %d", has_large_length_change);
    } else {
      avoid_points.back() = combined_ap;  // Update the last points by merging the current point
      being_merged_points.push_back(ap_prev);
      DEBUG_PRINT("trim! has_large_length_change = %d", has_large_length_change);
    }
  }

  alignShiftPointsOrder(avoid_points);

  DEBUG_PRINT("size %lu -> %lu", avoid_points_orig.size(), avoid_points.size());
}

/**
 * Remove short "return to center" shift point. ¯¯\_/¯¯　-> ¯¯¯¯¯¯
 *
 * Is the shift point for "return to center"?
 *  - no : Do not trim anything.
 *  - yes: Is it short distance enough to be removed?
 *     - no : Do not trim anything.
 *     - yes: Remove the "return" shift point.
 *            Recalculate longitudinal distance and modify the shift point.
 */
void AvoidanceModule::trimMomentaryReturn(AvoidPointArray & shift_points) const
{
  const auto isZero = [](double v) { return std::abs(v) < 1.0e-5; };

  AvoidPointArray shift_points_orig = shift_points;
  shift_points.clear();

  const double DISTANCE_AFTER_RETURN_THR = 5.0 * getNominalAvoidanceEgoSpeed();

  const auto & arclength = avoidance_data_.arclength_from_ego;

  const auto check_reduce_shift = [](const double now_length, const double prev_length) {
    const auto abs_shift_diff = std::abs(now_length) - std::abs(prev_length);
    const auto has_same_sign = (now_length * prev_length >= 0.0);
    const bool is_reduce_shift = (abs_shift_diff < 0.0 && has_same_sign);
    return is_reduce_shift;
  };

  for (size_t i = 0; i < shift_points_orig.size(); ++i) {
    const auto sp_now = shift_points_orig.at(i);
    const auto sp_prev_length =
      shift_points.empty() ? getCurrentLinearShift() : shift_points.back().length;
    const auto abs_shift_diff = std::abs(sp_now.length) - std::abs(sp_prev_length);
    const bool is_reduce_shift = check_reduce_shift(sp_now.length, sp_prev_length);

    // Do nothing for non-reduce shift point
    if (!is_reduce_shift) {
      shift_points.push_back(sp_now);
      DEBUG_PRINT(
        "i = %lu, not reduce shift. take this one.　abs_shift_diff = %f, sp_now.length = %f, "
        "sp_prev_length = %f, sp_now.length * sp_prev_length = %f",
        i, abs_shift_diff, sp_now.length, sp_prev_length, sp_now.length * sp_prev_length);
      continue;
    }

    // The last point is out of target of this function.
    const bool is_last_sp = (i == shift_points_orig.size() - 1);
    if (is_last_sp) {
      shift_points.push_back(sp_now);
      DEBUG_PRINT("i = %lu, last shift. take this one.", i);
      continue;
    }

    // --- From here, the shift point is "return to center" or "straight". ---
    // -----------------------------------------------------------------------

    const auto sp_next = shift_points_orig.at(i + 1);

    // there is no straight interval, combine them. ¯¯\/¯¯ -> ¯¯¯¯¯¯
    if (!isZero(sp_next.getRelativeLength())) {
      DEBUG_PRINT(
        "i = %lu, return-shift is detected, next shift_diff (%f) is nonzero. combine them. (skip "
        "next shift).",
        i, sp_next.getRelativeLength());
      auto sp_modified = sp_next;
      setStartData(
        sp_modified, sp_now.length, sp_now.start, sp_now.start_idx, sp_now.start_longitudinal);
      sp_modified.parent_ids = concatParentIds(sp_modified.parent_ids, sp_now.parent_ids);
      shift_points.push_back(sp_modified);
      ++i;  // skip next shift point
      continue;
    }

    // Find next shifting point, i.e.  ¯¯\____"/"¯¯
    //                               now ↑     ↑ target
    const auto next_avoid_idx = [&]() {
      for (size_t j = i + 1; j < shift_points_orig.size(); ++j) {
        if (!isZero(shift_points_orig.at(j).getRelativeLength())) {
          return j;
        }
      }
      return shift_points_orig.size();
    }();

    // The straight distance lasts until end. take this one.
    // ¯¯\______
    if (next_avoid_idx == shift_points_orig.size()) {
      shift_points.push_back(sp_now);
      DEBUG_PRINT("i = %lu, back -> straight lasts until end. take this one.", i);
      continue;
    }

    const auto sp_next_avoid = shift_points_orig.at(next_avoid_idx);
    const auto straight_distance = sp_next_avoid.start_longitudinal - sp_now.end_longitudinal;

    // The straight distance after "return to center" is long enough. take this one.
    // ¯¯\______/¯¯ (enough long straight line!)
    if (straight_distance > DISTANCE_AFTER_RETURN_THR) {
      shift_points.push_back(sp_now);
      DEBUG_PRINT("i = %lu, back -> straight: distance is long. take this one", i);
      continue;
    }

    // From here, back to center and go straight, straight distance is too short.
    // ¯¯\______/¯¯ (short straight line!)

    const auto relative_shift = sp_next_avoid.length - sp_now.length;
    const auto avoid_distance = getNominalAvoidanceDistance(relative_shift);

    // Calculate start point from end point and avoidance distance.
    auto sp_next_modified = sp_next_avoid;
    sp_next_modified.start_length = sp_prev_length;
    sp_next_modified.start_longitudinal =
      std::max(sp_next_avoid.end_longitudinal - avoid_distance, sp_now.start_longitudinal);
    sp_next_modified.start_idx =
      findPathIndexFromArclength(arclength, sp_next_modified.start_longitudinal);
    sp_next_modified.start =
      avoidance_data_.reference_path.points.at(sp_next_modified.start_idx).point.pose;
    sp_next_modified.parent_ids = calcParentIds(current_raw_shift_points_, sp_next_modified);

    // Straight shift point
    if (sp_next_modified.start_idx > sp_now.start_idx) {  // the case where a straight route exists.
      auto sp_now_modified = sp_now;
      sp_now_modified.start_length = sp_prev_length;
      setEndData(
        sp_now_modified, sp_prev_length, sp_next_modified.start, sp_next_modified.start_idx,
        sp_next_modified.start_longitudinal);
      sp_now_modified.parent_ids = calcParentIds(current_raw_shift_points_, sp_now_modified);
      shift_points.push_back(sp_now_modified);
    }
    shift_points.push_back(sp_next_modified);

    DEBUG_PRINT(
      "i = %lu, find remove target!: next_avoid_idx = %lu, shift length = (now: %f, prev: %f, "
      "next_avoid: %f, next_mod: %f).",
      i, next_avoid_idx, sp_now.length, sp_prev_length, sp_next_avoid.length,
      sp_next_modified.length);

    i = next_avoid_idx;  // skip shifting until next_avoid_idx.
  }

  alignShiftPointsOrder(shift_points);

  DEBUG_PRINT(
    "trimMomentaryReturn: size %lu -> %lu", shift_points_orig.size(), shift_points.size());
}

void AvoidanceModule::trimSharpReturn(AvoidPointArray & shift_points) const
{
  AvoidPointArray shift_points_orig = shift_points;
  shift_points.clear();

  const auto isZero = [](double v) { return std::abs(v) < 0.01; };

  // check if the shift point is positive (avoiding) shift
  const auto isPositive = [&](const auto & sp) {
    constexpr auto POSITIVE_SHIFT_THR = 0.1;
    return std::abs(sp.length) - std::abs(sp.start_length) > POSITIVE_SHIFT_THR;
  };

  // check if the shift point is negative (returning) shift
  const auto isNegative = [&](const auto & sp) {
    constexpr auto NEGATIVE_SHIFT_THR = -0.1;
    return std::abs(sp.length) - std::abs(sp.start_length) < NEGATIVE_SHIFT_THR;
  };

  // combine two shift points. Be careful the order of "now" and "next".
  const auto combineShiftPoint = [this](const auto & sp_next, const auto & sp_now) {
    auto sp_modified = sp_now;
    setEndData(sp_modified, sp_next.length, sp_next.end, sp_next.end_idx, sp_next.end_longitudinal);
    sp_modified.parent_ids = concatParentIds(sp_modified.parent_ids, sp_now.parent_ids);
    return sp_modified;
  };

  // Check if the merged shift has a conflict with the original shifts.
  const auto hasViolation = [this](const auto & combined, const auto & combined_src) {
    constexpr auto VIOLATION_SHIFT_THR = 0.3;
    for (const auto & sp : combined_src) {
      const auto combined_shift = lerpShiftLengthOnArc(sp.end_longitudinal, combined);
      // if (sp.length < -0.01 && combined_shift > sp.length + VIOLATION_SHIFT_THR) {
      //   return true;
      // }
      // if (sp.length > 0.01 && combined_shift < sp.length - VIOLATION_SHIFT_THR) {
      //   return true;
      // }
      if (combined.start_length < -0.01 && combined_shift > sp.length + VIOLATION_SHIFT_THR) {
        return true;
      }
      if (combined.start_length > 0.01 && combined_shift < sp.length - VIOLATION_SHIFT_THR) {
        return true;
      }
    }
    return false;
  };

  // check for all shift points
  for (size_t i = 0; i < shift_points_orig.size(); ++i) {
    auto sp_now = shift_points_orig.at(i);
    sp_now.start_length =
      shift_points.empty() ? getCurrentLinearShift() : shift_points.back().length;

    if (sp_now.length * sp_now.start_length < -0.01) {
      DEBUG_PRINT("i = %lu, This is avoid shift for opposite direction. take this one", i);
      continue;
    }

    // Do nothing for non-reduce shift point
    if (!isNegative(sp_now)) {
      shift_points.push_back(sp_now);
      DEBUG_PRINT(
        "i = %lu, positive shift. take this one. sp_now.length * sp_now.start_length = %f", i,
        sp_now.length * sp_now.start_length);
      continue;
    }

    // The last point is out of target of this function.
    if (i == shift_points_orig.size() - 1) {
      shift_points.push_back(sp_now);
      DEBUG_PRINT("i = %lu, last shift. take this one.", i);
      continue;
    }

    // -----------------------------------------------------------------------
    // ------------ From here, the shift point is "negative" -----------------
    // -----------------------------------------------------------------------

    // if next shift is negative, combine them. loop until combined shift line
    // exceeds merged shift point.
    DEBUG_PRINT("i = %lu, found negative dist. search.", i);
    {
      auto sp_combined = sp_now;
      auto sp_combined_prev = sp_combined;
      AvoidPointArray sp_combined_array{sp_now};
      size_t j = i + 1;
      for (; i < shift_points_orig.size(); ++j) {
        const auto sp_combined = combineShiftPoint(shift_points_orig.at(j), sp_now);

        {
          std::stringstream ss;
          ss << "i = " << i << ", j = " << j << ": sp_combined = " << toStrInfo(sp_combined);
          DEBUG_PRINT("%s", ss.str().c_str());
        }

        // it gets positive. Finish merging.
        if (isPositive(sp_combined)) {
          shift_points.push_back(sp_combined);
          DEBUG_PRINT("reach positive.");
          break;
        }

        // Still negative, but it violates the original shift points.
        // Finish with the previous merge result.
        if (hasViolation(sp_combined, sp_combined_array)) {
          shift_points.push_back(sp_combined_prev);
          DEBUG_PRINT("violation found.");
          --j;
          break;
        }

        // Still negative, but it has an enough long distance. Finish merging.
        const auto nominal_distance = getNominalAvoidanceDistance(sp_combined.getRelativeLength());
        const auto long_distance =
          isZero(sp_combined.length) ? nominal_distance : nominal_distance * 5.0;
        if (sp_combined.getRelativeLongitudinal() > long_distance) {
          shift_points.push_back(sp_combined);
          DEBUG_PRINT("still negative, but long enough. Threshold = %f", long_distance);
          break;
        }

        // It reaches the last point. Still the shift is sharp, but merge with the current result.
        if (j == shift_points_orig.size() - 1) {
          shift_points.push_back(sp_combined);
          DEBUG_PRINT("reach end point.");
          break;
        }

        // Still negative shift, and the distance is not enough. Search next.
        sp_combined_prev = sp_combined;
        sp_combined_array.push_back(shift_points_orig.at(j));
      }
      i = j;
      continue;
    }
  }

  alignShiftPointsOrder(shift_points);

  DEBUG_PRINT("trimSharpReturn: size %lu -> %lu", shift_points_orig.size(), shift_points.size());
}

void AvoidanceModule::trimTooSharpShift(AvoidPointArray & avoid_points) const
{
  if (avoid_points.empty()) {
    return;
  }

  AvoidPointArray avoid_points_orig = avoid_points;
  avoid_points.clear();

  const auto isInJerkLimit = [this](const auto & ap) {
    const auto required_jerk = path_shifter_.calcJerkFromLatLonDistance(
      ap.getRelativeLength(), ap.getRelativeLongitudinal(), getSharpAvoidanceEgoSpeed());
    return std::fabs(required_jerk) < parameters_->max_lateral_jerk;
  };

  for (size_t i = 0; i < avoid_points_orig.size(); ++i) {
    auto ap_now = avoid_points_orig.at(i);

    if (isInJerkLimit(ap_now)) {
      avoid_points.push_back(ap_now);
      continue;
    }

    DEBUG_PRINT("over jerk is detected: i = %lu", i);
    printShiftPoints(AvoidPointArray{ap_now}, "points with over jerk");

    // The avoidance_point_now exceeds jerk limit, so merge it with the next avoidance_point.
    for (size_t j = i + 1; j < avoid_points_orig.size(); ++j) {
      auto ap_next = avoid_points_orig.at(j);
      setEndData(ap_now, ap_next.length, ap_next.end, ap_next.end_idx, ap_next.end_longitudinal);
      if (isInJerkLimit(ap_now)) {
        avoid_points.push_back(ap_now);
        DEBUG_PRINT("merge finished. i = %lu, j = %lu", i, j);
        i = j;  // skip check until j index.
        break;
      }
    }
  }

  alignShiftPointsOrder(avoid_points);

  DEBUG_PRINT("size %lu -> %lu", avoid_points_orig.size(), avoid_points.size());
}

/*
 * addReturnShiftPoint
 *
 * Pick up the last shift point, which is the most farthest from ego, from the current candidate
 * avoidance points and registered points in the shifter. If the last shift length of the point is
 * non-zero, add a return-shift to center line from the point. If there is no shift point in
 * candidate avoidance points nor registered points, and base_shift > 0, add a return-shift to
 * center line from ego.
 */
void AvoidanceModule::addReturnShiftPointFromEgo(
  AvoidPointArray & sp_candidates, AvoidPointArray & current_raw_shift_points) const
{
  constexpr double ep = 1.0e-3;
  const bool has_candidate_point = !sp_candidates.empty();
  const bool has_registered_point = !path_shifter_.getShiftPoints().empty();

  // If the return-to-center shift points are already registered, do nothing.
  if (!has_registered_point && std::fabs(getCurrentBaseShift()) < ep) {
    DEBUG_PRINT("No shift points, not base offset. Do not have to add return-shift.");
    return;
  }

  constexpr double RETURN_SHIFT_THRESHOLD = 0.1;
  DEBUG_PRINT("registered last shift = %f", path_shifter_.getLastShiftLength());
  if (std::abs(path_shifter_.getLastShiftLength()) < RETURN_SHIFT_THRESHOLD) {
    DEBUG_PRINT("Return shift is already registered. do nothing.");
    return;
  }

  // From here, the return-to-center is not registered. But perhaps the candidate is
  // already generated.

  // If it has a shift point, add return shift from the existing last shift point.
  // If not, add return shift from ego point. (prepare distance is considered for both.)
  ShiftPoint last_sp;  // the return-shift will be generated after the last shift point.
  {
    // avoidance points: Yes, shift points: No -> select last avoidance point.
    if (has_candidate_point && !has_registered_point) {
      alignShiftPointsOrder(sp_candidates, false);
      last_sp = sp_candidates.back();
    }

    // avoidance points: No, shift points: Yes -> select last shift point.
    if (!has_candidate_point && has_registered_point) {
      last_sp = fillAdditionalInfo(AvoidPoint{path_shifter_.getLastShiftPoint().get()});
    }

    // avoidance points: Yes, shift points: Yes -> select the last one from both.
    if (has_candidate_point && has_registered_point) {
      alignShiftPointsOrder(sp_candidates, false);
      const auto & ap = sp_candidates.back();
      const auto & sp = fillAdditionalInfo(AvoidPoint{path_shifter_.getLastShiftPoint().get()});
      last_sp = (sp.end_longitudinal > ap.end_longitudinal) ? sp : ap;
    }

    // avoidance points: No, shift points: No -> set the ego position to the last shift point
    // so that the return-shift will be generated from ego position.
    if (!has_candidate_point && !has_registered_point) {
      last_sp.end = getEgoPose().pose;
      last_sp.end_idx = avoidance_data_.ego_closest_path_index;
      last_sp.length = getCurrentBaseShift();
    }
  }
  printShiftPoints(ShiftPointArray{last_sp}, "last shift point");

  // There already is a shift point candidates to go back to center line, but it could be too sharp
  // due to detection noise or timing.
  // Here the return-shift from ego is added for the in case.
  if (std::fabs(last_sp.length) < RETURN_SHIFT_THRESHOLD) {
    const auto current_base_shift = getCurrentShift();
    if (std::abs(current_base_shift) < ep) {
      DEBUG_PRINT("last shift almost is zero, and current base_shift is zero. do nothing.");
      return;
    }

    // Is there a shift point in the opposite direction of the current_base_shift?
    //   No  -> we can overwrite the return shift, because the other shift points that decrease
    //          the shift length are for return-shift.
    //   Yes -> we can NOT overwrite, because it might be not a return-shift, but a avoiding
    //          shift to the opposite direction which can not be overwritten by the return-shift.
    for (const auto & sp : sp_candidates) {
      if (
        (current_base_shift > 0.0 && sp.length < -ep) ||
        (current_base_shift < 0.0 && sp.length > ep)) {
        DEBUG_PRINT(
          "try to put overwrite return shift, but there is shift for opposite direction. Skip "
          "adding return shift.");
        return;
      }
    }

    // set the return-shift from ego.
    DEBUG_PRINT(
      "return shift already exists, but they are all candidates. Add return shift for overwrite.");
    last_sp.end = getEgoPose().pose;
    last_sp.end_idx = avoidance_data_.ego_closest_path_index;
    last_sp.length = current_base_shift;
  }

  const auto & arclength_from_ego = avoidance_data_.arclength_from_ego;

  const auto nominal_prepare_distance = getNominalPrepareDistance();
  const auto nominal_avoid_distance = getNominalAvoidanceDistance(last_sp.length);

  if (arclength_from_ego.empty()) {
    return;
  }

  const auto remaining_distance = arclength_from_ego.back();

  // If the avoidance point has already been set, the return shift must be set after the point.
  const auto last_sp_distance = avoidance_data_.arclength_from_ego.at(last_sp.end_idx);

  // check if there is enough distance for return.
  if (last_sp_distance + 1.0 > remaining_distance) {  // tmp: add some small number (+1.0)
    DEBUG_PRINT("No enough distance for return.");
    return;
  }

  // If the remaining distance is not enough, the return shift needs to be shrunk.
  // (or another option is just to ignore the return-shift.)
  // But we do not want to change the last shift point, so we will shrink the distance after
  // the last shift point.
  //
  //  The line "===" is fixed, "---" is scaled.
  //
  // [Before Scaling]
  //  ego              last_sp_end             prepare_end            path_end    avoid_end
  // ==o====================o----------------------o----------------------o------------o
  //   |            prepare_dist                   |          avoid_dist               |
  //
  // [After Scaling]
  // ==o====================o------------------o--------------------------o
  //   |        prepare_dist_scaled            |    avoid_dist_scaled     |
  //
  const double variable_prepare_distance =
    std::max(nominal_prepare_distance - last_sp_distance, 0.0);

  double prepare_distance_scaled = std::max(nominal_prepare_distance, last_sp_distance);
  double avoid_distance_scaled = nominal_avoid_distance;
  if (remaining_distance < prepare_distance_scaled + avoid_distance_scaled) {
    const auto scale = (remaining_distance - last_sp_distance) /
                       std::max(nominal_avoid_distance + variable_prepare_distance, 0.1);
    prepare_distance_scaled = last_sp_distance + scale * nominal_prepare_distance;
    avoid_distance_scaled *= scale;
    DEBUG_PRINT(
      "last_sp_distance = %f, nominal_prepare_distance = %f, nominal_avoid_distance = %f, "
      "remaining_distance = %f, variable_prepare_distance = %f, scale = %f, "
      "prepare_distance_scaled = %f,avoid_distance_scaled = %f",
      last_sp_distance, nominal_prepare_distance, nominal_avoid_distance, remaining_distance,
      variable_prepare_distance, scale, prepare_distance_scaled, avoid_distance_scaled);
  } else {
    DEBUG_PRINT("there is enough distance. Use nominal for prepare & avoidance.");
  }

  // shift point for prepare distance: from last shift to return-start point.
  if (nominal_prepare_distance > last_sp_distance) {
    AvoidPoint ap;
    ap.id = getOriginalShiftPointUniqueId();
    ap.start_idx = last_sp.end_idx;
    ap.start = last_sp.end;
    ap.start_longitudinal = arclength_from_ego.at(ap.start_idx);
    ap.end_idx = findPathIndexFromArclength(arclength_from_ego, prepare_distance_scaled);
    ap.end = avoidance_data_.reference_path.points.at(ap.end_idx).point.pose;
    ap.end_longitudinal = arclength_from_ego.at(ap.end_idx);
    ap.length = last_sp.length;
    ap.start_length = last_sp.length;
    sp_candidates.push_back(ap);
    printShiftPoints(AvoidPointArray{ap}, "prepare for return");
    debug_data_.extra_return_shift.push_back(ap);

    // TODO(Horibe) think how to store the current object
    current_raw_shift_points.push_back(ap);
  }

  // shift point for return to center line
  {
    AvoidPoint ap;
    ap.id = getOriginalShiftPointUniqueId();
    ap.start_idx = findPathIndexFromArclength(arclength_from_ego, prepare_distance_scaled);
    ap.start = avoidance_data_.reference_path.points.at(ap.start_idx).point.pose;
    ap.start_longitudinal = arclength_from_ego.at(ap.start_idx);
    ap.end_idx = findPathIndexFromArclength(
      arclength_from_ego, prepare_distance_scaled + avoid_distance_scaled);
    ap.end = avoidance_data_.reference_path.points.at(ap.end_idx).point.pose;
    ap.end_longitudinal = arclength_from_ego.at(ap.end_idx);
    ap.length = 0.0;
    ap.start_length = last_sp.length;
    sp_candidates.push_back(ap);
    printShiftPoints(AvoidPointArray{ap}, "return point");
    debug_data_.extra_return_shift = AvoidPointArray{ap};

    // TODO(Horibe) think how to store the current object
    current_raw_shift_points.push_back(ap);
  }

  DEBUG_PRINT("Return Shift is added.");
}

double AvoidanceModule::getRightShiftBound() const
{
  // TODO(Horibe) write me. Real lane boundary must be considered here.
  return -parameters_->max_right_shift_length;
}

double AvoidanceModule::getLeftShiftBound() const
{
  // TODO(Horibe) write me. Real lane boundary must be considered here.
  return parameters_->max_left_shift_length;
}

// TODO(murooka) judge when and which way to extend drivable area. current implementation is keep
// extending during avoidance module
// TODO(murooka) freespace during turning in intersection where there is no neighbour lanes
// NOTE: Assume that there is no situation where there is an object in the middle lane of more than
// two lanes since which way to avoid is not obvious
void AvoidanceModule::generateExtendedDrivableArea(ShiftedPath * shifted_path) const
{
  const auto & route_handler = planner_data_->route_handler;
  const auto & current_lanes = avoidance_data_.current_lanelets;
  lanelet::ConstLanelets extended_lanelets = current_lanes;

  for (const auto & current_lane : current_lanes) {
    if (!parameters_->enable_avoidance_over_opposite_direction) {
      break;
    }

    const auto extend_from_current_lane = std::invoke(
      [this, &route_handler](const lanelet::ConstLanelet & lane) {
        const auto ignore_opposite = !parameters_->enable_avoidance_over_opposite_direction;
        if (ignore_opposite) {
          return route_handler->getAllSharedLineStringLanelets(lane, true, true, ignore_opposite);
        }

        return route_handler->getAllSharedLineStringLanelets(lane);
      },
      current_lane);
    extended_lanelets.reserve(extended_lanelets.size() + extend_from_current_lane.size());
    extended_lanelets.insert(
      extended_lanelets.end(), extend_from_current_lane.begin(), extend_from_current_lane.end());

    // 2. when there are multiple turning lanes whose previous lanelet is the same in
    // intersection
    const lanelet::ConstLanelets next_lanes_from_intersection = std::invoke(
      [&route_handler](const lanelet::ConstLanelet & lane) {
        if (!lane.hasAttribute("turn_direction")) {
          return lanelet::ConstLanelets{};
        }

        // get previous lane, and return false if previous lane does not exist
        lanelet::ConstLanelets prev_lanes;
        if (!route_handler->getPreviousLaneletsWithinRoute(lane, &prev_lanes)) {
          return lanelet::ConstLanelets{};
        }

        lanelet::ConstLanelets next_lanes;
        for (const auto & prev_lane : prev_lanes) {
          const auto next_lanes_from_prev = route_handler->getNextLanelets(prev_lane);
          next_lanes.reserve(next_lanes.size() + next_lanes_from_prev.size());
          next_lanes.insert(
            next_lanes.end(), next_lanes_from_prev.begin(), next_lanes_from_prev.end());
        }
        return next_lanes;
      },
      current_lane);

    // 2.1 look for neighbour lane, where end line of the lane is connected to end line of the
    // original lane
    std::copy_if(
      next_lanes_from_intersection.begin(), next_lanes_from_intersection.end(),
      std::back_inserter(extended_lanelets),
      [&current_lane](const lanelet::ConstLanelet & neighbor_lane) {
        const auto & next_left_back_point_2d = neighbor_lane.leftBound2d().back().basicPoint();
        const auto & next_right_back_point_2d = neighbor_lane.rightBound2d().back().basicPoint();

        const auto & orig_left_back_point_2d = current_lane.leftBound2d().back().basicPoint();
        const auto & orig_right_back_point_2d = current_lane.rightBound2d().back().basicPoint();
        constexpr double epsilon = 1e-5;
        const bool is_neighbour_lane =
          (next_left_back_point_2d - orig_right_back_point_2d).norm() < epsilon ||
          (next_right_back_point_2d - orig_left_back_point_2d).norm() < epsilon;
        return (current_lane.id() != neighbor_lane.id() && is_neighbour_lane);
      });
  }

  {
    const auto & p = planner_data_->parameters;
    shifted_path->path.drivable_area = util::generateDrivableArea(
      shifted_path->path, extended_lanelets, p.drivable_area_resolution, p.vehicle_length,
      planner_data_);
  }
}

void AvoidanceModule::modifyPathVelocityToPreventAccelerationOnAvoidance(ShiftedPath & path) const
{
  const auto ego_idx = avoidance_data_.ego_closest_path_index;
  const auto N = path.shift_length.size();

  // find first shift-change point from ego
  constexpr auto SHIFT_DIFF_THR = 0.1;
  size_t target_idx = N;
  const auto current_shift = path.shift_length.at(ego_idx);
  for (size_t i = ego_idx + 1; i < N; ++i) {
    if (std::abs(path.shift_length.at(i) - current_shift) > SHIFT_DIFF_THR) {
      // this index do not have to be accurate, so it can be i or i + 1.
      // but if the ego point is already on the shift-change point, ego index should be a target_idx
      // so that the distance for acceleration will be 0 and the ego speed is directly applied
      // to the path velocity (no acceleration while avoidance)
      target_idx = i - 1;
      break;
    }
  }
  if (target_idx == N) {
    DEBUG_PRINT("shift length has no changes. No velocity limit is applied.");
    return;
  }

  // calc time to the shift-change point
  constexpr auto NO_ACCEL_TIME_THR = 3.0;
  const auto s = avoidance_data_.arclength_from_ego.at(target_idx) -
                 avoidance_data_.arclength_from_ego.at(ego_idx);
  const auto t = s / std::max(getEgoSpeed(), 1.0);
  if (t > NO_ACCEL_TIME_THR) {
    DEBUG_PRINT(
      "shift point is far (s: %f, t: %f, ego_i: %lu, target_i: %lu). No velocity limit is applied.",
      s, t, ego_idx, target_idx);
    return;
  }

  // calc max velocity with given acceleration
  const auto v0 = getEgoSpeed();
  const auto vmax = std::max(
    parameters_->min_avoidance_speed_for_acc_prevention,
    std::sqrt(v0 * v0 + 2.0 * s * parameters_->max_avoidance_acceleration));

  // apply velocity limit
  constexpr size_t V_LIM_APPLY_IDX_MARGIN = 0;
  for (size_t i = ego_idx + V_LIM_APPLY_IDX_MARGIN; i < N; ++i) {
    path.path.points.at(i).point.longitudinal_velocity_mps =
      std::min(path.path.points.at(i).point.longitudinal_velocity_mps, static_cast<float>(vmax));
  }

  DEBUG_PRINT(
    "s: %f, t: %f, v0: %f, a: %f, vmax: %f, ego_i: %lu, target_i: %lu", s, t, v0,
    parameters_->max_avoidance_acceleration, vmax, ego_idx, target_idx);
}

// TODO(Horibe) clean up functions: there is a similar code in util as well.
PathWithLaneId AvoidanceModule::calcCenterLinePath(
  const std::shared_ptr<const PlannerData> & planner_data, const PoseStamped & pose) const
{
  const auto & p = planner_data->parameters;
  const auto & route_handler = planner_data->route_handler;

  PathWithLaneId centerline_path;

  // special for avoidance: take behind distance upt ot shift-start-point if it exist.
  const auto longest_dist_to_shift_point = [&]() {
    double max_dist = 0.0;
    for (const auto & pnt : path_shifter_.getShiftPoints()) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), pnt.start));
    }
    for (const auto & sp : registered_raw_shift_points_) {
      max_dist = std::max(max_dist, calcDistance2d(getEgoPose(), sp.start));
    }
    return max_dist;
  }();

  printShiftPoints(path_shifter_.getShiftPoints(), "path_shifter_.getShiftPoints()");
  printShiftPoints(registered_raw_shift_points_, "registered_raw_shift_points_");

  const auto extra_margin = 10.0;  // Since distance does not consider arclength, but just line.
  const auto backward_length =
    std::max(p.backward_path_length, longest_dist_to_shift_point + extra_margin);

  DEBUG_PRINT(
    "p.backward_path_length = %f, longest_dist_to_shift_point = %f, backward_length = %f",
    p.backward_path_length, longest_dist_to_shift_point, backward_length);

  const auto input_lanes = [this]() {
    lanelet::ConstLanelets ret{};
    for (const auto & p : previous_module_output_.reference_path->points) {
      ret.push_back(planner_data_->route_handler->getLaneletsFromId(p.lane_ids.front()));
    }
    return ret;
  }();

  lanelet::ConstLanelet current_lane;
  if (!lanelet::utils::query::getClosestLanelet(input_lanes, getEgoPose().pose, &current_lane)) {
    return centerline_path;
  }

  const auto current_lanes = route_handler->getLaneletSequence(
    current_lane, pose.pose, backward_length, p.forward_path_length);

  // const lanelet::ConstLanelets current_lanes =
  //   calcLaneAroundPose(planner_data, pose.pose, backward_length);
  centerline_path = util::getCenterLinePath(
    *route_handler, current_lanes, pose.pose, backward_length, p.forward_path_length, p);

  // for debug: check if the path backward distance is same as the desired length.
  // {
  //   const auto back_to_ego = motion_utils::calcSignedArcLength(
  //     centerline_path.points, centerline_path.points.front().point.pose.position,
  //     getEgoPosition());
  //   RCLCPP_INFO(getLogger(), "actual back_to_ego distance = %f", back_to_ego);
  // }

  centerline_path.header = route_handler->getRouteHeader();

  return centerline_path;
}

boost::optional<AvoidPoint> AvoidanceModule::calcIntersectionShiftPoint(
  const AvoidancePlanningData & data) const
{
  boost::optional<PathPointWithLaneId> intersection_point{};
  for (const auto & p : avoidance_data_.reference_path.points) {
    for (const auto & id : p.lane_ids) {
      const lanelet::ConstLanelet ll = planner_data_->route_handler->getLaneletsFromId(id);
      std::string turn_direction = ll.attributeOr("turn_direction", "else");
      if (turn_direction == "right" || turn_direction == "left") {
        intersection_point = p;
        RCLCPP_INFO(getLogger(), "intersection is found.");
        break;
      }
    }
    if (intersection_point) {
      break;
    }
  }

  const auto calcBehindPose = [&data](const Point & p, const double dist) {
    const auto & path = data.reference_path;
    const size_t start = findNearestIndex(path.points, p);
    double sum = 0.0;
    for (size_t i = start - 1; i > 1; --i) {
      sum += calcDistance2d(path.points.at(i), path.points.at(i + 1));
      if (sum > dist) {
        return path.points.at(i).point.pose;
      }
    }
    return path.points.at(0).point.pose;
  };

  const auto intersection_shift_point = [&]() {
    boost::optional<AvoidPoint> shift_point{};
    if (!intersection_point) {
      RCLCPP_INFO(getLogger(), "No intersection.");
      return shift_point;
    }

    const double ego_to_intersection_dist = calcSignedArcLength(
      data.reference_path.points, getEgoPosition(), intersection_point->point.pose.position);

    if (ego_to_intersection_dist <= 5.0) {
      RCLCPP_INFO(getLogger(), "No enough margin to intersection.");
      return shift_point;
    }

    // Search obstacles around the intersection.
    // If it exists, use its shift distance on the intersection.
    constexpr double intersection_obstacle_check_dist = 10.0;
    constexpr double intersection_shift_margin = 1.0;

    double shift_length = 0.0;  // default (no obstacle) is zero.
    for (const auto & obj : avoidance_data_.target_objects) {
      if (
        std::abs(obj.longitudinal - ego_to_intersection_dist) > intersection_obstacle_check_dist) {
        continue;
      }
      if (isOnRight(obj)) {
        continue;  // TODO(Horibe) Now only think about the left side obstacle.
      }
      shift_length = std::min(shift_length, obj.overhang_dist - intersection_shift_margin);
    }
    RCLCPP_INFO(getLogger(), "Intersection shift_length = %f", shift_length);

    AvoidPoint p{};
    p.length = shift_length;
    p.start =
      calcBehindPose(intersection_point->point.pose.position, intersection_obstacle_check_dist);
    p.end = intersection_point->point.pose;
    shift_point = p;
    return shift_point;
  }();

  return intersection_shift_point;
}

void AvoidanceModule::insertWaitPoint(const bool hard_constraints, ShiftedPath & shifted_path) const
{
  const auto & p = parameters_;
  const auto & data = avoidance_data_;
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto & lon_collision_safety_buffer = p->longitudinal_collision_safety_buffer_front;

  if (data.target_objects.empty()) {
    return;
  }

  if (data.avoiding_now) {
    return;
  }

  //         D5
  //      |<---->|                               D4
  //      |<----------------------------------------------------------------------->|
  // +-----------+            D1                 D2                      D3         +-----------+
  // |           |        |<------->|<------------------------->|<----------------->|           |
  // |    ego    |======= x ======= x ========================= x ==================|    obj    |
  // |           |    stop_point  avoid                       avoid                 |           |
  // +-----------+                start                        end                  +-----------+
  //
  // D1: p.min_prepare_distance
  // D2: min_avoid_distance
  // D3: longitudinal_avoid_margin_front (margin + D5)
  // D4: o_front.longitudinal
  // D5: base_link2front

  const auto o_front = data.target_objects.front();

  const auto min_avoid_vel = getMinimumAvoidanceEgoSpeed();
  const auto variable = getMinimumAvoidanceDistance(getShiftLength(o_front, min_avoid_vel));
  const auto constant = p->min_prepare_distance + lon_collision_safety_buffer + base_link2front;
  const auto start_longitudinal =
    o_front.longitudinal -
    std::clamp(variable + constant, p->stop_min_distance, p->stop_max_distance);

  if (!hard_constraints) {
    insertDecelPoint(
      getEgoPosition(), start_longitudinal, 0.0, shifted_path.path, debug_data_.stop_pose);
    return;
  }

  const auto stop_distance = getMildDecelDistance(0.0);

  if (!stop_distance) {
    return;
  }

  const auto insert_distance = std::max(start_longitudinal, stop_distance.get());

  insertDecelPoint(
    getEgoPosition(), insert_distance, 0.0, shifted_path.path, debug_data_.stop_pose);
}

void AvoidanceModule::insertYieldVelocity(ShiftedPath & shifted_path) const
{
  const auto & p = parameters_;
  const auto & data = avoidance_data_;

  if (data.target_objects.empty()) {
    return;
  }

  if (data.avoiding_now) {
    return;
  }

  const auto decel_distance = getMildDecelDistance(p->yield_velocity);

  if (!decel_distance) {
    return;
  }

  insertDecelPoint(
    getEgoPosition(), decel_distance.get(), p->yield_velocity, shifted_path.path,
    debug_data_.slow_pose);
}

void AvoidanceModule::insertPrepairVelocity(const bool avoidable, ShiftedPath & shifted_path) const
{
  const auto & data = avoidance_data_;

  if (data.target_objects.empty()) {
    return;
  }

  if (data.target_objects.front().reason != AvoidanceDebugFactor::TOO_LARGE_JERK) {
    return;
  }

  if (data.avoiding_now) {
    return;
  }

  if (avoidable) {
    return;
  }

  const auto decel_distance = getFeasibleDecelDistance(0.0);

  if (!decel_distance) {
    return;
  }

  insertDecelPoint(
    getEgoPosition(), decel_distance.get(), 0.0, shifted_path.path, debug_data_.slow_pose);
}

void AvoidanceModule::insertAvoidanceVelocity(
  const double avoidance_velocity, const bool show_wall, ShiftedPath & shifted_path) const
{
  const auto & p = parameters_;
  const auto & data = avoidance_data_;
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto & lon_collision_safety_buffer = p->longitudinal_collision_safety_buffer_front;

  if (getEgoSpeed() < avoidance_velocity) {
    return;
  }

  if (data.target_objects.empty()) {
    return;
  }

  //                                             +-----------+
  //                                             |           |
  //                                             |    obj    |
  //        object.longitudinal(from base_link)  |           |
  //     |<------------------------------------->+-----------+
  // +-----------+                    base_link2front
  // |           |                       |<----->|
  // |    ego    |====================== x =================== <- shifted_path
  // |           |                  decel_point
  // +-----------+

  const auto decel_distance = getMildDecelDistance(avoidance_velocity);

  if (!decel_distance) {
    return;
  }

  const auto o_front = data.target_objects.front();
  const auto longitudinal_avoid_margin_front = lon_collision_safety_buffer + base_link2front;
  const auto start_longitudinal = o_front.longitudinal - longitudinal_avoid_margin_front;
  const auto max_decel_distance = std::max(decel_distance.get(), start_longitudinal);

  insertDecelPoint(
    getEgoPosition(), max_decel_distance, avoidance_velocity, shifted_path.path,
    debug_data_.slow_pose);

  if (!show_wall) {
    debug_data_.slow_pose = {};
  }
}

bool AvoidanceModule::isSafePath(
  const PathShifter & path_shifter, ShiftedPath & shifted_path, DebugData & debug) const
{
  stop_watch_.tic(__func__);

  const auto & p = parameters_;

  const auto & forward_check_distance = p->object_check_forward_distance;
  const auto & backward_check_distance = p->adjacent_lane_check_backward_distance;
  const auto check_lanes =
    getAdjacentLane(path_shifter, forward_check_distance, backward_check_distance);

  PathWithLaneId smoothed_path;
  if (!util::applyVelocitySmoothing(
        planner_data_, shifted_path.path, forward_check_distance, smoothed_path)) {
    // TODO(Satoshi OTA)  Think later the process in the case of failed to velocity smoothing.
    RCLCPP_WARN(
      rclcpp::get_logger("behavior_path_planner").get_child("avoidance"),
      "failed velocity smoothing.");
    return true;
  }
  debug_data_.velocity_smoothed_path = smoothed_path;

  smoothed_path = util::resamplePathWithSpline(smoothed_path, 0.5);

  const auto is_safe = isSafePath(smoothed_path, check_lanes, debug);

  RCLCPP_INFO_EXPRESSION(
    getLogger(), p->print_processing_time, "- %s: %f ms", __func__,
    stop_watch_.toc(__func__, true));

  return is_safe;
}

bool AvoidanceModule::isSafePath(
  const PathWithLaneId & path, const lanelet::ConstLanelets & check_lanes, DebugData & debug) const
{
  if (path.points.empty()) {
    return true;
  }

  const auto path_with_time = [&path]() {
    std::vector<std::pair<PathPointWithLaneId, double>> ret{};

    float travel_time = 0.0;
    ret.push_back(std::make_pair(path.points.front(), travel_time));

    for (size_t i = 1; i < path.points.size(); ++i) {
      const auto & p1 = path.points.at(i - 1);
      const auto & p2 = path.points.at(i);

      const auto v = std::max(p1.point.longitudinal_velocity_mps, float{1.0});
      const auto ds = calcDistance2d(p1, p2);

      travel_time += ds / v;

      ret.push_back(std::make_pair(p2, travel_time));
    }

    return ret;
  }();

  const auto move_objects = getAdjacentLaneObjects(check_lanes);
  const auto stop_objects = getTargetObjects();

  {
    debug.unsafe_objects.clear();
    debug.margin_data_array.clear();
    debug.exist_adjacent_objects = !move_objects.empty();
  }

  bool is_enough_move_object = true;
  bool is_enough_stop_object = true;
  for (const auto & p : path_with_time) {
    MarginData margin_data{};
    margin_data.pose = getPose(p.first);

    if (p.second > 10.0) {
      break;
    }

    for (const auto & o : move_objects) {
      const auto is_enough_margin = isEnoughMargin(p.first, p.second, o, false, margin_data);

      if (!is_enough_margin) {
        debug.unsafe_objects.push_back(o);
      }

      is_enough_move_object = is_enough_move_object && is_enough_margin;
    }

    debug.margin_data_array.push_back(margin_data);
  }

  return is_enough_move_object && is_enough_stop_object;
}

bool AvoidanceModule::isEnoughMargin(
  const PathPointWithLaneId & p_ego, const double t, const ObjectData & object,
  const bool is_stop_object, MarginData & margin_data) const
{
  const auto & common_param = planner_data_->parameters;
  const auto & vehicle_width = common_param.vehicle_width;
  const auto & base_link2front = common_param.base_link2front;
  const auto & base_link2rear = common_param.base_link2rear;

  const auto p_ref = [this, &p_ego]() {
    const auto idx = findNearestIndex(avoidance_data_.reference_path.points, getPoint(p_ego));
    return getPose(avoidance_data_.reference_path.points.at(idx));
  }();

  const auto & v_ego = p_ego.point.longitudinal_velocity_mps;
  const auto & v_ego_lon = getLongitudinalVelocity(p_ref, getPose(p_ego), v_ego);
  const auto & v_obj = object.object.kinematics.initial_twist_with_covariance.twist.linear.x;

  if (!isTargetObjectType(object.object)) {
    return true;
  }

  // |           centerline
  // |               ^ x
  // |  +-------+    |
  // |  |       |    |
  // |  |       | D1 |     D2      D4
  // |  |  obj  |<-->|<---------->|<->|
  // |  |       | D3 |        +-------+
  // |  |       |<----------->|       |
  // |  +-------+    |        |       |
  // |               |        |  ego  |
  // |               |        |       |
  // |               |        |       |
  // |               |        +-------+
  // |        y <----+
  // D1: overhang_dist (signed value)
  // D2: shift_length (signed value)
  // D3: lateral_distance (should be larger than margin that's calculated from relative velocity.)
  // D4: vehicle_width (unsigned value)

  const auto reliable_path = std::max_element(
    object.object.kinematics.predicted_paths.begin(),
    object.object.kinematics.predicted_paths.end(),
    [](const PredictedPath & a, const PredictedPath & b) { return a.confidence < b.confidence; });

  if (reliable_path == object.object.kinematics.predicted_paths.end()) {
    return true;
  }

  const auto p_obj = [&t, &reliable_path]() {
    boost::optional<Pose> ret{boost::none};

    const auto dt = rclcpp::Duration(reliable_path->time_step).seconds();
    const auto idx = static_cast<size_t>(std::floor(t / dt));
    const auto res = t - dt * idx;

    if (idx > reliable_path->path.size() - 2) {
      return ret;
    }

    const auto & p_src = reliable_path->path.at(idx);
    const auto & p_dst = reliable_path->path.at(idx + 1);
    ret = calcInterpolatedPose(p_src, p_dst, res / dt);
    return ret;
  }();

  if (!p_obj) {
    return true;
  }

  const auto v_obj_lon = getLongitudinalVelocity(p_ref, p_obj.get(), v_obj);

  double hysteresis_factor = 1.0;
  if (yield_now_ && !is_stop_object) {
    hysteresis_factor = parameters_->safety_check_hysteresis_factor;
    // const auto forward_clearane = std::max(10.0, getEgoSpeed() * 10.0);
    // const auto backward_clearance = -1.0 * std::max(10.0, v_obj_lon * 10.0);
    // if (object.longitudinal < forward_clearane && object.longitudinal > backward_clearance) {
    //   return false;  // tmp.
    // }
  }

  const auto shift_length = calcLateralDeviation(p_ref, getPoint(p_ego));
  const auto lateral_distance = std::abs(object.overhang_dist - shift_length) - 0.5 * vehicle_width;
  const auto lateral_margin = getLateralMarginFromVelocity(std::abs(v_ego_lon - v_obj_lon));

  if (lateral_distance > lateral_margin * hysteresis_factor) {
    return true;
  }

  const auto lon_deviation = calcLongitudinalDeviation(getPose(p_ego), p_obj.get().position);
  const auto is_front_object = lon_deviation > 0.0;
  const auto longitudinal_margin =
    getLongitudinalMarginFromVelocity(v_ego_lon, v_obj_lon, is_stop_object, is_front_object);
  const auto vehicle_offset = is_front_object ? base_link2front : base_link2rear;
  const auto longitudinal_distance =
    std::abs(lon_deviation) - vehicle_offset - 0.5 * object.object.shape.dimensions.x;

  {
    margin_data.pose.orientation = p_ref.orientation;
    margin_data.enough_lateral_margin = false;
    margin_data.longitudinal_distance =
      std::min(margin_data.longitudinal_distance, longitudinal_distance);
    margin_data.longitudinal_margin =
      std::max(margin_data.longitudinal_margin, longitudinal_margin);
    margin_data.vehicle_width = vehicle_width;
    margin_data.base_link2front = base_link2front;
    margin_data.base_link2rear = base_link2rear;
  }

  if (longitudinal_distance > longitudinal_margin * hysteresis_factor) {
    return true;
  }

  return false;
}

lanelet::ConstLanelets AvoidanceModule::getAdjacentLane(
  const PathShifter & path_shifter, const double forward_distance,
  const double backward_distance) const
{
  const auto & rh = planner_data_->route_handler;

  bool has_left_shift = false;
  bool has_right_shift = false;

  for (const auto & sp : path_shifter.getShiftPoints()) {
    if (sp.length > 0.01) {
      has_left_shift = true;
      continue;
    }

    if (sp.length < -0.01) {
      has_right_shift = true;
      continue;
    }
  }

  lanelet::ConstLanelet current_lane;
  if (!rh->getClosestLaneletWithinRoute(getEgoPose().pose, &current_lane)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("behavior_path_planner").get_child("avoidance"),
      "failed to find closest lanelet within route!!!");
    return {};  // TODO(Satoshi Ota)
  }

  const auto ego_succeeding_lanes =
    rh->getLaneletSequence(current_lane, getEgoPose().pose, backward_distance, forward_distance);

  lanelet::ConstLanelets check_lanes{};
  for (const auto & lane : ego_succeeding_lanes) {
    const auto opt_left_lane = rh->getLeftLanelet(lane);
    if (has_left_shift && opt_left_lane) {
      check_lanes.push_back(opt_left_lane.get());
    }

    const auto opt_right_lane = rh->getRightLanelet(lane);
    if (has_right_shift && opt_right_lane) {
      check_lanes.push_back(opt_right_lane.get());
    }

    const auto right_opposite_lanes = rh->getRightOppositeLanelets(lane);
    if (has_right_shift && !right_opposite_lanes.empty()) {
      check_lanes.push_back(right_opposite_lanes.front());
    }
  }

  return check_lanes;
}

void AvoidanceModule::fillAvoidancePath(AvoidancePlanningData & data, DebugData & debug) const
{
  const auto & p = parameters_;
  const auto max_avoidance_speed = getMaximumAvoidanceEgoSpeed();
  const auto min_avoidance_speed = getMinimumAvoidanceEgoSpeed();

  constexpr double AVOIDING_SHIFT_THR = 0.1;
  data.avoiding_now = std::abs(getCurrentShift()) > AVOIDING_SHIFT_THR;

  /**
   * The avoidance lateral margin value depends on the avoidance velocity, and the faster the
   * avoidance speed, the larger the margin. But If the lateral margin is made unnecessarily large,
   * the avoidance maneuver will be unnatural. So, before searching for the safe avoidance speed,
   * calculate the feasible maximum speed that can be reached before reaching the front target
   * object, and start velocity searching from the value.
   */
  const auto feasible_speed = [&]() {
    if (data.target_objects.empty()) {
      return max_avoidance_speed;
    }

    PathWithLaneId smoothed_path;
    const auto o_front = data.target_objects.front();
    const auto check_distance = o_front.longitudinal + o_front.length;
    util::applyVelocitySmoothing(planner_data_, data.reference_path, check_distance, smoothed_path);

    const auto & p_obj = o_front.object.kinematics.initial_pose_with_covariance.pose.position;
    const auto idx = findNearestIndex(smoothed_path.points, p_obj);
    return std::min(
      max_avoidance_speed, smoothed_path.points.at(idx).point.longitudinal_velocity_mps);
  }();

  const auto max_search_speed =
    std::ceil(feasible_speed / p->avoidance_speed_step) * p->avoidance_speed_step;
  const auto min_search_speed = std::min(max_search_speed, min_avoidance_speed);

  /**
   * Search safety avoidance path. In this iteration, an avoidance path is generated for the target
   * obstacle, and check whether there is a risk of collision with vehicles in adjacent lanes, if
   * the ego follows the avoidance path.
   */
  for (float v = max_search_speed; min_search_speed < v + 1e-3; v -= p->avoidance_speed_step) {
    auto path_shifter = path_shifter_;

    /**
     * STEP 1
     * Create raw shift points from target object. The lateral margin between the ego and the target
     * object varies depending on the relative speed between the ego and the target object.
     */
    data.unapproved_raw_sp = calcRawShiftPointsFromObjects(v, data, debug);

    /**
     * STEP 2
     * Modify the raw shift points. (Merging, Triming)
     */
    const auto processed_raw_sp = applyPreProcessToRawShiftPoints(data.unapproved_raw_sp, debug);

    /**
     * STEP 3
     * Find new shift point
     */
    data.unapproved_new_sp = findNewShiftPoint(processed_raw_sp, path_shifter);
    data.found_avoidance_path =
      data.unapproved_new_sp.size() > 0 || !path_shifter.getShiftPoints().empty();
    data.lateral_margin_variable = getLateralMarginFromVelocity(v);

    /**
     * STEP 4
     * If there are new shift points, these shift points are registered in path_shifter.
     */
    if (!data.unapproved_new_sp.empty()) {
      addNewShiftPoints(path_shifter, data.unapproved_new_sp);
    }

    if (path_shifter.getShiftPoints().empty()) {
      continue;
    }

    /**
     * STEP 5
     * Generate avoidance path.
     */
    auto avoidance_path = generateAvoidancePath(path_shifter);

    /**
     * STEP 6
     * Insert avoidance velocity in avodidance path.
     */
    insertAvoidanceVelocity(v, false, avoidance_path);

    /**
     * STEP 7
     * Check avoidance path safety. For each target objects and the objects in adjacent lanes,
     * check that there is a certain amount of margin in the lateral and longitudinal direction.
     */
    data.safe = isSafePath(path_shifter, avoidance_path, debug);

    if (data.safe) {
      data.safe_path = avoidance_path;
      data.safe_new_sp = data.unapproved_new_sp;
      data.safe_velocity = v;
      break;
    }
  }

  /**
   * Check whether the ego should avoid the front object. When the ego follows reference path,
   * if the lateral distance is smaller than minimum margin, the ego should avoid the object.
   */
  if (!data.target_objects.empty()) {
    const auto o_front = data.target_objects.front();
    data.avoid_required = o_front.avoid_required;
  }

  /**
   * If the avoidance path is not safe in situation where the ego should avoid object, the ego
   * stops in front of the front object with the necessary distance to avoid the object.
   */
  if (!data.safe && data.avoid_required) {
    data.yield_required = data.found_avoidance_path && data.avoid_required;
    data.safe_path = toShiftedPath(data.reference_path);
    data.safe_velocity = min_search_speed;  // TMP
    RCLCPP_WARN(getLogger(), "not found safe avoidance path. transit yield avoidance maneuver...");
  }

  /**
   * TODO(Satoshi OTA) Think yield maneuver in the middle of avoidance.
   * Even if it is determined that a yield is necessary, the yield maneuver is not executed
   * if the avoidance has already been initiated.
   */
  if (!data.safe && data.avoiding_now) {
    data.yield_required = false;
    data.safe = true;  // OVERWRITE SAFETY JUDGE
    data.safe_velocity = getNominalAvoidanceEgoSpeed();
    data.safe_new_sp = data.unapproved_new_sp;
    RCLCPP_WARN(getLogger(), "avoiding now. could not transit yield avoidance maneuver!!!");
  }

  // /**
  //  *
  //  */
  // if (!data.safe && !data.yield_required && !data.avoiding_now) {
  //   data.safe = true;
  //   data.safe_velocity =
  //     p.enable_slow_down ? getMinimumAvoidanceEgoSpeed() : getMaximumAvoidanceEgoSpeed();
  //   RCLCPP_WARN(
  //     getLogger(),
  //     "unavoidable, but the lateral margin is larger than zero. the ego pass through narrow area.
  //     " "Be careful!!!");
  // }

  /**
   * For debug.
   */
  if (!data.target_objects.empty() && !data.avoiding_now) {
    const auto o_front = data.target_objects.front();
    const auto & base_link2front = planner_data_->parameters.base_link2front;
    data.shift_length = getShiftLength(o_front, data.safe_velocity);
    data.avoid_variable_distance = getSharpAvoidanceDistance(data.shift_length);
    data.avoid_constant_distance = getNominalPrepareDistance() +
                                   parameters_->longitudinal_collision_safety_buffer_front +
                                   base_link2front;
    data.front_object_longitudinal = o_front.longitudinal;
    data.required_jerk = PathShifter::calcJerkFromLatLonDistance(
      data.shift_length, data.front_object_longitudinal - data.avoid_constant_distance,
      getSharpAvoidanceEgoSpeed());
  }

  {
    debug.output_shift = data.safe_path.shift_length;
    debug.current_raw_shift = data.unapproved_raw_sp;
    debug.new_shift_points = data.safe_new_sp;
  }

  if (!data.target_objects.empty() && !data.avoiding_now && !data.unapproved_new_sp.empty()) {
    const auto total_distance = data.avoid_variable_distance + data.avoid_constant_distance;
    const auto opt_feasible_bound = calcLongitudinalOffsetPose(
      data.reference_path.points, getEgoPosition(),
      data.front_object_longitudinal - total_distance);
    if (opt_feasible_bound) {
      debug.feasible_bound = opt_feasible_bound.get();
    } else {
      debug.feasible_bound = getPose(data.reference_path.points.front());
    }
  }
}

AvoidanceState AvoidanceModule::updateEgoState(const AvoidancePlanningData & data) const
{
  if (!data.avoid_required) {
    return AvoidanceState::NOT_AVOID;
  }

  if (!data.found_avoidance_path) {
    return AvoidanceState::AVOID_PATH_NOT_READY;
  }

  if (data.yield_required && parameters_->enable_avoidance_yield) {
    return AvoidanceState::YIELD;
  }

  if (isWaitingApproval() && path_shifter_.getShiftPoints().empty()) {
    return AvoidanceState::AVOID_PATH_READY;
  }

  return AvoidanceState::AVOID_EXECUTE;
}

BehaviorModuleOutput AvoidanceModule::plan()
{
  stop_watch_.tic(__func__);

  const auto & p = parameters_;
  const auto & data = avoidance_data_;
  const auto & debug = debug_data_;

  RCLCPP_INFO_EXPRESSION(getLogger(), p->print_processing_time, "=== %s ===", __func__);

  resetPathCandidate();
  resetPathReference();

  /**
   * Has new shift point?
   *   Yes -> Is it approved?
   *       Yes -> add the shift point.
   *       No  -> set approval_handler to WAIT_APPROVAL state.
   *   No -> waiting approval?
   *       Yes -> clear WAIT_APPROVAL state.
   *       No  -> do nothing.
   */
  if (!data.safe_new_sp.empty()) {
    printShiftPoints(data.safe_new_sp, "safe_new_sp");

    const auto ap = getNotStraightShiftPoint(data.safe_new_sp);
    if (ap.getRelativeLength() > 0.0) {
      removePreviousRTCStatusRight();
    } else if (ap.getRelativeLength() < 0.0) {
      removePreviousRTCStatusLeft();
    } else {
      RCLCPP_WARN_STREAM(getLogger(), "Direction is UNKNOWN");
    }

    addShiftPointIfApproved(data.safe_new_sp, data.unapproved_raw_sp);
  } else if (isWaitingApproval()) {
    clearWaitingApproval();
    removeCandidateRTCStatus();
  }

  RCLCPP_INFO_EXPRESSION(
    getLogger(), p->print_processing_time, "- step1: %f ms", stop_watch_.toc(__func__, false));

  // generate path with shift points that have been inserted.
  auto avoidance_path = generateAvoidancePath(path_shifter_);

  // Drivable area generation.
  generateExtendedDrivableArea(&avoidance_path);

  // modify max speed to prevent acceleration in avoidance maneuver.
  modifyPathVelocityToPreventAccelerationOnAvoidance(avoidance_path);

  RCLCPP_INFO_EXPRESSION(
    getLogger(), p->print_processing_time, "- step2: %f ms", stop_watch_.toc(__func__, false));

  // post processing
  {
    postProcess(path_shifter_);  // remove old shift points
    prev_output_ = avoidance_path;
    prev_linear_shift_path_ = toShiftedPath(data.reference_path);
    path_shifter_.generate(&prev_linear_shift_path_, true, SHIFT_TYPE::LINEAR);
    prev_reference_ = data.reference_path;
  }

  BehaviorModuleOutput output;
  output.turn_signal_info = calcTurnSignalInfo(avoidance_path);
  // sparse resampling for computational cost
  {
    avoidance_path.path =
      util::resamplePathWithSpline(avoidance_path.path, p->resample_interval_for_output);
  }

  avoidance_data_.state = updateEgoState(data);
  switch (avoidance_data_.state) {
    case AvoidanceState::NOT_AVOID: {
      yield_now_ = false;
      break;
    }
    case AvoidanceState::YIELD: {
      insertYieldVelocity(avoidance_path);
      insertWaitPoint(p->hard_constraints, avoidance_path);
      removeAllRegisteredShiftPoints(path_shifter_);
      yield_now_ = true;
      break;
    }
    case AvoidanceState::AVOID_PATH_NOT_READY: {
      insertPrepairVelocity(false, avoidance_path);
      insertWaitPoint(p->hard_constraints, avoidance_path);
      yield_now_ = false;
      break;
    }
    case AvoidanceState::AVOID_PATH_READY: {
      insertPrepairVelocity(true, avoidance_path);
      insertWaitPoint(p->hard_constraints, avoidance_path);
      yield_now_ = false;
      break;
    }
    case AvoidanceState::AVOID_EXECUTE: {
      insertAvoidanceVelocity(data.safe_velocity, true, avoidance_path);
      yield_now_ = false;
      break;
    }
    default:
      throw std::domain_error("invalid behaivor");
  }

  output.path = std::make_shared<PathWithLaneId>(avoidance_path.path);
  output.reference_path = previous_module_output_.reference_path;

  clipPathLength(*output.path);

  DEBUG_PRINT("exit plan(): set prev output (back().lat = %f)", prev_output_.shift_length.back());

  updateRegisteredRTCStatus(avoidance_path.path);

  if (p->publish_debug_marker) {
    setDebugData(data, path_shifter_, debug);
  }

  RCLCPP_INFO_EXPRESSION(
    getLogger(), p->print_processing_time, "- step3: %f ms", stop_watch_.toc(__func__, false));

  return output;
}

CandidateOutput AvoidanceModule::planCandidate() const
{
  DEBUG_PRINT("AVOIDANCE planCandidate start");

  const auto & data = avoidance_data_;

  CandidateOutput output;

  auto shifted_path = data.safe_path;

  if (!data.safe_new_sp.empty()) {  // clip from shift start index for visualize
    clipByMinStartIdx(data.safe_new_sp, shifted_path.path);
    const auto ap = getNotStraightShiftPoint(data.safe_new_sp);
    output.lateral_shift = ap.getRelativeLength();
    output.distance_to_path_change = data.safe_new_sp.front().start_longitudinal;
  }

  clipPathLength(shifted_path.path);

  output.path_candidate = shifted_path.path;

  return output;
}

BehaviorModuleOutput AvoidanceModule::planWaitingApproval()
{
  // we can execute the plan() since it handles the approval appropriately.
  BehaviorModuleOutput out = plan();
  out.turn_signal_info = previous_module_output_.turn_signal_info;
  const auto candidate = planCandidate();
  constexpr double threshold_to_update_status = -1.0e-03;
  if (candidate.distance_to_path_change > threshold_to_update_status) {
    updateCandidateRTCStatus(candidate);
    waitApproval();
  } else {
    clearWaitingApproval();
    removeCandidateRTCStatus();
  }
  path_candidate_ = std::make_shared<PathWithLaneId>(candidate.path_candidate);
  path_reference_ = out.reference_path;
  return out;
}

void AvoidanceModule::addShiftPointIfApproved(
  const AvoidPointArray & unapproved_new_sp, const AvoidPointArray & unapproved_raw_sp)
{
  if (isActivated()) {
    DEBUG_PRINT("We want to add this shift point, and approved. ADD SHIFT POINT!");
    const size_t prev_size = path_shifter_.getShiftPointsSize();
    addNewShiftPoints(path_shifter_, unapproved_new_sp);

    current_raw_shift_points_ = unapproved_raw_sp;

    // register original points for consistency
    registerRawShiftPoints(unapproved_new_sp);

    const auto ap = getNotStraightShiftPoint(unapproved_new_sp);
    if (ap.getRelativeLength() > 0.0) {
      left_shift_array_.push_back({uuid_left_, unapproved_new_sp.front().start});
    } else if (ap.getRelativeLength() < 0.0) {
      right_shift_array_.push_back({uuid_right_, unapproved_new_sp.front().start});
    }

    uuid_left_ = generateUUID();
    uuid_right_ = generateUUID();
    candidate_uuid_ = generateUUID();

    DEBUG_PRINT("shift_point size: %lu -> %lu", prev_size, path_shifter_.getShiftPointsSize());
  } else {
    DEBUG_PRINT("We want to add this shift point, but NOT approved. waiting...");
    waitApproval();
  }
}

/**
 * set new shift points. remove old shift points if it has a conflict.
 */
void AvoidanceModule::addNewShiftPoints(
  PathShifter & path_shifter, const AvoidPointArray & new_shift_points) const
{
  ShiftPointArray future = toShiftPointArray(new_shift_points);

  size_t min_start_idx = std::numeric_limits<size_t>::max();
  for (const auto & sp : new_shift_points) {
    min_start_idx = std::min(min_start_idx, sp.start_idx);
  }

  const auto current_shift_points = path_shifter.getShiftPoints();

  DEBUG_PRINT("min_start_idx = %lu", min_start_idx);

  // Remove shift points that starts later than the new_shift_point from path_shifter.
  //
  // Why? Because shifter sorts by start position and applies shift points, so if there is a
  // shift point that starts after the one you are going to put in, new one will be affected
  // by the old one.
  //
  // Is it ok? This is a situation where the vehicle was originally going to avoid at the farther
  // point, but decided to avoid it at a closer point. In this case, it is reasonable to cancel the
  // farther avoidance.
  for (const auto & sp : current_shift_points) {
    if (sp.start_idx >= min_start_idx) {
      DEBUG_PRINT(
        "sp.start_idx = %lu, this sp starts after new proposal. remove this one.", sp.start_idx);
    } else {
      DEBUG_PRINT("sp.start_idx = %lu, no conflict. keep this one.", sp.start_idx);
      future.push_back(sp);
    }
  }

  path_shifter.setShiftPoints(future);
}

AvoidPointArray AvoidanceModule::findNewShiftPoint(
  const AvoidPointArray & candidates, const PathShifter & shifter) const
{
  (void)shifter;

  debug_data_.unfeasible_shift.clear();

  if (candidates.empty()) {
    DEBUG_PRINT("shift candidates is empty. return None.");
    return {};
  }

  printShiftPoints(candidates, "findNewShiftPoint: candidates");

  // Retrieve the subsequent linear shift point from the given index point.
  const auto getShiftPointWithSubsequentStraight = [this, &candidates](size_t i) {
    AvoidPointArray subsequent{candidates.at(i)};
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      const auto next_shift = candidates.at(j);
      if (std::abs(next_shift.getRelativeLength()) < 1.0e-2) {
        subsequent.push_back(next_shift);
        DEBUG_PRINT("j = %lu, relative shift is zero. add together.", j);
      } else {
        DEBUG_PRINT("j = %lu, relative shift is not zero = %f.", j, next_shift.getRelativeLength());
        break;
      }
    }
    return subsequent;
  };

  const auto calcJerk = [this](const auto & ap) {
    return path_shifter_.calcJerkFromLatLonDistance(
      ap.getRelativeLength(), ap.getRelativeLongitudinal(), getSharpAvoidanceEgoSpeed());
  };

  for (size_t i = 0; i < candidates.size(); ++i) {
    const auto & candidate = candidates.at(i);
    std::stringstream ss;
    ss << "i = " << i << ", id = " << candidate.id;
    const auto pfx = ss.str().c_str();

    if (prev_reference_.points.size() != prev_linear_shift_path_.path.points.size()) {
      throw std::logic_error("prev_reference_ and prev_linear_shift_path_ must have same size.");
    }

    // new shift points must exist in front of Ego
    if (candidate.start_longitudinal < 0.0) {
      continue;
    }

    // TODO(Horibe): this code prohibits the changes on ego pose. Think later.
    // if (candidate.start_idx < avoidance_data_.ego_closest_path_index) {
    //   DEBUG_PRINT("%s, start_idx is behind ego. skip.", pfx);
    //   continue;
    // }

    const auto current_shift = prev_linear_shift_path_.shift_length.at(
      findNearestIndex(prev_reference_.points, candidate.end.position));

    // TODO(Horibe) test fails with this print. why?
    // DEBUG_PRINT("%s, shift current: %f, candidate: %f", pfx, current_shift, candidate.length);

    const auto new_point_threshold = parameters_->avoidance_execution_lateral_threshold;
    if (std::abs(candidate.length - current_shift) > new_point_threshold) {
      constexpr double LATERAL_JERK_TOR = 0.3;
      if (calcJerk(candidate) > parameters_->max_lateral_jerk + LATERAL_JERK_TOR) {
        debug_data_.unfeasible_shift.push_back(candidate);
        DEBUG_PRINT(
          "%s, Failed to find new shift: jerk limit over (%f).", pfx, calcJerk(candidate));
        break;
      }

      DEBUG_PRINT(
        "%s, New shift point is found!!! shift change: %f -> %f", pfx, current_shift,
        candidate.length);
      return getShiftPointWithSubsequentStraight(i);
    }
  }

  DEBUG_PRINT("No new shift point exists.");
  return {};
}

ObjectDataArray AvoidanceModule::getTargetObjects() const { return avoidance_data_.target_objects; }

ObjectDataArray AvoidanceModule::getAdjacentLaneObjects(
  const lanelet::ConstLanelets & adjacent_lanes) const
{
  ObjectDataArray objects;
  for (const auto & o : avoidance_data_.ignore_objects) {
    if (isCentroidWithinLanelets(o.object, adjacent_lanes)) {
      objects.push_back(o);
    }
  }

  return objects;
}

double AvoidanceModule::getEgoSpeed() const
{
  return std::abs(planner_data_->self_odometry->twist.twist.linear.x);
}

float AvoidanceModule::getMinimumAvoidanceEgoSpeed() const
{
  const auto & p = parameters_;
  return p->target_velocity_matrix.front();
}

float AvoidanceModule::getMaximumAvoidanceEgoSpeed() const
{
  const auto & p = parameters_;
  return p->target_velocity_matrix.at(p->col_size - 1);
}

double AvoidanceModule::getNominalAvoidanceEgoSpeed() const
{
  return std::max(getEgoSpeed(), parameters_->min_nominal_avoidance_speed);
}
double AvoidanceModule::getNominalAvoidanceEgoSpeed(const double v) const
{
  return std::min(v, parameters_->min_nominal_avoidance_speed);
}
double AvoidanceModule::getSharpAvoidanceEgoSpeed() const
{
  return std::max(getEgoSpeed(), parameters_->min_sharp_avoidance_speed);
}

Point AvoidanceModule::getEgoPosition() const { return planner_data_->self_pose->pose.position; }

PoseStamped AvoidanceModule::getEgoPose() const { return *(planner_data_->self_pose); }

PoseStamped AvoidanceModule::getUnshiftedEgoPose(const ShiftedPath & prev_path) const
{
  const auto ego_pose = getEgoPose();

  if (prev_path.path.points.empty()) {
    return ego_pose;
  }

  // un-shifted fot current ideal pose
  const auto closest = findNearestIndex(prev_path.path.points, ego_pose.pose.position);

  PoseStamped unshifted_pose = ego_pose;
  unshifted_pose.pose.orientation = prev_path.path.points.at(closest).point.pose.orientation;

  util::shiftPose(&unshifted_pose.pose, -prev_path.shift_length.at(closest));
  unshifted_pose.pose.orientation = ego_pose.pose.orientation;

  return unshifted_pose;
}

double AvoidanceModule::getMinimumAvoidanceDistance(const double shift_length) const
{
  const auto & p = parameters_;
  const auto distance_by_jerk = path_shifter_.calcLongitudinalDistFromJerk(
    shift_length, p->nominal_lateral_jerk, getMinimumAvoidanceEgoSpeed());
  // shift_length, p.nominal_lateral_jerk, p.min_sharp_avoidance_speed);

  return std::max(p->min_avoidance_distance, distance_by_jerk);
}

double AvoidanceModule::getNominalAvoidanceDistance(const double shift_length) const
{
  const auto & p = parameters_;
  const auto distance_by_jerk = path_shifter_.calcLongitudinalDistFromJerk(
    shift_length, parameters_->nominal_lateral_jerk, getNominalAvoidanceEgoSpeed());

  return std::max(p->min_avoidance_distance, distance_by_jerk);
}

double AvoidanceModule::getNominalAvoidanceDistance(const double shift_length, const double v) const
{
  const auto & p = parameters_;
  const auto distance_by_jerk = path_shifter_.calcLongitudinalDistFromJerk(
    shift_length, p->nominal_lateral_jerk,
    std::clamp(getEgoSpeed(), p->min_nominal_avoidance_speed, v));

  return std::max(p->min_avoidance_distance, distance_by_jerk);
}

double AvoidanceModule::getSharpAvoidanceDistance(const double shift_length) const
{
  const auto & p = parameters_;
  const auto distance_by_jerk = path_shifter_.calcLongitudinalDistFromJerk(
    shift_length, parameters_->max_lateral_jerk, getSharpAvoidanceEgoSpeed());

  return std::max(p->min_avoidance_distance, distance_by_jerk);
}

double AvoidanceModule::getNominalPrepareDistance() const
{
  const auto & p = parameters_;
  const auto epsilon_m = 0.01;  // for floating error to pass "has_enough_distance" check.
  const auto nominal_distance = std::max(getEgoSpeed() * p->prepare_time, p->min_prepare_distance);
  return nominal_distance + epsilon_m;
}

double AvoidanceModule::getLateralMarginFromVelocity(const double velocity) const
{
  const auto & p = parameters_;

  if (p->col_size < 2 || p->col_size * 2 != p->target_velocity_matrix.size()) {
    throw std::logic_error("invalid matrix col size.");
  }

  if (velocity < p->target_velocity_matrix.front()) {
    return p->target_velocity_matrix.at(p->col_size);
  }

  if (velocity > p->target_velocity_matrix.at(p->col_size - 1)) {
    return p->target_velocity_matrix.back();
  }

  for (size_t i = 1; i < p->col_size; ++i) {
    if (velocity < p->target_velocity_matrix.at(i)) {
      const auto v1 = p->target_velocity_matrix.at(i - 1);
      const auto v2 = p->target_velocity_matrix.at(i);
      const auto m1 = p->target_velocity_matrix.at(i - 1 + p->col_size);
      const auto m2 = p->target_velocity_matrix.at(i + p->col_size);

      const auto v_clamp = std::clamp(velocity, v1, v2);
      return m1 + (m2 - m1) * (v_clamp - v1) / (v2 - v1);
    }
  }

  return p->target_velocity_matrix.back();
}

double AvoidanceModule::getLongitudinalMarginFromVelocity(
  const double v_ego, const double v_obj, const bool is_stop_object,
  const bool is_front_object) const
{
  constexpr double max_decel = 2.5;  // m/ss
                                     //
  const auto & p = parameters_;
  const auto & lon_collision_safety_buffer_front = p->longitudinal_collision_safety_buffer_front;
  const auto & lon_collision_safety_buffer_back = p->longitudinal_collision_safety_buffer_back;

  if (is_stop_object) {
    return is_front_object ? lon_collision_safety_buffer_front : lon_collision_safety_buffer_back;
  }

  double longitudinal_margin = 0.0;

  if (is_front_object) {
    longitudinal_margin =
      std::pow(v_obj, 2.0) / (2.0 * max_decel) - std::pow(v_ego, 2.0) / (2.0 * max_decel);
  } else {
    longitudinal_margin =
      std::pow(v_ego, 2.0) / (2.0 * max_decel) - std::pow(v_obj, 2.0) / (2.0 * max_decel);
  }

  return std::max(p->min_longitudinal_margin_for_moving_object, longitudinal_margin);
}

boost::optional<double> AvoidanceModule::getFeasibleDecelDistance(
  const double target_velocity) const
{
  const auto & a_now = planner_data_->self_acceleration->accel.accel.linear.x;
  const auto & a_lim = parameters_->max_deceleration;
  const auto & j_lim = parameters_->max_jerk;
  return calcDecelDistWithJerkAndAccConstraints(
    getEgoSpeed(), target_velocity, a_now, a_lim, j_lim, -1.0 * j_lim);
}

boost::optional<double> AvoidanceModule::getMildDecelDistance(const double target_velocity) const
{
  const auto & a_now = planner_data_->self_acceleration->accel.accel.linear.x;
  const auto & a_lim = parameters_->nominal_deceleration;
  const auto & j_lim = parameters_->nominal_jerk;
  return calcDecelDistWithJerkAndAccConstraints(
    getEgoSpeed(), target_velocity, a_now, a_lim, j_lim, -1.0 * j_lim);
}

double AvoidanceModule::getMinimumLateralMargin() const
{
  const auto & p = parameters_;
  return p->target_velocity_matrix.at(p->col_size);
}

double AvoidanceModule::getMaximumLateralMargin() const
{
  const auto & p = parameters_;
  return p->target_velocity_matrix.back();
}

ShiftedPath AvoidanceModule::generateAvoidancePath(PathShifter & path_shifter) const
{
  stop_watch_.tic(__func__);

  DEBUG_PRINT("path_shifter: base shift = %f", getCurrentBaseShift());
  printShiftPoints(path_shifter.getShiftPoints(), "path_shifter shift points");

  ShiftedPath shifted_path;
  if (!path_shifter.generate(&shifted_path)) {
    RCLCPP_ERROR(getLogger(), "failed to generate shifted path.");
    return toShiftedPath(avoidance_data_.reference_path);
  }

  RCLCPP_INFO_EXPRESSION(
    getLogger(), parameters_->print_processing_time, "- %s: %f ms", __func__,
    stop_watch_.toc(__func__, true));

  return shifted_path;
}

void AvoidanceModule::removeAllRegisteredShiftPoints(PathShifter & path_shifter)
{
  current_raw_shift_points_.clear();
  registered_raw_shift_points_.clear();
  path_shifter.setShiftPoints(ShiftPointArray{});
}

void AvoidanceModule::postProcess(PathShifter & path_shifter) const
{
  path_shifter.removeBehindShiftPointAndSetBaseOffset(getEgoPosition());
}

void AvoidanceModule::updateData()
{
  stop_watch_.tic(__func__);

  RCLCPP_INFO_EXPRESSION(getLogger(), parameters_->print_processing_time, "=== %s ===", __func__);

  debug_data_ = DebugData();
  avoidance_data_ = calcAvoidancePlanningData(debug_data_);

  RCLCPP_INFO_EXPRESSION(
    getLogger(), parameters_->print_processing_time, "- step1: %f ms",
    stop_watch_.toc(__func__, false));

  // TODO(Horibe): this is not tested yet, disable now.
  updateRegisteredObject(avoidance_data_.target_objects);
  CompensateDetectionLost(avoidance_data_.target_objects, avoidance_data_.ignore_objects);

  RCLCPP_INFO_EXPRESSION(
    getLogger(), parameters_->print_processing_time, "- step2: %f ms",
    stop_watch_.toc(__func__, false));

  std::sort(
    avoidance_data_.target_objects.begin(), avoidance_data_.target_objects.end(),
    [](auto a, auto b) { return a.longitudinal < b.longitudinal; });

  path_shifter_.setPath(avoidance_data_.reference_path);

  // update registered shift point for new reference path & remove past objects
  updateRegisteredRawShiftPoints();

  // for the first time
  if (prev_output_.path.points.empty()) {
    prev_output_.path = avoidance_data_.reference_path;
    prev_output_.shift_length = std::vector<double>(prev_output_.path.points.size(), 0.0);
  }
  if (prev_linear_shift_path_.path.points.empty()) {
    prev_linear_shift_path_.path = avoidance_data_.reference_path;
    prev_linear_shift_path_.shift_length =
      std::vector<double>(prev_linear_shift_path_.path.points.size(), 0.0);
  }
  if (prev_reference_.points.empty()) {
    prev_reference_ = avoidance_data_.reference_path;
  }

  RCLCPP_INFO_EXPRESSION(
    getLogger(), parameters_->print_processing_time, "- step3: %f ms",
    stop_watch_.toc(__func__, false));

  fillAvoidancePath(avoidance_data_, debug_data_);

  RCLCPP_INFO_EXPRESSION(
    getLogger(), parameters_->print_processing_time, "- step4: %f ms",
    stop_watch_.toc(__func__, false));
}

/*
 * updateRegisteredObject
 *
 * Same object is observed this time -> update registered object with the new one.
 * Not observed -> increment the lost_count. if it exceeds the threshold, remove it.
 * How to check if it is same object?
 *   - it has same ID
 *   - it has different id, but sn object is found around similar position
 */
void AvoidanceModule::updateRegisteredObject(const ObjectDataArray & now_objects)
{
  const auto updateIfDetectedNow = [&now_objects, this](auto & registered_object) {
    const auto & n = now_objects;
    const auto r_id = registered_object.object.object_id;
    const auto same_id_obj = std::find_if(
      n.begin(), n.end(), [&r_id](const auto & o) { return o.object.object_id == r_id; });

    // same id object is detected. update registered.
    if (same_id_obj != n.end()) {
      registered_object = *same_id_obj;
      return true;
    }

    constexpr auto POS_THR = 1.5;
    const auto r_pos = registered_object.object.kinematics.initial_pose_with_covariance.pose;
    const auto similar_pos_obj = std::find_if(n.begin(), n.end(), [&](const auto & o) {
      return calcDistance2d(r_pos, o.object.kinematics.initial_pose_with_covariance.pose) < POS_THR;
    });

    // same id object is not detected, but object is found around registered. update registered.
    if (similar_pos_obj != n.end()) {
      registered_object = *similar_pos_obj;
      return true;
    }

    // Same ID nor similar position object does not found.
    return false;
  };

  // -- check registered_objects, remove if lost_count exceeds limit. --
  for (int i = static_cast<int>(registered_objects_.size()) - 1; i >= 0; --i) {
    auto & r = registered_objects_.at(i);
    const std::string s = getUuidStr(r);

    // registered object is not detected this time. lost count up.
    if (!updateIfDetectedNow(r)) {
      r.lost_time = (clock_->now() - r.last_seen).seconds();
    } else {
      r.last_seen = clock_->now();
      r.lost_time = 0.0;
    }

    // lost count exceeds threshold. remove object from register.
    if (r.lost_time > parameters_->object_last_seen_threshold) {
      registered_objects_.erase(registered_objects_.begin() + i);
    }
  }

  const auto isAlreadyRegistered = [this](const auto & n_id) {
    const auto & r = registered_objects_;
    return std::any_of(
      r.begin(), r.end(), [&n_id](const auto & o) { return o.object.object_id == n_id; });
  };

  // -- check now_objects, add it if it has new object id --
  for (const auto & now_obj : now_objects) {
    if (!isAlreadyRegistered(now_obj.object.object_id)) {
      registered_objects_.push_back(now_obj);
    }
  }
}

/*
 * CompensateDetectionLost
 *
 * add registered object if the now_objects does not contain the same object_id.
 *
 */
void AvoidanceModule::CompensateDetectionLost(
  ObjectDataArray & now_objects, ObjectDataArray & ignore_objects) const
{
  const auto old_size = now_objects.size();  // for debug

  const auto isDetectedNow = [&](const auto & r_id) {
    const auto & n = now_objects;
    return std::any_of(
      n.begin(), n.end(), [&r_id](const auto & o) { return o.object.object_id == r_id; });
  };

  const auto isIgnoreObject = [&](const auto & r_id) {
    const auto & n = ignore_objects;
    return std::any_of(
      n.begin(), n.end(), [&r_id](const auto & o) { return o.object.object_id == r_id; });
  };

  for (const auto & registered : registered_objects_) {
    if (
      !isDetectedNow(registered.object.object_id) && !isIgnoreObject(registered.object.object_id)) {
      now_objects.push_back(registered);
    }
  }
  DEBUG_PRINT("object size: %lu -> %lu", old_size, now_objects.size());
}

void AvoidanceModule::fillObjectMovingTime(ObjectData & object_data) const
{
  const auto & object_vel =
    object_data.object.kinematics.initial_twist_with_covariance.twist.linear.x;
  const auto is_faster_than_threshold = object_vel > parameters_->threshold_speed_object_is_stopped;

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    stopped_objects_.begin(), stopped_objects_.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  const auto is_new_object = same_id_obj == stopped_objects_.end();

  if (!is_faster_than_threshold) {
    object_data.last_stop = clock_->now();
    object_data.move_time = 0.0;
    if (is_new_object) {
      object_data.stop_time = 0.0;
      object_data.last_move = clock_->now();
      stopped_objects_.push_back(object_data);
    } else {
      same_id_obj->stop_time = (clock_->now() - same_id_obj->last_move).seconds();
      same_id_obj->last_stop = clock_->now();
      same_id_obj->move_time = 0.0;
      object_data.stop_time = same_id_obj->stop_time;
    }
    return;
  }

  if (is_new_object) {
    object_data.move_time = std::numeric_limits<double>::max();
    object_data.stop_time = 0.0;
    object_data.last_move = clock_->now();
    return;
  }

  object_data.last_stop = same_id_obj->last_stop;
  object_data.move_time = (clock_->now() - same_id_obj->last_stop).seconds();
  object_data.stop_time = 0.0;

  if (object_data.move_time > parameters_->threshold_time_object_is_moving) {
    stopped_objects_.erase(same_id_obj);
  }
}

void AvoidanceModule::fillObjectEnvelopePolygon(
  const Pose & closest_pose, ObjectData & object_data) const
{
  using boost::geometry::within;

  const auto id = object_data.object.object_id;
  const auto same_id_obj = std::find_if(
    registered_objects_.begin(), registered_objects_.end(),
    [&id](const auto & o) { return o.object.object_id == id; });

  if (same_id_obj == registered_objects_.end()) {
    object_data.envelope_poly =
      createEnvelopePolygon(object_data, closest_pose, parameters_->object_envelope_buffer);
    return;
  }

  Polygon2d object_polygon{};
  util::calcObjectPolygon(object_data.object, &object_polygon);

  if (!within(object_polygon, same_id_obj->envelope_poly)) {
    object_data.envelope_poly =
      createEnvelopePolygon(object_data, closest_pose, parameters_->object_envelope_buffer);
    return;
  }

  object_data.envelope_poly = same_id_obj->envelope_poly;
}

void AvoidanceModule::onEntry()
{
  DEBUG_PRINT("AVOIDANCE onEntry. wait approval!");
  initVariables();
  // current_state_ = ModuleStatus::IDLE;
}

void AvoidanceModule::onExit()
{
  DEBUG_PRINT("AVOIDANCE onExit");
  initVariables();
  current_state_ = ModuleStatus::IDLE;
  clearWaitingApproval();
  removeRTCStatus();
}

void AvoidanceModule::initVariables()
{
  prev_output_ = ShiftedPath();
  prev_linear_shift_path_ = ShiftedPath();
  prev_reference_ = PathWithLaneId();
  path_shifter_ = PathShifter{};
  left_shift_array_.clear();
  right_shift_array_.clear();

  debug_avoidance_msg_array_ptr_.reset();
  debug_data_ = DebugData();
  debug_marker_.markers.clear();
  registered_raw_shift_points_ = {};
  current_raw_shift_points_ = {};
  original_unique_id = 0;

  resetPathCandidate();
  resetPathReference();
}

void AvoidanceModule::clipPathLength(PathWithLaneId & path) const
{
  const double forward = planner_data_->parameters.forward_path_length;
  const double backward = planner_data_->parameters.backward_path_length;

  util::clipPathLength(path, getEgoPose().pose, forward, backward);
}

bool AvoidanceModule::isTargetObjectType(const PredictedObject & object) const
{
  using autoware_auto_perception_msgs::msg::ObjectClassification;
  const auto t = util::getHighestProbLabel(object.classification);
  const auto is_object_type =
    ((t == ObjectClassification::CAR && parameters_->avoid_car) ||
     (t == ObjectClassification::TRUCK && parameters_->avoid_truck) ||
     (t == ObjectClassification::BUS && parameters_->avoid_bus) ||
     (t == ObjectClassification::TRAILER && parameters_->avoid_trailer) ||
     (t == ObjectClassification::UNKNOWN && parameters_->avoid_unknown) ||
     (t == ObjectClassification::BICYCLE && parameters_->avoid_bicycle) ||
     (t == ObjectClassification::MOTORCYCLE && parameters_->avoid_motorcycle) ||
     (t == ObjectClassification::PEDESTRIAN && parameters_->avoid_pedestrian));
  return is_object_type;
}

TurnSignalInfo AvoidanceModule::calcTurnSignalInfo(const ShiftedPath & path) const
{
  TurnSignalInfo turn_signal;

  const auto shift_points = path_shifter_.getShiftPoints();
  if (shift_points.empty()) {
    return {};
  }

  const auto latest_shift_point = shift_points.front();  // assuming it is sorted.

  const auto turn_info = util::getPathTurnSignal(
    avoidance_data_.current_lanelets, path, latest_shift_point, planner_data_->self_pose->pose,
    planner_data_->self_odometry->twist.twist.linear.x, planner_data_->parameters);

  // Set turn signal if the vehicle across the lane.
  if (!path.shift_length.empty()) {
    if (isAvoidancePlanRunning()) {
      turn_signal.turn_signal.command = turn_info.first.command;
    }
  }

  // calc distance from ego to latest_shift_point end point.
  if (turn_info.second >= 0.0) {
    turn_signal.signal_distance = turn_info.second;
  }

  return turn_signal;
}

double AvoidanceModule::getCurrentShift() const
{
  return prev_output_.shift_length.at(findNearestIndex(prev_output_.path.points, getEgoPosition()));
}
double AvoidanceModule::getCurrentLinearShift() const
{
  return prev_linear_shift_path_.shift_length.at(
    findNearestIndex(prev_linear_shift_path_.path.points, getEgoPosition()));
}

double AvoidanceModule::getLongitudinalVelocity(
  const Pose & p_ref, const Pose & p_target, const double v)
{
  const auto yaw = tier4_autoware_utils::calcYawDeviation(p_ref, p_target);
  return v * std::cos(yaw);
}

AvoidPoint AvoidanceModule::getNotStraightShiftPoint(const AvoidPointArray & shift_points)
{
  for (const auto & ap : shift_points) {
    if (fabs(ap.getRelativeLength()) > 0.01) {
      return ap;
    }
  }

  return {};
}

void AvoidanceModule::setDebugData(
  const AvoidancePlanningData & data, const PathShifter & shifter, const DebugData & debug) const
{
  using marker_utils::createLaneletsAreaMarkerArray;
  using marker_utils::createObjectsMarkerArray;
  using marker_utils::createPathMarkerArray;
  using marker_utils::createPoseMarkerArray;
  using marker_utils::createShiftLengthMarkerArray;
  using marker_utils::createShiftPointMarkerArray;
  using marker_utils::avoidance_marker::createAvoidPointMarkerArray;
  using marker_utils::avoidance_marker::createEgoStatusMarkerArray;
  using marker_utils::avoidance_marker::createOtherObjectsMarkerArray;
  using marker_utils::avoidance_marker::createOverhangFurthestLineStringMarkerArray;
  using marker_utils::avoidance_marker::createPredictedVehiclePositions;
  using marker_utils::avoidance_marker::createTargetObjectsMarkerArray;
  using marker_utils::avoidance_marker::createUnavoidableObjectsMarkerArray;
  using marker_utils::avoidance_marker::createUnsafeObjectsMarkerArray;
  using marker_utils::avoidance_marker::createYieldAlertMarkerArray;
  using motion_utils::createDeadLineVirtualWallMarker;
  using motion_utils::createSlowDownVirtualWallMarker;
  using motion_utils::createStopVirtualWallMarker;
  using tier4_autoware_utils::appendMarkerArray;
  using tier4_autoware_utils::calcOffsetPose;

  debug_marker_.markers.clear();
  const auto & base_link2front = planner_data_->parameters.base_link2front;
  const auto current_time = rclcpp::Clock{RCL_ROS_TIME}.now();

  const auto add = [this](const MarkerArray & added) {
    tier4_autoware_utils::appendMarkerArray(added, &debug_marker_);
  };

  const auto addAvoidPoint =
    [&](const AvoidPointArray & ap_arr, const auto & ns, auto r, auto g, auto b, double w = 0.1) {
      add(createAvoidPointMarkerArray(ap_arr, ns, r, g, b, w));
    };

  const auto addShiftPoint =
    [&](const ShiftPointArray & sp_arr, const auto & ns, auto r, auto g, auto b, double w = 0.1) {
      add(createShiftPointMarkerArray(sp_arr, shifter.getBaseOffset(), ns, r, g, b, w));
    };

  const auto & path = data.reference_path;
  add(createPathMarkerArray(debug.center_line, "centerline", 0, 0.0, 0.5, 0.9));
  add(createPathMarkerArray(path, "centerline_resampled", 0, 0.0, 0.9, 0.5));
  add(createPathMarkerArray(prev_linear_shift_path_.path, "prev_linear_shift", 0, 0.5, 0.4, 0.6));
  add(createPoseMarkerArray(data.reference_pose, "reference_pose", 0, 0.9, 0.3, 0.3));

  if (debug.stop_pose) {
    const auto p_front = calcOffsetPose(debug.stop_pose.get(), base_link2front, 0.0, 0.0);
    appendMarkerArray(
      createStopVirtualWallMarker(p_front, "avoidance stop", current_time, 0L), &debug_marker_);
  }

  if (debug.slow_pose) {
    const auto p_front = calcOffsetPose(debug.slow_pose.get(), base_link2front, 0.0, 0.0);
    appendMarkerArray(
      createSlowDownVirtualWallMarker(p_front, "avoidance slow", current_time, 0L), &debug_marker_);
  }

  if (debug.feasible_bound) {
    const auto p_front = calcOffsetPose(debug.feasible_bound.get(), base_link2front, 0.0, 0.0);
    appendMarkerArray(
      createDeadLineVirtualWallMarker(p_front, "feasible bound", current_time, 0L), &debug_marker_);
  }

  add(createYieldAlertMarkerArray(
    data.state, getEgoPose().pose, debug.margin_data_array, debug.exist_adjacent_objects));

  add(createEgoStatusMarkerArray(data, getEgoPose().pose, "ego_status"));

  add(createLaneletsAreaMarkerArray(*debug.current_lanelets, "current_lanelet", 0.0, 1.0, 0.0));
  add(createLaneletsAreaMarkerArray(*debug.expanded_lanelets, "expanded_lanelet", 0.8, 0.8, 0.0));
  add(createTargetObjectsMarkerArray(data.target_objects, "target_objects"));

  add(createOtherObjectsMarkerArray(
    data.ignore_objects, AvoidanceDebugFactor::OBJECT_IS_BEHIND_THRESHOLD));
  add(createOtherObjectsMarkerArray(data.ignore_objects, AvoidanceDebugFactor::OBJECT_IS_NOT_TYPE));
  add(createOtherObjectsMarkerArray(
    data.ignore_objects, AvoidanceDebugFactor::OBJECT_IS_IN_FRONT_THRESHOLD));
  add(createOtherObjectsMarkerArray(
    data.ignore_objects, AvoidanceDebugFactor::OBJECT_BEHIND_PATH_GOAL));
  add(createOtherObjectsMarkerArray(
    data.ignore_objects, AvoidanceDebugFactor::TOO_NEAR_TO_CENTERLINE));
  add(createOtherObjectsMarkerArray(data.ignore_objects, AvoidanceDebugFactor::NOT_PARKING_OBJECT));
  add(createOtherObjectsMarkerArray(data.ignore_objects, AvoidanceDebugFactor::MERGING_OBJECT));
  add(createOtherObjectsMarkerArray(data.ignore_objects, std::string("MovingObject")));
  add(createOtherObjectsMarkerArray(data.ignore_objects, std::string("OutOfTargetArea")));

  add(createUnavoidableObjectsMarkerArray(debug.unavoidable_objects, "unavoidable_objects"));
  add(createUnsafeObjectsMarkerArray(debug.unsafe_objects, "unsafe_objects"));
  add(createPredictedVehiclePositions(debug.velocity_smoothed_path, "predicted_vehicle_positions"));
  add(createOverhangFurthestLineStringMarkerArray(
    *debug.farthest_linestring_from_overhang, "farthest_linestring_from_overhang", 1.0, 0.0, 1.0));

  // parent object info
  addAvoidPoint(debug.registered_raw_shift, "p_registered_shift", 0.8, 0.8, 0.0);
  addAvoidPoint(debug.current_raw_shift, "p_current_raw_shift", 0.5, 0.2, 0.2);
  addAvoidPoint(debug.extra_return_shift, "p_extra_return_shift", 0.0, 0.5, 0.8);

  // merged shift
  const auto & linear_shift = prev_linear_shift_path_.shift_length;
  add(createShiftLengthMarkerArray(debug.pos_shift, path, "m_pos_shift_line", 0, 0.7, 0.5));
  add(createShiftLengthMarkerArray(debug.neg_shift, path, "m_neg_shift_line", 0, 0.5, 0.7));
  add(createShiftLengthMarkerArray(debug.total_shift, path, "m_total_shift_line", 0.99, 0.4, 0.2));
  add(createShiftLengthMarkerArray(debug.output_shift, path, "m_output_shift_line", 0.8, 0.8, 0.2));
  add(createShiftLengthMarkerArray(linear_shift, path, "m_output_linear_line", 0.9, 0.3, 0.3));

  // child shift points
  addAvoidPoint(debug.merged, "c_0_merged", 0.345, 0.968, 1.0);
  addAvoidPoint(debug.trim_similar_grad_shift, "c_1_trim_similar_grad_shift", 0.976, 0.328, 0.910);
  addAvoidPoint(debug.quantized, "c_2_quantized", 0.505, 0.745, 0.969);
  addAvoidPoint(debug.trim_small_shift, "c_3_trim_small_shift", 0.663, 0.525, 0.941);
  addAvoidPoint(
    debug.trim_similar_grad_shift_second, "c_4_trim_similar_grad_shift", 0.97, 0.32, 0.91);
  addAvoidPoint(debug.trim_momentary_return, "c_5_trim_momentary_return", 0.976, 0.078, 0.878);
  addAvoidPoint(debug.trim_too_sharp_shift, "c_6_trim_too_sharp_shift", 0.576, 0.0, 0.978);

  addShiftPoint(shifter.getShiftPoints(), "path_shifter_registered_points", 0.99, 0.99, 0.0, 0.5);
  addAvoidPoint(debug.new_shift_points, "path_shifter_proposed_points", 0.99, 0.0, 0.0, 0.5);
  addAvoidPoint(debug.unfeasible_shift, "unfeasible_shift", 1.0, 1.0, 1.0);
}

void AvoidanceModule::updateAvoidanceDebugData(
  std::vector<AvoidanceDebugMsg> & avoidance_debug_msg_array) const
{
  debug_data_.avoidance_debug_msg_array.avoidance_info.clear();
  auto & debug_data_avoidance = debug_data_.avoidance_debug_msg_array.avoidance_info;
  debug_data_avoidance = avoidance_debug_msg_array;
  if (!debug_avoidance_initializer_for_shift_point_.empty()) {
    const bool is_info_old_ =
      (clock_->now() - debug_avoidance_initializer_for_shift_point_time_).seconds() > 0.1;
    if (!is_info_old_) {
      debug_data_avoidance.insert(
        debug_data_avoidance.end(), debug_avoidance_initializer_for_shift_point_.begin(),
        debug_avoidance_initializer_for_shift_point_.end());
    }
  }
}

}  // namespace behavior_path_planner
