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

#include "behavior_path_planner/path_utilities.hpp"
#include "behavior_path_planner/scene_module/avoidance/avoidance_module.hpp"
#include "behavior_path_planner/scene_module/avoidance/avoidance_module_data.hpp"
#include "behavior_path_planner/utilities.hpp"

#include <autoware_auto_tf2/tf2_autoware_auto_msgs.hpp>
#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/query.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace behavior_path_planner
{

using motion_utils::calcLongitudinalOffsetPoint;
using motion_utils::findNearestSegmentIndex;
using motion_utils::insertTargetPoint;
using tier4_autoware_utils::calcDistance2d;
using tier4_autoware_utils::calcOffsetPose;
using tier4_autoware_utils::createQuaternionFromRPY;
using tier4_autoware_utils::getPose;
using tier4_autoware_utils::pose2transform;

namespace
{

geometry_msgs::msg::Point32 createPoint32(const double x, const double y, const double z)
{
  geometry_msgs::msg::Point32 p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

// geometry_msgs::msg::Polygon toMsg(const Polygon & polygon, const double z)
geometry_msgs::msg::Polygon toMsg(const tier4_autoware_utils::Polygon2d & polygon, const double z)
{
  geometry_msgs::msg::Polygon ret;
  for (const auto & p : polygon.outer()) {
    ret.points.push_back(createPoint32(p.x(), p.y(), z));
  }
  return ret;
}

bool validCheckDecelPlan(
  const double v_end, const double a_end, const double v_target, const double a_target,
  const double v_margin, const double a_margin)
{
  const double v_min = v_target - std::abs(v_margin);
  const double v_max = v_target + std::abs(v_margin);
  const double a_min = a_target - std::abs(a_margin);
  const double a_max = a_target + std::abs(a_margin);

  if (v_end < v_min || v_max < v_end) {
    RCLCPP_DEBUG_STREAM(
      rclcpp::get_logger("validCheckDecelPlan"),
      "[validCheckDecelPlan] valid check error! v_target = " << v_target << ", v_end = " << v_end);
    return false;
  }
  if (a_end < a_min || a_max < a_end) {
    RCLCPP_DEBUG_STREAM(
      rclcpp::get_logger("validCheckDecelPlan"),
      "[validCheckDecelPlan] valid check error! a_target = " << a_target << ", a_end = " << a_end);
    return false;
  }

  return true;
}

/**
 * @brief calculate distance until velocity is reached target velocity (TYPE1)
 * @param (v0) current velocity [m/s]
 * @param (vt) target velocity [m/s]
 * @param (a0) current acceleration [m/ss]
 * @param (am) minimum deceleration [m/ss]
 * @param (ja) maximum jerk [m/sss]
 * @param (jd) minimum jerk [m/sss]
 * @param (t_min) duration of constant deceleration [s]
 * @return moving distance until velocity is reached vt [m]
 * @detail TODO(Satoshi Ota)
 */
boost::optional<double> calcDecelDistPlanType1(
  const double v0, const double vt, const double a0, const double am, const double ja,
  const double jd, const double t_min)
{
  constexpr double epsilon = 1e-3;

  const double j1 = am < a0 ? jd : ja;
  const double t1 = epsilon < (am - a0) / j1 ? (am - a0) / j1 : 0.0;
  const double a1 = a0 + j1 * t1;
  const double v1 = v0 + a0 * t1 + 0.5 * j1 * t1 * t1;
  const double x1 = v0 * t1 + 0.5 * a0 * t1 * t1 + (1.0 / 6.0) * j1 * t1 * t1 * t1;

  const double t2 = epsilon < t_min ? t_min : 0.0;
  const double a2 = a1;
  const double v2 = v1 + a1 * t2;
  const double x2 = x1 + v1 * t2 + 0.5 * a1 * t2 * t2;

  const double t3 = epsilon < (0.0 - am) / ja ? (0.0 - am) / ja : 0.0;
  const double a3 = a2 + ja * t3;
  const double v3 = v2 + a2 * t3 + 0.5 * ja * t3 * t3;
  const double x3 = x2 + v2 * t3 + 0.5 * a2 * t3 * t3 + (1.0 / 6.0) * ja * t3 * t3 * t3;

  const double a_target = 0.0;
  const double v_margin = 0.3;  // [m/s]
  const double a_margin = 0.1;  // [m/s^2]

  if (!validCheckDecelPlan(v3, a3, vt, a_target, v_margin, a_margin)) {
    return {};
  }

  return x3;
}

/**
 * @brief calculate distance until velocity is reached target velocity (TYPE2)
 * @param (v0) current velocity [m/s]
 * @param (vt) target velocity [m/s]
 * @param (a0) current acceleration [m/ss]
 * @param (am) minimum deceleration [m/ss]
 * @param (ja) maximum jerk [m/sss]
 * @param (jd) minimum jerk [m/sss]
 * @return moving distance until velocity is reached vt [m]
 * @detail TODO(Satoshi Ota)
 */
boost::optional<double> calcDecelDistPlanType2(
  const double v0, const double vt, const double a0, const double ja, const double jd)
{
  constexpr double epsilon = 1e-3;

  const double a1_square = (vt - v0 - 0.5 * (0.0 - a0) / jd * a0) * (2.0 * ja * jd / (ja - jd));
  const double a1 = -std::sqrt(a1_square);

  const double t1 = epsilon < (a1 - a0) / jd ? (a1 - a0) / jd : 0.0;
  const double v1 = v0 + a0 * t1 + 0.5 * jd * t1 * t1;
  const double x1 = v0 * t1 + 0.5 * a0 * t1 * t1 + (1.0 / 6.0) * jd * t1 * t1 * t1;

  const double t2 = epsilon < (0.0 - a1) / ja ? (0.0 - a1) / ja : 0.0;
  const double a2 = a1 + ja * t2;
  const double v2 = v1 + a1 * t2 + 0.5 * ja * t2 * t2;
  const double x2 = x1 + v1 * t2 + 0.5 * a1 * t2 * t2 + (1.0 / 6.0) * ja * t2 * t2 * t2;

  const double a_target = 0.0;
  const double v_margin = 0.3;
  const double a_margin = 0.1;

  if (!validCheckDecelPlan(v2, a2, vt, a_target, v_margin, a_margin)) {
    return {};
  }

  return x2;
}

/**
 * @brief calculate distance until velocity is reached target velocity (TYPE3)
 * @param (v0) current velocity [m/s]
 * @param (vt) target velocity [m/s]
 * @param (a0) current acceleration [m/ss]
 * @param (ja) maximum jerk [m/sss]
 * @return moving distance until velocity is reached vt [m]
 * @detail TODO(Satoshi Ota)
 */
boost::optional<double> calcDecelDistPlanType3(
  const double v0, const double vt, const double a0, const double ja)
{
  constexpr double epsilon = 1e-3;

  const double t_acc = (0.0 - a0) / ja;

  const double t1 = epsilon < t_acc ? t_acc : 0.0;
  const double a1 = a0 + ja * t1;
  const double v1 = v0 + a0 * t1 + 0.5 * ja * t1 * t1;
  const double x1 = v0 * t1 + 0.5 * a0 * t1 * t1 + (1.0 / 6.0) * ja * t1 * t1 * t1;

  const double a_target = 0.0;
  const double v_margin = 0.3;
  const double a_margin = 0.1;

  if (!validCheckDecelPlan(v1, a1, vt, a_target, v_margin, a_margin)) {
    return {};
  }

  return x1;
}
}  // namespace

bool isOnRight(const ObjectData & obj) { return obj.lateral < 0.0; }

double calcShiftLength(
  const bool & is_object_on_right, const double & overhang_dist, const double & avoid_margin)
{
  const auto shift_length =
    is_object_on_right ? (overhang_dist + avoid_margin) : (overhang_dist - avoid_margin);
  return std::fabs(shift_length) > 1e-3 ? shift_length : 0.0;
}

bool isSameDirectionShift(const bool & is_object_on_right, const double & shift_length)
{
  return (is_object_on_right == std::signbit(shift_length));
}

lanelet::ConstLanelets calcLaneAroundPose(
  const std::shared_ptr<const PlannerData> & planner_data, const geometry_msgs::msg::Pose & pose,
  const double backward_length)
{
  const auto & p = planner_data->parameters;
  const auto & route_handler = planner_data->route_handler;

  lanelet::ConstLanelet current_lane;
  if (!route_handler->getClosestLaneletWithinRoute(pose, &current_lane)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("behavior_path_planner").get_child("avoidance"),
      "failed to find closest lanelet within route!!!");
    return {};  // TODO(Horibe)
  }

  // For current_lanes with desired length
  lanelet::ConstLanelets current_lanes =
    route_handler->getLaneletSequence(current_lane, pose, backward_length, p.forward_path_length);

  return current_lanes;
}

ShiftedPath toShiftedPath(const PathWithLaneId & path)
{
  ShiftedPath out;
  out.path = path;
  out.shift_length.resize(path.points.size());
  std::fill(out.shift_length.begin(), out.shift_length.end(), 0.0);
  return out;
}

ShiftPointArray toShiftPointArray(const AvoidPointArray & avoid_points)
{
  ShiftPointArray shift_points;
  for (const auto & ap : avoid_points) {
    shift_points.push_back(ap);
  }
  return shift_points;
}

size_t findPathIndexFromArclength(
  const std::vector<double> & path_arclength_arr, const double target_arc)
{
  if (path_arclength_arr.empty()) {
    return 0;
  }

  for (size_t i = 0; i < path_arclength_arr.size(); ++i) {
    if (path_arclength_arr.at(i) > target_arc) {
      return i;
    }
  }
  return path_arclength_arr.size() - 1;
}

std::vector<size_t> concatParentIds(
  const std::vector<size_t> & ids1, const std::vector<size_t> & ids2)
{
  std::set<size_t> id_set{ids1.begin(), ids1.end()};
  for (const auto id : ids2) {
    id_set.insert(id);
  }
  const auto v = std::vector<size_t>{id_set.begin(), id_set.end()};
  return v;
}

double lerpShiftLengthOnArc(double arc, const AvoidPoint & ap)
{
  if (ap.start_longitudinal <= arc && arc < ap.end_longitudinal) {
    if (std::abs(ap.getRelativeLongitudinal()) < 1.0e-5) {
      return ap.length;
    }
    const auto start_weight = (ap.end_longitudinal - arc) / ap.getRelativeLongitudinal();
    return start_weight * ap.start_length + (1.0 - start_weight) * ap.length;
  }
  return 0.0;
}

void clipByMinStartIdx(const AvoidPointArray & shift_points, PathWithLaneId & path)
{
  if (path.points.empty()) {
    return;
  }

  size_t min_start_idx = std::numeric_limits<size_t>::max();
  for (const auto & sp : shift_points) {
    min_start_idx = std::min(min_start_idx, sp.start_idx);
  }
  min_start_idx = std::min(min_start_idx, path.points.size() - 1);
  path.points =
    std::vector<PathPointWithLaneId>{path.points.begin() + min_start_idx, path.points.end()};
}

void fillLongitudinalAndLengthByClosestFootprint(
  const PathWithLaneId & path, const PredictedObject & object, const Point & ego_pos,
  ObjectData & obj)
{
  tier4_autoware_utils::Polygon2d object_poly{};
  util::calcObjectPolygon(object, &object_poly);

  const double distance = motion_utils::calcSignedArcLength(
    path.points, ego_pos, object.kinematics.initial_pose_with_covariance.pose.position);
  double min_distance = distance;
  double max_distance = distance;
  for (const auto & p : object_poly.outer()) {
    const auto point = tier4_autoware_utils::createPoint(p.x(), p.y(), 0.0);
    const double arc_length = motion_utils::calcSignedArcLength(path.points, ego_pos, point);
    min_distance = std::min(min_distance, arc_length);
    max_distance = std::max(max_distance, arc_length);
  }
  obj.longitudinal = min_distance;
  obj.length = max_distance - min_distance;
  return;
}

void fillLongitudinalAndLengthByClosestEnvelopeFootprint(
  const PathWithLaneId & path, const Point & ego_pos, ObjectData & obj)
{
  const double distance = motion_utils::calcSignedArcLength(
    path.points, ego_pos, obj.object.kinematics.initial_pose_with_covariance.pose.position);
  double min_distance = distance;
  double max_distance = distance;
  for (const auto & p : obj.envelope_poly.outer()) {
    const auto point = tier4_autoware_utils::createPoint(p.x(), p.y(), 0.0);
    const double arc_length = motion_utils::calcSignedArcLength(path.points, ego_pos, point);
    min_distance = std::min(min_distance, arc_length);
    max_distance = std::max(max_distance, arc_length);
  }
  obj.longitudinal = min_distance;
  obj.length = max_distance - min_distance;
  return;
}

double calcEnvelopeOverhangDistance(
  const ObjectData & object_data, const Pose & base_pose, Point & overhang_pose)
{
  double largest_overhang = isOnRight(object_data) ? -100.0 : 100.0;  // large number

  for (const auto & p : object_data.envelope_poly.outer()) {
    const auto point = tier4_autoware_utils::createPoint(p.x(), p.y(), 0.0);
    const auto lateral = tier4_autoware_utils::calcLateralDeviation(base_pose, point);

    const auto & overhang_pose_on_right = [&overhang_pose, &largest_overhang, &point, &lateral]() {
      if (lateral > largest_overhang) {
        overhang_pose = point;
      }
      return std::max(largest_overhang, lateral);
    };

    const auto & overhang_pose_on_left = [&overhang_pose, &largest_overhang, &point, &lateral]() {
      if (lateral < largest_overhang) {
        overhang_pose = point;
      }
      return std::min(largest_overhang, lateral);
    };

    largest_overhang = isOnRight(object_data) ? overhang_pose_on_right() : overhang_pose_on_left();
  }
  return largest_overhang;
}

double calcOverhangDistance(
  const ObjectData & object_data, const Pose & base_pose, Point & overhang_pose)
{
  double largest_overhang = isOnRight(object_data) ? -100.0 : 100.0;  // large number

  tier4_autoware_utils::Polygon2d object_poly{};
  util::calcObjectPolygon(object_data.object, &object_poly);

  for (const auto & p : object_poly.outer()) {
    const auto point = tier4_autoware_utils::createPoint(p.x(), p.y(), 0.0);
    const auto lateral = tier4_autoware_utils::calcLateralDeviation(base_pose, point);

    const auto & overhang_pose_on_right = [&overhang_pose, &largest_overhang, &point, &lateral]() {
      if (lateral > largest_overhang) {
        overhang_pose = point;
      }
      return std::max(largest_overhang, lateral);
    };

    const auto & overhang_pose_on_left = [&overhang_pose, &largest_overhang, &point, &lateral]() {
      if (lateral < largest_overhang) {
        overhang_pose = point;
      }
      return std::min(largest_overhang, lateral);
    };

    largest_overhang = isOnRight(object_data) ? overhang_pose_on_right() : overhang_pose_on_left();
  }
  return largest_overhang;
}

void setEndData(
  AvoidPoint & ap, const double length, const geometry_msgs::msg::Pose & end, const size_t end_idx,
  const double end_dist)
{
  ap.length = length;
  ap.end = end;
  ap.end_idx = end_idx;
  ap.end_longitudinal = end_dist;
}

void setStartData(
  AvoidPoint & ap, const double start_length, const geometry_msgs::msg::Pose & start,
  const size_t start_idx, const double start_dist)
{
  ap.start_length = start_length;
  ap.start = start;
  ap.start_idx = start_idx;
  ap.start_longitudinal = start_dist;
}

std::string getUuidStr(const ObjectData & obj)
{
  std::stringstream hex_value;
  for (const auto & uuid : obj.object.object_id.uuid) {
    hex_value << std::hex << std::setfill('0') << std::setw(2) << +uuid;
  }
  return hex_value.str();
}

std::vector<std::string> getUuidStr(const ObjectDataArray & objs)
{
  std::vector<std::string> uuids;
  uuids.reserve(objs.size());
  for (const auto & o : objs) {
    uuids.push_back(getUuidStr(o));
  }
  return uuids;
}

Polygon2d createEnvelopePolygon(
  const ObjectData & object_data, const Pose & closest_pose, const double envelope_buffer)
{
  namespace bg = boost::geometry;
  using tier4_autoware_utils::Point2d;
  using tier4_autoware_utils::Polygon2d;
  using Box = bg::model::box<Point2d>;

  Polygon2d object_polygon{};
  util::calcObjectPolygon(object_data.object, &object_polygon);

  const auto toPolygon2d = [](const geometry_msgs::msg::Polygon & polygon) {
    Polygon2d ret{};

    for (const auto & p : polygon.points) {
      ret.outer().push_back(Point2d(p.x, p.y));
    }

    return ret;
  };

  Pose pose_2d = closest_pose;
  pose_2d.orientation = createQuaternionFromRPY(0.0, 0.0, tf2::getYaw(closest_pose.orientation));

  TransformStamped geometry_tf{};
  geometry_tf.transform = pose2transform(pose_2d);

  tf2::Transform tf;
  tf2::fromMsg(geometry_tf.transform, tf);
  TransformStamped inverse_geometry_tf{};
  inverse_geometry_tf.transform = tf2::toMsg(tf.inverse());

  geometry_msgs::msg::Polygon out_ros_polygon{};
  tf2::doTransform(
    toMsg(object_polygon, closest_pose.position.z), out_ros_polygon, inverse_geometry_tf);

  const auto envelope_box = bg::return_envelope<Box>(toPolygon2d(out_ros_polygon));

  Polygon2d envelope_poly{};
  bg::convert(envelope_box, envelope_poly);

  geometry_msgs::msg::Polygon envelope_ros_polygon{};
  tf2::doTransform(
    toMsg(envelope_poly, closest_pose.position.z), envelope_ros_polygon, geometry_tf);

  std::vector<Polygon2d> offset_polygons{};
  bg::strategy::buffer::distance_symmetric<double> distance_strategy(envelope_buffer);
  bg::strategy::buffer::join_miter join_strategy;
  bg::strategy::buffer::end_flat end_strategy;
  bg::strategy::buffer::side_straight side_strategy;
  bg::strategy::buffer::point_circle point_strategy;
  bg::buffer(
    toPolygon2d(envelope_ros_polygon), offset_polygons, distance_strategy, side_strategy,
    join_strategy, end_strategy, point_strategy);

  return offset_polygons.front();
}

lanelet::ConstLanelets getTargetLanelets(
  const std::shared_ptr<const PlannerData> & planner_data, lanelet::ConstLanelets & route_lanelets,
  const double left_offset, const double right_offset)
{
  const auto & rh = planner_data->route_handler;

  lanelet::ConstLanelets target_lanelets{};
  for (const auto & lane : route_lanelets) {
    auto l_offset = 0.0;
    auto r_offset = 0.0;

    const auto opt_left_lane = rh->getLeftLanelet(lane);
    if (opt_left_lane) {
      target_lanelets.push_back(opt_left_lane.get());
    } else {
      l_offset = left_offset;
    }

    const auto opt_right_lane = rh->getRightLanelet(lane);
    if (opt_right_lane) {
      target_lanelets.push_back(opt_right_lane.get());
    } else {
      r_offset = right_offset;
    }

    const auto expand_lane = lanelet::utils::getExpandedLanelet(lane, l_offset, r_offset);
    target_lanelets.push_back(expand_lane);
  }

  return target_lanelets;
}

void insertDecelPoint(
  const Point & p_src, const double offset, const double velocity, PathWithLaneId & path,
  boost::optional<Pose> & p_out)
{
  const auto decel_point = calcLongitudinalOffsetPoint(path.points, p_src, offset);

  if (!decel_point) {
    // TODO(Satoshi OTA)  Think later the process in the case of no decel point found.
    return;
  }

  const auto seg_idx = findNearestSegmentIndex(path.points, decel_point.get());
  const auto insert_idx = insertTargetPoint(seg_idx, decel_point.get(), path.points);

  if (!insert_idx) {
    // TODO(Satoshi OTA)  Think later the process in the case of no decel point found.
    return;
  }

  const auto insertVelocity = [&insert_idx](PathWithLaneId & path, const float v) {
    for (size_t i = insert_idx.get(); i < path.points.size(); ++i) {
      const auto & original_velocity = path.points.at(i).point.longitudinal_velocity_mps;
      path.points.at(i).point.longitudinal_velocity_mps = std::min(original_velocity, v);
    }
  };

  insertVelocity(path, velocity);

  p_out = getPose(path.points.at(insert_idx.get()));
}

bool isCentroidWithinLanelets(
  const PredictedObject & object, const lanelet::ConstLanelets & target_lanelets)
{
  if (target_lanelets.empty()) {
    return false;
  }

  const auto & object_pos = object.kinematics.initial_pose_with_covariance.pose.position;
  lanelet::BasicPoint2d object_centroid(object_pos.x, object_pos.y);

  for (const auto & llt : target_lanelets) {
    if (boost::geometry::within(object_centroid, llt.polygon2d().basicPolygon())) {
      return true;
    }
  }

  return false;
}

boost::optional<double> calcDecelDistWithJerkAndAccConstraints(
  const double current_vel, const double target_vel, const double current_acc, const double acc_min,
  const double jerk_acc, const double jerk_dec)
{
  constexpr double epsilon = 1e-3;
  const double t_dec =
    acc_min < current_acc ? (acc_min - current_acc) / jerk_dec : (acc_min - current_acc) / jerk_acc;
  const double t_acc = (0.0 - acc_min) / jerk_acc;
  const double t_min = (target_vel - current_vel - current_acc * t_dec -
                        0.5 * jerk_dec * t_dec * t_dec - 0.5 * acc_min * t_acc) /
                       acc_min;

  // check if it is possible to decelerate to the target velocity
  // by simply bringing the current acceleration to zero.
  const auto is_decel_needed =
    0.5 * (0.0 - current_acc) / jerk_acc * current_acc > target_vel - current_vel;

  if (t_min > epsilon) {
    return calcDecelDistPlanType1(
      current_vel, target_vel, current_acc, acc_min, jerk_acc, jerk_dec, t_min);
  } else if (is_decel_needed || current_acc > epsilon) {
    return calcDecelDistPlanType2(current_vel, target_vel, current_acc, jerk_acc, jerk_dec);
  } else {
    return calcDecelDistPlanType3(current_vel, target_vel, current_acc, jerk_acc);
  }

  return {};
}
}  // namespace behavior_path_planner
