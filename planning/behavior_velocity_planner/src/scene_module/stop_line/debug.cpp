// Copyright 2020 Tier IV, Inc.
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

#include <scene_module/stop_line/scene.hpp>
#include <utilization/marker_helper.hpp>
#include <utilization/util.hpp>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

namespace behavior_velocity_planner
{
namespace
{
using DebugData = StopLineModule::DebugData;

visualization_msgs::msg::MarkerArray createMarkers(
  const DebugData & debug_data, const int64_t module_id)
{
  visualization_msgs::msg::MarkerArray msg;
  tf2::Transform tf_base_link2front(
    tf2::Quaternion(0.0, 0.0, 0.0, 1.0), tf2::Vector3(debug_data.base_link2front, 0.0, 0.0));

  // Stop VirtualWall
  if (debug_data.stop_pose) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.ns = "stop_virtual_wall";
    marker.id = module_id;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    tf2::Transform tf_map2base_link;
    tf2::fromMsg(*debug_data.stop_pose, tf_map2base_link);
    tf2::Transform tf_map2front = tf_map2base_link * tf_base_link2front;
    tf2::toMsg(tf_map2front, marker.pose);
    marker.pose.position.z += 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 5.0;
    marker.scale.z = 2.0;
    marker.color.a = 0.5;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    msg.markers.push_back(marker);
  }

  // Factor Text
  if (debug_data.stop_pose) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.ns = "factor_text";
    marker.id = module_id;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    tf2::Transform tf_map2base_link;
    tf2::fromMsg(*debug_data.stop_pose, tf_map2base_link);
    tf2::Transform tf_map2front = tf_map2base_link * tf_base_link2front;
    tf2::toMsg(tf_map2front, marker.pose);
    marker.pose.position.z += 2.0;
    marker.scale.x = 0.0;
    marker.scale.y = 0.0;
    marker.scale.z = 1.0;
    marker.color.a = 0.999;  // Don't forget to set the alpha!
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.text = "stop line";
    msg.markers.push_back(marker);
  }

  return msg;
}

visualization_msgs::msg::MarkerArray createStopLineCollisionCheck(
  const DebugData & debug_data, const int64_t module_id)
{
  visualization_msgs::msg::MarkerArray msg;

  // Search Segments
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.ns = "search_segments";
    marker.id = module_id;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    for (const auto & e : debug_data.search_segments) {
      marker.points.push_back(
        geometry_msgs::build<geometry_msgs::msg::Point>().x(e.at(0).x()).y(e.at(0).y()).z(0.0));
      marker.points.push_back(
        geometry_msgs::build<geometry_msgs::msg::Point>().x(e.at(1).x()).y(e.at(1).y()).z(0.0));
    }
    marker.scale = createMarkerScale(0.1, 0.1, 0.1);
    marker.color = createMarkerColor(0.0, 0.0, 1.0, 0.999);
    msg.markers.push_back(marker);
  }

  // Search stopline
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.ns = "search_stopline";
    marker.id = module_id;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    const auto p0 = debug_data.search_stopline.at(0);
    marker.points.push_back(
      geometry_msgs::build<geometry_msgs::msg::Point>().x(p0.x()).y(p0.y()).z(0.0));
    const auto p1 = debug_data.search_stopline.at(1);
    marker.points.push_back(
      geometry_msgs::build<geometry_msgs::msg::Point>().x(p1.x()).y(p1.y()).z(0.0));

    marker.scale = createMarkerScale(0.1, 0.1, 0.1);
    marker.color = createMarkerColor(1.0, 0.0, 0.0, 0.999);
    msg.markers.push_back(marker);
  }

  return msg;
}

}  // namespace

visualization_msgs::msg::MarkerArray StopLineModule::createDebugMarkerArray()
{
  visualization_msgs::msg::MarkerArray debug_marker_array;

  appendMarkerArray(
    createMarkers(debug_data_, module_id_), this->clock_->now(), &debug_marker_array);

  if (planner_param_.show_stopline_collision_check) {
    appendMarkerArray(
      createStopLineCollisionCheck(debug_data_, module_id_), this->clock_->now(),
      &debug_marker_array);
  }

  return debug_marker_array;
}
}  // namespace behavior_velocity_planner
