//
//  Copyright 2022 TIER IV, Inc. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#ifndef TRAFFIC_LIGHT_PUBLISH_PANEL_HPP_
#define TRAFFIC_LIGHT_PUBLISH_PANEL_HPP_

#include <qt5/QtWidgets/QComboBox>
#include <qt5/QtWidgets/QPushButton>
#include <qt5/QtWidgets/QSpinBox>
#include <qt5/QtWidgets/QTableWidget>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <rviz_common/panel.hpp>

#include <autoware_auto_perception_msgs/msg/traffic_signal_array.hpp>

#include <chrono>

namespace rviz_plugins
{

using autoware_auto_perception_msgs::msg::TrafficLight;
using autoware_auto_perception_msgs::msg::TrafficSignal;
using autoware_auto_perception_msgs::msg::TrafficSignalArray;

class TrafficLightPublishPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit TrafficLightPublishPanel(QWidget * parent = nullptr);
  void onInitialize() override;

public Q_SLOTS:
  void onRateChanged(int new_rate);
  void onSetTrafficLightState();
  void onResetTrafficLightState();
  void onPublishTrafficLightState();

protected:
  void onTimer();
  void createWallTimer();

  rclcpp::Node::SharedPtr raw_node_;
  rclcpp::TimerBase::SharedPtr pub_timer_;
  rclcpp::Publisher<TrafficSignalArray>::SharedPtr pub_traffic_signals_;

  QSpinBox * publishing_rate_input_;
  QSpinBox * traffic_light_id_input_;
  QComboBox * light_color_combo_;
  QPushButton * set_button_;
  QPushButton * reset_button_;
  QPushButton * publish_button_;
  QTableWidget * traffic_table_;

  TrafficSignalArray extra_traffic_signals_;

  bool enable_publish_;
};

}  // namespace rviz_plugins

#endif  // TRAFFIC_LIGHT_PUBLISH_PANEL_HPP_
