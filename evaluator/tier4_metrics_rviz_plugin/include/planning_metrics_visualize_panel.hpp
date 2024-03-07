//  Copyright 2024 TIER IV, Inc. All rights reserved.
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

#ifndef PLANNING_METRICS_VISUALIZE_PANEL_HPP_
#define PLANNING_METRICS_VISUALIZE_PANEL_HPP_

#ifndef Q_MOC_RUN
#include <QChartView>
#include <QColor>
#include <QGridLayout>
#include <QHeaderView>
#include <QLabel>
#include <QLineSeries>
#include <QPainter>
#include <QTableWidget>
#include <QVBoxLayout>
#endif

#include "metrics_visualize_panel.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>

#include <limits>
#include <string>
#include <unordered_map>

namespace rviz_plugins
{

using diagnostic_msgs::msg::DiagnosticArray;
using diagnostic_msgs::msg::DiagnosticStatus;
using diagnostic_msgs::msg::KeyValue;
using QtCharts::QChart;
using QtCharts::QChartView;
using QtCharts::QLineSeries;

class PlanningMetricsVisualizePanel : public MetricsVisualizePanel
{
public:
  explicit PlanningMetricsVisualizePanel(QWidget * parent = nullptr) : MetricsVisualizePanel(parent)
  {
    metrics_topic_ = "/diagnostic/planning_evaluator/metrics";
  }

  virtual ~PlanningMetricsVisualizePanel() {}
};
}  // namespace rviz_plugins

#endif  // PLANNING_METRICS_VISUALIZE_PANEL_HPP_
