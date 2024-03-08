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

#include "metrics_visualize_panel.hpp"

#include <rviz_common/display_context.hpp>

#include <X11/Xlib.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace rviz_plugins
{
MetricsVisualizePanel::MetricsVisualizePanel(QWidget * parent)
: rviz_common::Panel(parent), grid_(new QGridLayout())
{
  // 新しいQTabWidgetのインスタンスを作成
  tab_widget_ = new QTabWidget();  // QTabWidgetを初期化

  // 全メトリクスを表示するためのタブとして、既存のgrid_レイアウトを含むQWidgetを作成
  QWidget * all_metrics_widget = new QWidget();
  grid_ = new QGridLayout();
  all_metrics_widget->setLayout(grid_);

  // トピックセレクターを追加
  topic_selector_ = new QComboBox();
  for (const auto & topic : topics_) {
    topic_selector_->addItem(QString::fromStdString(topic));
  }
  grid_->addWidget(topic_selector_, 0, 0, 1, -1);  // トピックセレクターをgrid_に追加
  connect(topic_selector_, SIGNAL(currentIndexChanged(int)), this, SLOT(onTopicChanged()));

  // 作成したQWidgetをタブとして追加
  tab_widget_->addTab(all_metrics_widget, "All Metrics");

  // 新しい特定のメトリクスを表示するためのタブを作成ここでは(空のQWidgetを追加しています)
  QWidget * specific_metrics_widget = new QWidget();
  QVBoxLayout * specific_metrics_layout = new QVBoxLayout();
  specific_metric_selector_ = new QComboBox();
  specific_metrics_layout->addWidget(specific_metric_selector_);

  specific_metric_table_ = new QTableWidget();
  specific_metrics_layout->addWidget(specific_metric_table_, 1);


  specific_metric_table_->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
specific_metric_table_->horizontalHeader()->setStretchLastSection(true);
specific_metric_table_->verticalHeader()->setVisible(false);  // 垂直ヘッダーを非表示にする
specific_metric_table_->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
specific_metric_table_->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
specific_metrics_layout->setStretch(1, 0); // テーブルは縦方向の伸縮を固定
  

  specific_metric_chart_view_ = new QChartView();
  specific_metrics_layout->addWidget(specific_metric_chart_view_,2);


  


  specific_metrics_widget->setLayout(specific_metrics_layout);

  tab_widget_->addTab(specific_metrics_widget, "Specific Metrics");
  connect(
    specific_metric_selector_, SIGNAL(currentIndexChanged(int)), this,
    SLOT(onSpecificMetricChanged()));

  // タブウィジェットをメインレイアウトに追加
  QVBoxLayout * main_layout = new QVBoxLayout;
  main_layout->addWidget(tab_widget_);
  setLayout(main_layout);
}

void MetricsVisualizePanel::onInitialize()
{
  using std::placeholders::_1;

  raw_node_ = this->getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  for (const auto & topic_name : topics_) {
    const auto callback = [this, topic_name](const DiagnosticArray::ConstSharedPtr msg) {
      this->onMetrics(msg, topic_name);
    };
    const auto subscription =
      raw_node_->create_subscription<DiagnosticArray>(topic_name, rclcpp::QoS{1}, callback);
    subscriptions_[topic_name] = subscription;
  }

  const auto period = std::chrono::milliseconds(static_cast<int64_t>(1e3 / 10));
  timer_ = raw_node_->create_wall_timer(period, [&]() { onTimer(); });
}

void MetricsVisualizePanel::updateWidgetVisibility(
  const std::string & target_topic, const bool show)
{
  for (const auto & [topic_name, metric_widgets_pair] : topic_widgets_map_) {
    const bool is_target_topic = (topic_name == target_topic);
    if ((!is_target_topic && show) || (is_target_topic && !show)) {
      continue;
    }
    for (const auto & [metric, widgets] : metric_widgets_pair) {
      widgets.first->setVisible(show);
      widgets.second->setVisible(show);
    }
  }
}

void MetricsVisualizePanel::showCurrentTopicWidgets()
{
  const std::string current_topic = topic_selector_->currentText().toStdString();
  updateWidgetVisibility(current_topic, true);
}

void MetricsVisualizePanel::hideInactiveTopicWidgets()
{
  const std::string current_topic = topic_selector_->currentText().toStdString();
  updateWidgetVisibility(current_topic, false);
}

void MetricsVisualizePanel::onTopicChanged()
{
  std::lock_guard<std::mutex> message_lock(mutex_);
  hideInactiveTopicWidgets();
  showCurrentTopicWidgets();
}

void MetricsVisualizePanel::onSpecificMetricChanged()
{
    QString selected_metric = specific_metric_selector_->currentText();

    // 選択されたメトリックスがmetrics_マップに存在するか確認
    if (metrics_.find(selected_metric.toStdString()) != metrics_.end()) {
        Metric &metric = metrics_.at(selected_metric.toStdString());
        // チャートビューを更新
        specific_metric_chart_view_->setChart(metric.getChartView()->chart());
    }

    updateSpecificMetricTable();
}

void MetricsVisualizePanel::updateSpecificMetricTable()
{
    QString selected_metric = specific_metric_selector_->currentText();

    // 選択されたメトリックスがmetrics_マップに存在するか確認
    if (metrics_.find(selected_metric.toStdString()) != metrics_.end()) {
        Metric &metric = metrics_.at(selected_metric.toStdString());

        // テーブルビューを更新
        // テーブルの現在の内容をクリア
        specific_metric_table_->clear();
        specific_metric_table_->setRowCount(1); // 仮に1行と仮定します。
        const auto labels = metric.getLabels();
        specific_metric_table_->setColumnCount(labels.size()); // labelsマップのサイズを使用して列数を設定します。

        // ヘッダーラベルを更新
        QStringList headers;
        for (const auto &label : labels) {
            headers << QString::fromStdString(label.first);
        }
        specific_metric_table_->setHorizontalHeaderLabels(headers);

        // labelsマップから値を取得してテーブルに設定
        int col = 0;
        for (const auto &label_item : labels) {
            specific_metric_table_->setItem(0, col, new QTableWidgetItem(label_item.second->text()));
            col++;
        }
    }
}




// void MetricsVisualizePanel::onSpecificMetricChanged()
// {
//     QString selected_metric = specific_metric_selector_->currentText();

//     // 選択されたメトリックスがmetrics_マップに存在するか確認
//     if (metrics_.find(selected_metric.toStdString()) != metrics_.end()) {
//         const Metric &metric = metrics_.at(selected_metric.toStdString());

//         // テーブルの表示を更新する前に、現在のテーブルウィジェットをレイアウトから削除
//         auto *specific_metrics_widget = dynamic_cast<QWidget *>(tab_widget_->widget(1));
//         auto *specific_metrics_layout = dynamic_cast<QVBoxLayout *>(specific_metrics_widget->layout());
//         specific_metrics_layout->removeWidget(specific_metric_table_);

//         // 新しいメトリックスデータでテーブルウィジェットを更新
//         specific_metric_table_ = metric.getTable(); // 新しいテーブルウィジェットを設定

//         // テーブルウィジェットをレイアウトに再度追加
//         specific_metrics_layout->addWidget(specific_metric_table_,1);

//         // 以前のテーブルウィジェットを破棄する必要がある場合（メモリ管理）、適切な処理をここに追加
//         // 注意: specific_metric_table_ の再利用または更新が安全かどうかを検討してください
//         // delete previousTableWidget;  // 以前のテーブルウィジェットを安全に削除できる場合

//         // チャートビューを更新
//         specific_metric_chart_view_->setChart(metric.getChartView()->chart());
//     }
// }



void MetricsVisualizePanel::onTimer()
{
  std::lock_guard<std::mutex> message_lock(mutex_);

  for (auto & [name, metric] : metrics_) {
    metric.updateGraph();
    metric.updateTable();
  }

  updateSpecificMetricTable();
}

void MetricsVisualizePanel::onMetrics(
  const DiagnosticArray::ConstSharedPtr & msg, const std::string & topic_name)
{
  std::lock_guard<std::mutex> message_lock(mutex_);

  const auto time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
  constexpr size_t GRAPH_COL_SIZE = 5;

  for (const auto & status : msg->status) {
    const size_t num_current_metrics = topic_widgets_map_[topic_name].size();
    if (metrics_.count(status.name) == 0) {
      const auto metric = Metric(status);
      metrics_.emplace(status.name, metric);

      // Calculate grid position
      const size_t row = num_current_metrics / GRAPH_COL_SIZE * 2 +
                         2;  // start from 2 to leave space for the topic selector and tab widget
      const size_t col = num_current_metrics % GRAPH_COL_SIZE;

      // Get the widgets from the metric
      const auto tableWidget = metric.getTable();
      const auto chartViewWidget = metric.getChartView();

      // Get the layout for the "All Metrics" tab
      auto all_metrics_widget = dynamic_cast<QWidget *>(tab_widget_->widget(0));
      QGridLayout * all_metrics_layout = dynamic_cast<QGridLayout *>(all_metrics_widget->layout());

      // Add the widgets to the "All Metrics" tab layout
      all_metrics_layout->addWidget(tableWidget, row, col);
      all_metrics_layout->setRowStretch(row, false);
      all_metrics_layout->addWidget(chartViewWidget, row + 1, col);
      all_metrics_layout->setRowStretch(row + 1, true);
      all_metrics_layout->setColumnStretch(col, true);

      // Also add the widgets to the topic_widgets_map_ for easy management
      topic_widgets_map_[topic_name][status.name] = std::make_pair(tableWidget, chartViewWidget);
    }

    metrics_.at(status.name).updateData(time, status);
  }

  QSignalBlocker blocker(specific_metric_selector_);  // 追加中にシグナルをブロック
  for (const auto & status : msg->status) {
    if (specific_metric_selector_->findText(QString::fromStdString(status.name)) == -1) {
      specific_metric_selector_->addItem(QString::fromStdString(status.name));
    }
  }
}

}  // namespace rviz_plugins

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_plugins::MetricsVisualizePanel, rviz_common::Panel)
