#ifndef VEHICLE_ADAPTOR_COMPENSATOR_H
#define VEHICLE_ADAPTOR_COMPENSATOR_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "inputs_prediction.h"
#include "nominal_dynamics.h"
#include "transform_vehicle_adaptor_model.h"
#include <pybind11/stl.h>
#include "linear_regression_compensator.h"
#include "inputs_ref_smoother.h"
#include "vehicle_adaptor_utils.h"

Eigen::VectorXd states_vehicle_to_world(Eigen::VectorXd states_vehicle, double yaw);
Eigen::VectorXd states_world_to_vehicle(Eigen::VectorXd states_world, double yaw);

class PolynomialRegressionPredictor
{
private:
  int degree_ = 2;
  int num_samples_ = 10;
  std::vector<double> lambda_ = {0.0, 0.0};
  Eigen::MatrixXd coef_matrix_, prediction_matrix_;
  bool ignore_intercept_ = false;
  double oldest_sample_weight_ = 1.0;
public:
  PolynomialRegressionPredictor();
  virtual ~PolynomialRegressionPredictor();
  void set_params(int degree, int num_samples, std::vector<double> lambda);
  void set_oldest_sample_weight(double oldest_sample_weight);
  void set_ignore_intercept();
  void calc_coef_matrix();
  void calc_prediction_matrix(int horizon_len);
  Eigen::VectorXd predict(Eigen::VectorXd vec);
};

class SgFilter
{
private:
  int degree_;
  int window_size_;
  std::vector<Eigen::VectorXd> sg_vector_left_edge_, sg_vector_right_edge_;
  Eigen::VectorXd sg_vector_center_;

  // Template function to handle both Eigen::MatrixXd and Eigen::VectorXd
  template <typename T>
  std::vector<T> sg_filter_impl(const std::vector<T>& raw_data) {
      int input_len = raw_data.size();
      std::vector<T> result;
      for (int i = 0; i < input_len; i++) {
          T tmp_filtered_vector = T::Zero(raw_data[0].rows(), raw_data[0].cols());
          if (i < window_size_) {
              for (int j = 0; j < i + window_size_ + 1; j++) {
                  tmp_filtered_vector += sg_vector_left_edge_[i][j] * raw_data[j];
              }
          }
          else if (i > input_len - window_size_ - 1) {
              for (int j = 0; j < input_len - i + window_size_; j++) {
                  tmp_filtered_vector += sg_vector_right_edge_[input_len - i - 1][j] * raw_data[i - window_size_ + j];
              }
          }
          else {
              for (int j = 0; j < 2 * window_size_ + 1; j++) {
                  tmp_filtered_vector += sg_vector_center_[j] * raw_data[i - window_size_ + j];
              }
          }
          result.push_back(tmp_filtered_vector);
      }
      return result;
  }
public:
  SgFilter();
  virtual ~SgFilter();
  void set_params(int degree, int window_size);
  void calc_sg_filter_weight();
  std::vector<Eigen::MatrixXd> sg_filter(const std::vector<Eigen::MatrixXd>& raw_data);
  std::vector<Eigen::VectorXd> sg_filter(const std::vector<Eigen::VectorXd>& raw_data);
};
class FilterDiffNN
{
private:
  int state_size_;
  int h_dim_;
  int acc_queue_size_;
  int steer_queue_size_;
  int predict_step_;
  double control_dt_;
  SgFilter sg_filter_;
public:
  FilterDiffNN();
  virtual ~FilterDiffNN();
  void set_sg_filter_params(int degree, int window_size, int state_size, int h_dim, int acc_queue_size, int steer_queue_size, int predict_step, double control_dt);
  void fit_transform_for_NN_diff(
    std::vector<Eigen::MatrixXd> A, std::vector<Eigen::MatrixXd> B, std::vector<Eigen::MatrixXd> C,std::vector<Eigen::MatrixXd> & dF_d_states,
    std::vector<Eigen::MatrixXd> & dF_d_inputs);

};
class ButterworthFilter
{
private:
  int order_;
  std::vector<double> a_;
  std::vector<double> b_;
  std::vector<Eigen::VectorXd> x_;
  std::vector<Eigen::VectorXd> y_;
  bool initialized_ = false;
public:
  ButterworthFilter();
  virtual ~ButterworthFilter();
  void set_params();
  Eigen::VectorXd apply(Eigen::VectorXd input_value);
};
class TrainedDynamics
{
private:
  NominalDynamics nominal_dynamics_;
  TransformModelToEigen transform_model_to_eigen_;
  double wheel_base_ = 2.79;
  double acc_time_delay_ = 0.1;
  double steer_time_delay_ = 0.27;
  double acc_time_constant_ = 0.1;
  double steer_time_constant_ = 0.24;
  int acc_queue_size_ = 15;
  int steer_queue_size_ = 15;

  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;
  int state_size_ = 6;

  int acc_input_start_index_ = 6;
  int acc_input_end_index_ = 6 + acc_queue_size_ - 1;
  int steer_input_start_index_ = 6 + acc_queue_size_;
  int steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
  double control_dt_ = 0.033;
  int predict_step_ = 3;
  double predict_dt_ = control_dt_ * predict_step_;

  double error_decay_rate_ = 0.5;

  int acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
  int steer_delay_step_ =
    std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);

  FilterDiffNN filter_diff_NN_;
  int h_dim_;
  int h_dim_full_;
  int state_dim_ = 6;
  double minimum_steer_diff_ = 0.03;
  double minimum_acc_diff_ = 0.1;

  std::vector<std::string> all_state_name_ = {"x", "y", "vel", "yaw", "acc", "steer"};
  std::vector<std::string> state_component_predicted_ = {"acc", "steer"};
  std::vector<int> state_component_predicted_index_;
  LinearRegressionCompensator linear_regression_compensation_;
public:
  TrainedDynamics();
  virtual ~TrainedDynamics();
  void set_vehicle_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step);
  void set_NN_params(
    const Eigen::MatrixXd & weight_acc_encoder_layer_1, const Eigen::MatrixXd & weight_steer_encoder_layer_1,
    const Eigen::MatrixXd & weight_acc_encoder_layer_2, const Eigen::MatrixXd & weight_steer_encoder_layer_2,
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_ih, const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_encoder_layer_1, const Eigen::VectorXd & bias_steer_encoder_layer_1,
    const Eigen::VectorXd & bias_acc_encoder_layer_2, const Eigen::VectorXd & bias_steer_encoder_layer_2,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const std::vector<Eigen::VectorXd> & bias_lstm_encoder_ih, const std::vector<Eigen::VectorXd> & bias_lstm_encoder_hh,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted);
  void clear_NN_params();
  void set_sg_filter_params(int degree, int window_size);
  Eigen::VectorXd nominal_prediction(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat);
  void update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    std::vector<Eigen::VectorXd> & h_lstm, std::vector<Eigen::VectorXd> & c_lstm, const Eigen::Vector2d & acc_steer_error);
  void initialize_compensation(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim);
  void update_state_queue_for_compensation(Eigen::VectorXd states);
  void update_one_step_for_compensation(Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm, Eigen::VectorXd error_vector);
  void update_regression_matrix_for_compensation();
  void save_state_queue_for_compensation();
  void load_state_queue_for_compensation();
  void initialize_for_candidates_compensation(int num_candidates);
  void set_offline_data_set_for_compensation(Eigen::MatrixXd XXT, Eigen::MatrixXd YYT);  
  void unset_offline_data_set_for_compensation();
  void set_projection_matrix_for_compensation(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix_for_compensation();
  Eigen::VectorXd prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm
  );
  Eigen::MatrixXd Prediction_for_compensation(
    Eigen::MatrixXd States, Eigen::MatrixXd Acc_input_history_concat, Eigen::MatrixXd Steer_input_history_concat, Eigen::MatrixXd H_lstm, Eigen::MatrixXd C_lstm
  );
  Eigen::VectorXd F_with_model_for_calc_controller_prediction_error(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model_without_compensation(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat,
    const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon);
  Eigen::VectorXd F_with_model_diff(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat,
    const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C);
  Eigen::VectorXd F_with_model_diff(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history,
    Eigen::VectorXd & steer_input_history, const Eigen::Vector2d & d_inputs,
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, const int horizon,
    Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C);
  Eigen::MatrixXd F_with_model_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, 
    Eigen::MatrixXd & Acc_input_history_concat, Eigen::MatrixXd & Steer_input_history_concat,
    const Eigen::MatrixXd & D_inputs,
    Eigen::MatrixXd & H_lstm, Eigen::MatrixXd & C_lstm, Eigen::MatrixXd & Previous_error, const int horizon);
  Eigen::MatrixXd F_with_model_for_candidates(
    const Eigen::MatrixXd & States, Eigen::MatrixXd & Acc_input_history,
    Eigen::MatrixXd & Steer_input_history, const Eigen::MatrixXd & D_inputs,
    Eigen::MatrixXd & H_lstm, Eigen::MatrixXd & C_lstm, Eigen::MatrixXd & Previous_error, const int horizon);
  void calc_forward_trajectory_with_diff(Eigen::VectorXd states, Eigen::VectorXd acc_input_history,
                                         Eigen::VectorXd steer_input_history, std::vector<Eigen::VectorXd> d_inputs_schedule,
                                         const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
                                         const Eigen::VectorXd & previous_error, std::vector<Eigen::VectorXd> & states_prediction,
                                         std::vector<Eigen::MatrixXd> & dF_d_states, std::vector<Eigen::MatrixXd> & dF_d_inputs,
                                         std::vector<Eigen::Vector2d> & inputs_schedule);
};
class AdaptorILQR
{
private:
  double wheel_base_ = 2.79;
  double acc_time_delay_ = 0.1;
  double steer_time_delay_ = 0.27;
  double acc_time_constant_ = 0.1;
  double steer_time_constant_ = 0.24;
  int acc_queue_size_ = 15;
  int steer_queue_size_ = 15;

  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;
  int state_size_ = 6;

  int acc_input_start_index_ = 6;
  int acc_input_end_index_ = 6 + acc_queue_size_ - 1;
  int steer_input_start_index_ = 6 + acc_queue_size_;
  int steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
  double control_dt_ = 0.033;
  int predict_step_ = 3;
  double predict_dt_ = control_dt_ * predict_step_;
  
  double error_decay_rate_ = 0.5;

  int acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
  int steer_delay_step_ =
    std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);

  int horizon_len_ = 10;
  int h_dim_;
  int state_dim_ = 6;
  TrainedDynamics trained_dynamics_;
  
  double x_cost_ = 0.0;
  double y_cost_ = 0.0;
  double vel_cost_ = 0.0;
  double yaw_cost_ = 1.0;

  double steer_cost_ = 10.0;
  double acc_cost_ = 1.0;
  double steer_rate_cost_ = 0.1;
  double acc_rate_cost_ = 1.0;
  
  double x_terminal_cost_ = 0.0;
  double y_terminal_cost_ = 0.0;
  double vel_terminal_cost_ = 0.0;
  double yaw_terminal_cost_ = 1.0;
  
  double steer_terminal_cost_ = 10.0;
  double acc_terminal_cost_ = 10.0;

  double steer_rate_rate_cost_ = 0.1;
  double acc_rate_rate_cost_ = 1.0;

  int intermediate_cost_index_ = 5;
  double x_intermediate_cost_ = 0.0;
  double y_intermediate_cost_ = 0.0;
  double vel_intermediate_cost_ = 0.0;
  double yaw_intermediate_cost_ = 1.0;
  double steer_intermediate_cost_ = 10.0;
  double acc_intermediate_cost_ = 10.0;
  std::vector<std::string> all_state_name_ = {"x", "y", "vel", "yaw", "acc", "steer"};
  std::vector<std::string> state_component_ilqr_ = {"vel", "acc", "steer"};
  std::vector<int> state_component_ilqr_index_;
  int num_state_component_ilqr_;

  std::vector<double> controller_acc_input_weight_target_table_, controller_longitudinal_coef_target_table_, controller_vel_error_domain_table_, controller_acc_error_domain_table_;
  std::vector<double> controller_steer_input_weight_target_table_, controller_lateral_coef_target_table_, controller_yaw_error_domain_table_, controller_steer_error_domain_table_;

  std::vector<double> vel_for_steer_rate_table_, steer_rate_cost_coef_by_vel_table_;
  


  std::vector<double> acc_rate_input_table_ = {0.001,0.01,0.1};
  std::vector<double> acc_rate_cost_coef_table_ = {100.0,10.0,1.0};
  std::vector<double> x_coef_by_acc_rate_table_, vel_coef_by_acc_rate_table_, acc_coef_by_acc_rate_table_;

  std::vector<double> steer_rate_input_table_ = {0.0001,0.001,0.01};
  std::vector<double> steer_rate_cost_coef_table_ = {100000.0,100.0,1.0};
  std::vector<double> y_coef_by_steer_rate_table_, yaw_coef_by_steer_rate_table_, steer_coef_by_steer_rate_table_;

  std::vector<double> x_error_domain_table_, y_error_domain_table_, vel_error_domain_table_, yaw_error_domain_table_, acc_error_domain_table_, steer_error_domain_table_;
  std::vector<double> x_error_target_table_, y_error_target_table_, vel_error_target_table_, yaw_error_target_table_, acc_error_target_table_, steer_error_target_table_;


public:
  AdaptorILQR();
  virtual ~AdaptorILQR();
  void set_params();
  void set_states_cost(
    double x_cost, double y_cost, double vel_cost, double yaw_cost, double acc_cost, double steer_cost
  );
  void set_inputs_cost(
     double acc_rate_cost, double steer_rate_cost
  );
  void set_rate_cost(
    double acc_rate_rate_cost, double steer_rate_rate_cost
  );
  void set_intermediate_cost(
    double x_intermediate_cost, double y_intermediate_cost, double vel_intermediate_cost, double yaw_intermediate_cost, double acc_intermediate_cost, double steer_intermediate_cost, int intermediate_cost_index
  );
  void set_terminal_cost(
    double x_terminal_cost, double y_terminal_cost, double vel_terminal_cost, double yaw_terminal_cost, double acc_terminal_cost, double steer_terminal_cost
  );
  void set_vehicle_params(
    double wheel_base, double acc_time_delay, double steer_time_delay, double acc_time_constant,
    double steer_time_constant, int acc_queue_size, int steer_queue_size, double control_dt,
    int predict_step);
  void set_NN_params(
    const Eigen::MatrixXd & weight_acc_encoder_layer_1, const Eigen::MatrixXd & weight_steer_encoder_layer_1,
    const Eigen::MatrixXd & weight_acc_encoder_layer_2, const Eigen::MatrixXd & weight_steer_encoder_layer_2,
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_ih, const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_encoder_layer_1, const Eigen::VectorXd & bias_steer_encoder_layer_1,
    const Eigen::VectorXd & bias_acc_encoder_layer_2, const Eigen::VectorXd & bias_steer_encoder_layer_2,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const std::vector<Eigen::VectorXd> & bias_lstm_encoder_ih, const std::vector<Eigen::VectorXd> & bias_lstm_encoder_hh,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted);
  void clear_NN_params();
  void set_sg_filter_params(int degree, int horizon_len,int window_size);
  Eigen::VectorXd nominal_prediction(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat);
  void update_lstm_states(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history_concat,
    const Eigen::VectorXd & steer_input_history_concat,
    std::vector<Eigen::VectorXd> & h_lstm, std::vector<Eigen::VectorXd> & c_lstm, const Eigen::Vector2d & previous_error);
  Eigen::VectorXd F_with_model_without_compensation(
    const Eigen::VectorXd & states, Eigen::VectorXd & acc_input_history_concat,
    Eigen::VectorXd & steer_input_history_concat, 
    Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm, Eigen::VectorXd & previous_error, int horizon);
  void initialize_compensation(int acc_queue_size, int steer_queue_size, int predict_step, int h_dim);
  void update_state_queue_for_compensation(Eigen::VectorXd states);
  void update_one_step_for_compensation(Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd & h_lstm, Eigen::VectorXd & c_lstm);
  void update_regression_matrix_for_compensation();
  void save_state_queue_for_compensation();
  void load_state_queue_for_compensation();
  void set_offline_data_set_for_compensation(Eigen::MatrixXd XXT, Eigen::MatrixXd YYT);
  void unset_offline_data_set_for_compensation();
  void set_projection_matrix_for_compensation(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix_for_compensation();
  Eigen::VectorXd prediction_for_compensation(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history_concat, Eigen::VectorXd steer_input_history_concat, Eigen::VectorXd h_lstm, Eigen::VectorXd c_lstm
  );
  void calc_forward_trajectory_with_cost(
    Eigen::VectorXd states, Eigen::VectorXd acc_input_history, Eigen::VectorXd steer_input_history,
    std::vector<Eigen::MatrixXd> D_inputs_schedule, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm,
    const Eigen::VectorXd & previous_error, std::vector<Eigen::MatrixXd> & states_prediction,
    const std::vector<Eigen::VectorXd> & states_ref, const std::vector<Eigen::VectorXd> & d_input_ref, 
    std::vector<Eigen::Vector2d> inputs_ref,
    double x_weight_coef, double y_weight_coef, double vel_weight_coef,
    double yaw_weight_coef, double acc_weight_coef, double steer_weight_coef,
    double acc_input_weight, double steer_input_weight,
    double acc_rate_weight_coef, double steer_rate_weight_coef,
    Eigen::VectorXd & Cost);
  void calc_inputs_ref_info(
    const Eigen::VectorXd & states, const Eigen::VectorXd & acc_input_history,
    const Eigen::VectorXd & steer_input_history, const Eigen::VectorXd & h_lstm,
    const Eigen::VectorXd & c_lstm, 
    const Eigen::VectorXd & previous_error, const std::vector<Eigen::VectorXd> & states_ref,
    const Eigen::VectorXd & acc_controller_input_schedule, const Eigen::VectorXd & steer_controller_input_schedule,
    const std::vector<Eigen::VectorXd> & states_prediction,
    std::vector<Eigen::Vector2d> & inputs_ref,
    const double & acc_input_change_rate, const double & steer_input_change_rate,
    double & x_weight_coef, double & y_weight_coef, double & vel_weight_coef,
    double & yaw_weight_coef, double & acc_weight_coef, double & steer_weight_coef,
    double & acc_input_weight, double & steer_input_weight,
    double & acc_rate_weight_coef, double & steer_rate_weight_coef,
    double & initial_prediction_x_weight_coef, double & initial_prediction_y_weight_coef, double & initial_prediction_vel_weight_coef,
    double & initial_prediction_yaw_weight_coef, double & initial_prediction_acc_weight_coef, double & initial_prediction_steer_weight_coef);
  Eigen::MatrixXd extract_dF_d_state(Eigen::MatrixXd dF_d_state_with_history);
  Eigen::MatrixXd extract_dF_d_input(Eigen::MatrixXd dF_d_input);
  Eigen::MatrixXd right_action_by_state_diff_with_history(Eigen::MatrixXd Mat, Eigen::MatrixXd dF_d_state_with_history);
  Eigen::MatrixXd left_action_by_state_diff_with_history(Eigen::MatrixXd dF_d_state_with_history, Eigen::MatrixXd Mat);
  Eigen::MatrixXd right_action_by_input_diff(Eigen::MatrixXd Mat, Eigen::MatrixXd dF_d_input);
  Eigen::MatrixXd left_action_by_input_diff(Eigen::MatrixXd dF_d_input, Eigen::MatrixXd Mat);
  void compute_ilqr_coefficients(
    const std::vector<Eigen::MatrixXd> & dF_d_states, const std::vector<Eigen::MatrixXd> & dF_d_inputs,
    const std::vector<Eigen::VectorXd> & states_prediction, const std::vector<Eigen::VectorXd> & d_inputs_schedule,
    const std::vector<Eigen::VectorXd> & states_ref,
    const std::vector<Eigen::VectorXd> & d_input_ref, const double prev_acc_rate, const double prev_steer_rate,
    std::vector<Eigen::Vector2d> inputs_ref, 
    double x_weight_coef, double y_weight_coef, double vel_weight_coef,
    double yaw_weight_coef, double acc_weight_coef, double steer_weight_coef,
    double acc_input_weight, double steer_input_weight,
    double acc_rate_weight_coef, double steer_rate_weight_coef,
    const std::vector<Eigen::Vector2d> & inputs_schedule,
    std::vector<Eigen::MatrixXd> & K, std::vector<Eigen::VectorXd> & k);
  std::vector<Eigen::MatrixXd> calc_line_search_candidates(std::vector<Eigen::MatrixXd> K, std::vector<Eigen::VectorXd> k, std::vector<Eigen::MatrixXd> dF_d_states, std::vector<Eigen::MatrixXd> dF_d_inputs,  std::vector<Eigen::VectorXd> d_inputs_schedule, Eigen::VectorXd ls_points);
  void compute_optimal_control(Eigen::VectorXd states, Eigen::VectorXd acc_input_history,
                              Eigen::VectorXd steer_input_history, std::vector<Eigen::VectorXd> & d_inputs_schedule,
                              const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
                              const std::vector<Eigen::VectorXd> & states_ref,
                              const std::vector<Eigen::VectorXd> & d_input_ref,
                              double acc_input_change_rate, double steer_input_change_rate,                              
                              Eigen::VectorXd & previous_error,
                              std::vector<Eigen::VectorXd> & states_prediction,
                              const Eigen::VectorXd & acc_controller_input_schedule, const Eigen::VectorXd & steer_controller_input_schedule,
                              double & acc_input, double & steer_input);

};
namespace Proxima{
class VehicleAdaptor
{
private:
  NominalDynamics nominal_dynamics_controller_;
  AdaptorILQR adaptor_ilqr_;
  PolynomialRegressionPredictor polynomial_reg_for_predict_acc_input_, polynomial_reg_for_predict_steer_input_;
  SgFilter sg_filter_for_d_inputs_schedule_;
  ButterworthFilter butterworth_filter_;
  InputsSchedulePrediction acc_input_schedule_prediction_, steer_input_schedule_prediction_;
  InputsRefSmoother acc_input_ref_smoother_, steer_input_ref_smoother_;
  // GetInitialHidden get_initial_hidden_;
  bool use_nonzero_initial_hidden_ = false;
  double prob_update_memory_bank_;
  int memory_bank_size_;
  int memory_bank_element_len_;
  bool initialized_memory_bank_ = false;


  int sg_window_size_for_d_inputs_schedule_, sg_deg_for_d_inputs_schedule_;
  bool use_sg_for_d_inputs_schedule_;
  double wheel_base_ = 2.79;
  double acc_time_delay_ = 0.1;
  double steer_time_delay_ = 0.27;
  double acc_time_constant_ = 0.1;
  double steer_time_constant_ = 0.24;
  int acc_queue_size_ = 15;
  int steer_queue_size_ = 15;
  double steer_dead_band_ = 0.0012;

  int x_index_ = 0;
  int y_index_ = 1;
  int vel_index_ = 2;
  int yaw_index_ = 3;
  int acc_index_ = 4;
  int steer_index_ = 5;

  int acc_input_start_index_ = 6;
  int acc_input_end_index_ = 6 + acc_queue_size_ - 1;
  int steer_input_start_index_ = 6 + acc_queue_size_;
  int steer_input_end_index_ = 6 + acc_queue_size_ + steer_queue_size_ - 1;
  double control_dt_ = 0.033;
  int predict_step_ = 3;
  double predict_dt_ = control_dt_ * predict_step_;

  double error_decay_rate_ = 0.5;

  int acc_delay_step_ = std::min(int(std::round(acc_time_delay_ / control_dt_)), acc_queue_size_);
  int steer_delay_step_ =
    std::min(int(std::round(steer_time_delay_ / control_dt_)), steer_queue_size_);
  int acc_delay_step_controller_;
  int steer_delay_step_controller_;
  int horizon_len_ = 10;
  int sg_deg_for_NN_diff_ = 2;
  int sg_window_size_for_NN_diff_ = 5;
  int h_dim_;
  int h_dim_full_;
  int state_dim_ = 6;
  double x_cost_ = 0.0;
  double y_cost_ = 0.0;
  double vel_cost_ = 0.0;
  double yaw_cost_ = 1.0;
  double steer_cost_ = 10.0;
  double acc_cost_ = 1.0;
  double steer_rate_cost_ = 0.1;
  double acc_rate_cost_ = 1.0;
  double x_terminal_cost_ = 0.0;
  double y_terminal_cost_ = 0.0;
  double vel_terminal_cost_ = 0.0;
  double yaw_terminal_cost_ = 1.0;
  double steer_terminal_cost_ = 10.0;
  double acc_terminal_cost_ = 10.0;
  double x_intermediate_cost_ = 0.0;
  double y_intermediate_cost_ = 0.0;
  double vel_intermediate_cost_ = 0.0;
  double yaw_intermediate_cost_ = 1.0;
  double acc_intermediate_cost_ = 10.0;
  double steer_intermediate_cost_ = 10.0;
  int intermediate_cost_index_ = 5;

  int NN_prediction_target_dim_;

  bool initialized_ = false;
  Eigen::VectorXd acc_input_history_;
  Eigen::VectorXd steer_input_history_;
  Eigen::VectorXd acc_controller_input_history_, steer_controller_input_history_;
  std::vector<Eigen::VectorXd> d_inputs_schedule_;
  std::vector<Eigen::VectorXd> state_history_lstm_, acc_input_history_lstm_, steer_input_history_lstm_;

  int controller_acc_input_history_len_, controller_steer_input_history_len_;
  int deg_controller_acc_input_history_, deg_controller_steer_input_history_;
  std::vector<double> lam_controller_acc_input_history_, lam_controller_steer_input_history_;
  double oldest_sample_weight_controller_acc_input_history_, oldest_sample_weight_controller_steer_input_history_;
  int acc_polynomial_prediction_len_, steer_polynomial_prediction_len_;

  bool use_acc_linear_extrapolation_ = false;
  bool use_steer_linear_extrapolation_ = false;
  int acc_linear_extrapolation_len_ = 15;
  int steer_linear_extrapolation_len_ = 15;
  int past_len_for_acc_linear_extrapolation_ = 3;
  int past_len_for_steer_linear_extrapolation_ = 3;


  int update_lstm_len_ = 50;
  int compensation_lstm_len_ = 10;

  Eigen::VectorXd previous_error_;


  std::string states_ref_mode_ = "predict_by_polynomial_regression";

  Eigen::VectorXd acc_controller_d_inputs_schedule_, steer_controller_d_inputs_schedule_;
  Eigen::VectorXd x_controller_prediction_, y_controller_prediction_, vel_controller_prediction_, yaw_controller_prediction_, acc_controller_prediction_, steer_controller_prediction_;

  double reflect_controller_d_input_ratio_ = 0.5;
  std::vector<double> time_stamp_obs_;
  std::vector<double> acc_input_history_obs_, steer_input_history_obs_, acc_controller_input_history_obs_, steer_controller_input_history_obs_;
  std::vector<Eigen::VectorXd> state_history_lstm_obs_, acc_input_history_lstm_obs_, steer_input_history_lstm_obs_;
  int max_queue_size_;

  bool use_controller_inputs_as_target_;
  std::vector<Eigen::VectorXd> states_prediction_;
  std::vector<double> mix_ratio_vel_target_table_, mix_ratio_vel_domain_table_;
  std::vector<double> mix_ratio_time_target_table_, mix_ratio_time_domain_table_;

  std::string input_filter_mode_ = "none";
  bool steer_controller_prediction_aided_ = false;
  double start_time_;
  bool use_acc_input_schedule_prediction_ = false;
  bool use_steer_input_schedule_prediction_ = false;
  int acc_input_schedule_prediction_len_, steer_input_schedule_prediction_len_;

  int num_layers_encoder_;
  bool acc_input_schedule_prediction_initialized_ = false;
  bool steer_input_schedule_prediction_initialized_ = false;

  Eigen::VectorXd offline_features_;
  bool use_offline_features_ = false;
  double past_acc_input_change_, past_steer_input_change_;
  double past_acc_input_change_decay_rate_, past_steer_input_change_decay_rate_;
  int acc_input_change_window_size_, steer_input_change_window_size_;
  double future_acc_input_change_, future_steer_input_change_;
  double future_acc_input_change_decay_rate_, future_steer_input_change_decay_rate_;
  double past_acc_input_change_weight_, past_steer_input_change_weight_;
public:
  bool use_controller_steer_input_schedule_ = false;
  bool use_vehicle_adaptor_;
  bool use_offline_features_autoware_;
  VehicleAdaptor();
  virtual ~VehicleAdaptor();
  void set_params();
  void set_NN_params(
    const Eigen::MatrixXd & weight_acc_encoder_layer_1, const Eigen::MatrixXd & weight_steer_encoder_layer_1,
    const Eigen::MatrixXd & weight_acc_encoder_layer_2, const Eigen::MatrixXd & weight_steer_encoder_layer_2,
    const Eigen::MatrixXd & weight_acc_layer_1, const Eigen::MatrixXd & weight_steer_layer_1,
    const Eigen::MatrixXd & weight_acc_layer_2, const Eigen::MatrixXd & weight_steer_layer_2,
    const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_ih, const std::vector<Eigen::MatrixXd> & weight_lstm_encoder_hh,
    const Eigen::MatrixXd & weight_lstm_ih, const Eigen::MatrixXd & weight_lstm_hh,
    const Eigen::MatrixXd & weight_complimentary_layer,
    const Eigen::MatrixXd & weight_linear_relu, const Eigen::MatrixXd & weight_final_layer,
    const Eigen::VectorXd & bias_acc_encoder_layer_1, const Eigen::VectorXd & bias_steer_encoder_layer_1,
    const Eigen::VectorXd & bias_acc_encoder_layer_2, const Eigen::VectorXd & bias_steer_encoder_layer_2,
    const Eigen::VectorXd & bias_acc_layer_1, const Eigen::VectorXd & bias_steer_layer_1,
    const Eigen::VectorXd & bias_acc_layer_2, const Eigen::VectorXd & bias_steer_layer_2,
    const std::vector<Eigen::VectorXd> & bias_lstm_encoder_ih, const std::vector<Eigen::VectorXd> & bias_lstm_encoder_hh,
    const Eigen::VectorXd & bias_lstm_ih, const Eigen::VectorXd & bias_lstm_hh,
    const Eigen::VectorXd & bias_complimentary_layer,
    const Eigen::VectorXd & bias_linear_relu, const Eigen::VectorXd & bias_final_layer,
    const double vel_scaling, const double vel_bias, const std::vector<std::string> state_component_predicted);
  void set_offline_features(const Eigen::VectorXd & offline_features);
  void set_NN_params_from_csv(std::string csv_dir);
  void set_offline_features_from_csv(std::string csv_dir);
  void clear_NN_params();
  void set_offline_data_set_for_compensation(Eigen::MatrixXd XXT, Eigen::MatrixXd YYT);
  void set_offline_data_set_for_compensation_from_csv(std::string csv_dir);
  void unset_offline_data_set_for_compensation();
  void set_projection_matrix_for_compensation(Eigen::MatrixXd projection_matrix);
  void unset_projection_matrix_for_compensation();
  void set_projection_matrix_for_compensation_from_csv(std::string csv_dir);
  void set_controller_d_inputs_schedule(const Eigen::VectorXd & acc_controller_d_inputs_schedule, const Eigen::VectorXd & steer_controller_d_inputs_schedule);
  void set_controller_d_steer_schedule(const Eigen::VectorXd & steer_controller_d_inputs_schedule);
  void set_controller_steer_input_schedule(double timestamp, const std::vector<double> & steer_controller_input_schedule);
  void set_controller_prediction(const Eigen::VectorXd & x_controller_prediction, const Eigen::VectorXd & y_controller_prediction, const Eigen::VectorXd & vel_controller_prediction, const Eigen::VectorXd & yaw_controller_prediction, const Eigen::VectorXd & acc_controller_prediction, const Eigen::VectorXd & steer_controller_prediction);
  void set_controller_steer_prediction(const Eigen::VectorXd & steer_controller_prediction);
  Eigen::VectorXd get_adjusted_inputs(
    double time_stamp, const Eigen::VectorXd & states, const double acc_controller_input, const double steer_controller_input);
  Eigen::MatrixXd get_states_prediction();
  Eigen::MatrixXd get_d_inputs_schedule();
  void send_initialized_flag();
};
}
#endif // VEHICLE_ADAPTOR_COMPENSATOR_H