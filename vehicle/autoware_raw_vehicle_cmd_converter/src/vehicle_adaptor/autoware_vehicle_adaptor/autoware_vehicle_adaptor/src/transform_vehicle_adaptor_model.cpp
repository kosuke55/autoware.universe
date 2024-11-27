#include "transform_vehicle_adaptor_model.h"

///////////////// TransformModelToEigen ///////////////////////

  TransformModelToEigen::TransformModelToEigen() {}
  TransformModelToEigen::~TransformModelToEigen() {}
  void TransformModelToEigen::set_params(
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
    const double vel_scaling, const double vel_bias)
  {
    weight_acc_encoder_layer_1_ = weight_acc_encoder_layer_1;
    weight_steer_encoder_layer_1_ = weight_steer_encoder_layer_1;
    weight_acc_encoder_layer_2_ = weight_acc_encoder_layer_2;
    weight_steer_encoder_layer_2_ = weight_steer_encoder_layer_2;

    weight_acc_layer_1_=weight_acc_layer_1;
    weight_steer_layer_1_=weight_steer_layer_1;
    weight_acc_layer_2_=weight_acc_layer_2;
    weight_steer_layer_2_=weight_steer_layer_2;
    weight_lstm_ih_=weight_lstm_ih;
    weight_lstm_hh_=weight_lstm_hh;
    weight_lstm_encoder_ih_=weight_lstm_encoder_ih;
    weight_lstm_encoder_hh_=weight_lstm_encoder_hh;
    weight_complimentary_layer_.push_back(weight_complimentary_layer);
    weight_linear_relu_.push_back(weight_linear_relu);
    weight_final_layer_.push_back(weight_final_layer);

    bias_acc_encoder_layer_1_=bias_acc_encoder_layer_1;
    bias_steer_encoder_layer_1_=bias_steer_encoder_layer_1;
    bias_acc_encoder_layer_2_=bias_acc_encoder_layer_2;
    bias_steer_encoder_layer_2_=bias_steer_encoder_layer_2;

    bias_acc_layer_1_=bias_acc_layer_1;
    bias_steer_layer_1_=bias_steer_layer_1;
    bias_acc_layer_2_=bias_acc_layer_2;
    bias_steer_layer_2_=bias_steer_layer_2;
    bias_lstm_encoder_ih_=bias_lstm_encoder_ih;
    bias_lstm_encoder_hh_=bias_lstm_encoder_hh;
    bias_lstm_ih_=bias_lstm_ih;
    bias_lstm_hh_=bias_lstm_hh;
    bias_complimentary_layer_.push_back(bias_complimentary_layer);
    bias_linear_relu_.push_back(bias_linear_relu);
    bias_final_layer_.push_back(bias_final_layer);
    vel_scaling_=vel_scaling;
    vel_bias_=vel_bias;
    h_dim_full_ = weight_lstm_encoder_hh[0].cols();
    h_dim_acc_ = weight_acc_layer_1.cols() - weight_acc_encoder_layer_1.cols();
    h_dim_steer_ = weight_steer_layer_1.cols() - weight_steer_encoder_layer_1.cols();
    h_dim_ = weight_lstm_hh.cols();
    num_layers_encoder_ = weight_lstm_encoder_ih.size();
    model_num_ += 1;
  }
  void TransformModelToEigen::clear_params()
  {
    weight_complimentary_layer_.clear();
    weight_linear_relu_.clear();
    weight_final_layer_.clear();
    bias_complimentary_layer_.clear();
    bias_linear_relu_.clear();
    bias_final_layer_.clear();
    model_num_ = 0;
  }
  void TransformModelToEigen::update_lstm(
    const Eigen::VectorXd & x, const std::vector<Eigen::VectorXd> & h_lstm, const std::vector<Eigen::VectorXd> & c_lstm,
    std::vector<Eigen::VectorXd> & h_lstm_next, std::vector<Eigen::VectorXd> & c_lstm_next, const Eigen::Vector2d & acc_steer_error)
  {
    Eigen::VectorXd acc_sub(acc_queue_size_ + predict_step_ + 2);
    Eigen::VectorXd steer_sub(steer_queue_size_ + predict_step_ + 2);
    acc_sub << vel_scaling_ * (x[vel_index_] - vel_bias_), x[acc_index_], x.segment(acc_input_start_index_, acc_queue_size_ + predict_step_);
    steer_sub << vel_scaling_ * (x[vel_index_] - vel_bias_), x[steer_index_],
      x.segment(steer_input_start_index_, steer_queue_size_ + predict_step_);

    const Eigen::VectorXd acc_layer_1 = relu(weight_acc_encoder_layer_1_ * acc_sub + bias_acc_encoder_layer_1_);
    const Eigen::VectorXd steer_layer_1 = relu(weight_steer_encoder_layer_1_ * steer_sub + bias_steer_encoder_layer_1_);

    const Eigen::VectorXd acc_layer_2 = relu(weight_acc_encoder_layer_2_ * acc_layer_1 + bias_acc_encoder_layer_2_);
    const Eigen::VectorXd steer_layer_2 = relu(weight_steer_encoder_layer_2_ * steer_layer_1 + bias_steer_encoder_layer_2_);

    Eigen::VectorXd h1 = Eigen::VectorXd(1 + acc_layer_2.size() + steer_layer_2.size() + 2);
    h1 << vel_scaling_ * (x[vel_index_] - vel_bias_), acc_layer_2, steer_layer_2, acc_steer_error;

    h_lstm_next = std::vector<Eigen::VectorXd>(num_layers_encoder_);
    c_lstm_next = std::vector<Eigen::VectorXd>(num_layers_encoder_);
    for (int i = 0; i< num_layers_encoder_;i++){
      Eigen::VectorXd lstm_input;
      if (i == 0){
        lstm_input = h1;
      }
      else{
        lstm_input = h_lstm_next[i - 1];
      }
      Eigen::VectorXd u_i_f_g_o = weight_lstm_encoder_ih_[i] * lstm_input + bias_lstm_encoder_ih_[i] + weight_lstm_encoder_hh_[i] * h_lstm[i] + bias_lstm_encoder_hh_[i];
      Eigen::VectorXd i_lstm = sigmoid(u_i_f_g_o.segment(0, h_dim_full_));
      Eigen::VectorXd f_lstm = sigmoid(u_i_f_g_o.segment(h_dim_full_, h_dim_full_));
      Eigen::VectorXd g_lstm = tanh(u_i_f_g_o.segment(2 * h_dim_full_, h_dim_full_));
      Eigen::VectorXd o_lstm = sigmoid(u_i_f_g_o.segment(3 * h_dim_full_, h_dim_full_));
      c_lstm_next[i] = f_lstm.array() * c_lstm[i].array() + i_lstm.array() * g_lstm.array();
      h_lstm_next[i] = o_lstm.array() * tanh(c_lstm_next[i]).array();
    }
  }
  void TransformModelToEigen::error_prediction(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next,
    Eigen::VectorXd & u_acc_layer_1, Eigen::VectorXd & u_steer_layer_1,
    Eigen::VectorXd & u_acc_layer_2, Eigen::VectorXd & u_steer_layer_2,
    Eigen::VectorXd & u_i_lstm, Eigen::VectorXd & u_f_lstm, Eigen::VectorXd & u_g_lstm,
    Eigen::VectorXd & u_o_lstm, Eigen::VectorXd & i_lstm, Eigen::VectorXd & f_lstm,
    Eigen::VectorXd & g_lstm, Eigen::VectorXd & o_lstm, std::vector<Eigen::VectorXd> & u_complimentary_layer,
    std::vector<Eigen::VectorXd> & u_linear_relu, Eigen::VectorXd & y)
  {
    Eigen::VectorXd acc_sub(acc_queue_size_ + predict_step_ + 2 + h_dim_acc_);
    Eigen::VectorXd steer_sub(steer_queue_size_ + predict_step_ + 2 + h_dim_steer_);
    Eigen::VectorXd h_lstm_head = h_lstm.head(h_dim_);
    Eigen::VectorXd c_lstm_head = c_lstm.head(h_dim_);
    Eigen::VectorXd h_lstm_acc = h_lstm.segment(h_dim_, h_dim_acc_);
    Eigen::VectorXd h_lstm_steer = h_lstm.segment(h_dim_ + h_dim_acc_, h_dim_steer_);
    acc_sub << vel_scaling_ * (x[vel_index_] - vel_bias_), x[acc_index_],
      x.segment(acc_input_start_index_, acc_queue_size_ + predict_step_), h_lstm_acc;
    steer_sub << vel_scaling_ * (x[vel_index_] - vel_bias_), x[steer_index_],
      x.segment(steer_input_start_index_, steer_queue_size_ + predict_step_), h_lstm_steer;

    u_acc_layer_1 = weight_acc_layer_1_ * acc_sub + bias_acc_layer_1_;
    u_steer_layer_1 = weight_steer_layer_1_ * steer_sub + bias_steer_layer_1_;

    const Eigen::VectorXd acc_layer_1 = relu(u_acc_layer_1);
    const Eigen::VectorXd steer_layer_1 = relu(u_steer_layer_1);
    u_acc_layer_2 = weight_acc_layer_2_ * acc_layer_1 + bias_acc_layer_2_;
    u_steer_layer_2 = weight_steer_layer_2_ * steer_layer_1 + bias_steer_layer_2_;

    const Eigen::VectorXd acc_layer_2 = relu(u_acc_layer_2);
    const Eigen::VectorXd steer_layer_2 = relu(u_steer_layer_2);

    Eigen::VectorXd h1 = Eigen::VectorXd(1 + acc_layer_2.size() + steer_layer_2.size());
    h1 << vel_scaling_ * (x[vel_index_] - vel_bias_), acc_layer_2, steer_layer_2;

    Eigen::VectorXd u_i_f_g_o = weight_lstm_ih_ * h1 + bias_lstm_ih_ + weight_lstm_hh_ * h_lstm_head + bias_lstm_hh_;


    u_i_lstm = u_i_f_g_o.segment(0, h_dim_);

    u_f_lstm = u_i_f_g_o.segment(h_dim_, h_dim_);
    u_g_lstm = u_i_f_g_o.segment(2 * h_dim_, h_dim_);
    u_o_lstm = u_i_f_g_o.segment(3 * h_dim_, h_dim_);

    i_lstm = sigmoid(u_i_lstm);
    f_lstm = sigmoid(u_f_lstm);
    g_lstm = tanh(u_g_lstm);
    o_lstm = sigmoid(u_o_lstm);

    c_lstm_next = c_lstm;
    h_lstm_next = h_lstm;

    c_lstm_next.head(h_dim_) = f_lstm.array() * c_lstm_head.array() + i_lstm.array() * g_lstm.array();
    h_lstm_next.head(h_dim_) = o_lstm.array() * tanh(c_lstm_next.head(h_dim_)).array();


    ///
    u_complimentary_layer.assign(model_num_, Eigen::VectorXd());
    u_linear_relu.assign(model_num_, Eigen::VectorXd());
    y = Eigen::VectorXd::Zero(bias_final_layer_[0].size());
    for (int i = 0; i < model_num_; i++){
      u_complimentary_layer[i] = weight_complimentary_layer_[i] * h1 + bias_complimentary_layer_[i];
      Eigen::VectorXd h_complimentary_layer = relu(u_complimentary_layer[i]);

      Eigen::VectorXd h_lstm_output = Eigen::VectorXd(h_dim_ + h_complimentary_layer.size());
      h_lstm_output << h_lstm_next.head(h_dim_), h_complimentary_layer;
      u_linear_relu[i] = weight_linear_relu_[i] * h_lstm_output + bias_linear_relu_[i];
      Eigen::VectorXd linear_relu = relu(u_linear_relu[i]);
      y += (weight_final_layer_[i] * linear_relu + bias_final_layer_[i]) / model_num_;
    }
  }
  void TransformModelToEigen::error_prediction(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next,
    Eigen::VectorXd & y)
  {
    Eigen::VectorXd u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, 
      u_i_lstm, u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm;
    std::vector<Eigen::VectorXd> u_complimentary_layer, u_linear_relu;
    error_prediction(
      x, h_lstm, c_lstm, h_lstm_next, c_lstm_next,
      u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, u_i_lstm,
      u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm,
      u_complimentary_layer, u_linear_relu, y);
  }
  void TransformModelToEigen::error_prediction_with_diff(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_lstm, const Eigen::VectorXd & c_lstm,
    Eigen::VectorXd & h_lstm_next, Eigen::VectorXd & c_lstm_next, Eigen::VectorXd & y,
    Eigen::MatrixXd & dy_dx, Eigen::MatrixXd & dy_dhc, Eigen::MatrixXd & dhc_dhc,
    Eigen::MatrixXd & dhc_dx)
  {
    Eigen::VectorXd u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, 
      u_i_lstm, u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm;
    std::vector<Eigen::VectorXd> u_complimentary_layer, u_linear_relu;
    error_prediction(
      x, h_lstm, c_lstm, h_lstm_next, c_lstm_next,
      u_acc_layer_1, u_steer_layer_1, u_acc_layer_2, u_steer_layer_2, u_i_lstm,
      u_f_lstm, u_g_lstm, u_o_lstm, i_lstm, f_lstm, g_lstm, o_lstm,
      u_complimentary_layer, u_linear_relu, y);
    Eigen::VectorXd c_lstm_next_head = c_lstm_next.head(h_dim_);
    dy_dx = Eigen::MatrixXd::Zero(y.size(), x.size());
    dy_dhc = Eigen::MatrixXd::Zero(y.size(), 2 * h_dim_);
    dhc_dhc = Eigen::MatrixXd::Zero(2 * h_dim_, 2 * h_dim_);
    dhc_dx = Eigen::MatrixXd::Zero(2 * h_dim_, x.size());
    int h1_size = 1 + u_acc_layer_2.size() + u_steer_layer_2.size();
    Eigen::MatrixXd dy_d_lstm = Eigen::MatrixXd::Zero(y.size(), h_dim_);
    Eigen::MatrixXd dy_dh1_via_complimentary_layer = Eigen::MatrixXd::Zero(y.size(), h1_size);
    for (int i = 0; i< model_num_; i++){
      Eigen::MatrixXd dy_d_linear_relu = weight_final_layer_[i];
      Eigen::MatrixXd dy_d_lstm_output =
        d_relu_product(dy_d_linear_relu, u_linear_relu[i]) * weight_linear_relu_[i];

      dy_d_lstm +=
        dy_d_lstm_output.block(0, 0, dy_d_lstm_output.rows(), h_dim_)/model_num_;
      Eigen::MatrixXd dy_d_complimentary_layer = dy_d_lstm_output.block(
        0, h_dim_, dy_d_lstm_output.rows(), bias_complimentary_layer_[i].size());

      dy_dh1_via_complimentary_layer  += d_relu_product(dy_d_complimentary_layer, u_complimentary_layer[i]) *
                                             weight_complimentary_layer_[i]/model_num_;

    }

    Eigen::MatrixXd dy_d_o_lstm = dy_d_lstm * tanh(c_lstm_next_head).asDiagonal();


    Eigen::MatrixXd dy_dc_lstm_next = d_tanh_product(dy_d_lstm * o_lstm.asDiagonal(), c_lstm_next_head);

      
    Eigen::MatrixXd dy_dh1 =
      d_sigmoid_product(dy_d_o_lstm, u_o_lstm) *
      weight_lstm_ih_.block(3 * h_dim_, 0, h_dim_, h1_size);

    dy_dh1 += d_sigmoid_product(dy_dc_lstm_next * c_lstm_next_head.asDiagonal(), u_f_lstm) *
              weight_lstm_ih_.block(1 * h_dim_, 0, h_dim_, h1_size);

    dy_dh1 += d_tanh_product(dy_dc_lstm_next * i_lstm.asDiagonal(), u_g_lstm) *
              weight_lstm_ih_.block(2 * h_dim_, 0, h_dim_, h1_size);
    dy_dh1 += d_sigmoid_product(dy_dc_lstm_next * g_lstm.asDiagonal(), u_i_lstm) *
              weight_lstm_ih_.block(0, 0, h_dim_, h1_size);
    dy_dh1 += dy_dh1_via_complimentary_layer;

    Eigen::MatrixXd dy_d_acc_layer_1 =
      d_relu_product(dy_dh1.block(0, 1, y.size(), u_acc_layer_2.size()), u_acc_layer_2) *
      weight_acc_layer_2_;
    Eigen::MatrixXd dy_d_steer_layer_1 =
      d_relu_product(
        dy_dh1.block(0, 1 + u_acc_layer_2.size(), y.size(), u_steer_layer_2.size()),
        u_steer_layer_2) *
      weight_steer_layer_2_;

    Eigen::MatrixXd dy_d_acc_sub =
      d_relu_product(dy_d_acc_layer_1, u_acc_layer_1) * weight_acc_layer_1_;
    Eigen::MatrixXd dy_d_steer_sub =
      d_relu_product(dy_d_steer_layer_1, u_steer_layer_1) * weight_steer_layer_1_;


    dy_dx.col(vel_index_) = vel_scaling_ * (dy_dh1.col(0) + dy_d_acc_sub.col(0) + dy_d_steer_sub.col(0));
    dy_dx.col(acc_index_) = dy_d_acc_sub.col(1);
    dy_dx.col(steer_index_) = dy_d_steer_sub.col(1);
    dy_dx.block(0, acc_input_start_index_, y.size(), acc_queue_size_ + predict_step_) =
      dy_d_acc_sub.block(0, 2, y.size(), acc_queue_size_ + predict_step_);
    dy_dx.block(0, steer_input_start_index_, y.size(), steer_queue_size_ + predict_step_) =
      dy_d_steer_sub.block(0, 2, y.size(), steer_queue_size_ + predict_step_);

    // calc dy_dhc,  dhc_dhc, dhc_dx
    const Eigen::VectorXd dc_du_f = d_sigmoid_product_vec(c_lstm_next_head, u_f_lstm);
    const Eigen::VectorXd dc_du_g = d_tanh_product_vec(i_lstm, u_g_lstm);
    const Eigen::VectorXd dc_du_i = d_sigmoid_product_vec(g_lstm, u_i_lstm);
    const Eigen::VectorXd dh_dc_next = d_tanh_product_vec(o_lstm, c_lstm_next_head);
    const Eigen::VectorXd dh_du_o = d_sigmoid_product_vec(tanh(c_lstm_next_head), u_o_lstm);

    const Eigen::MatrixXd dc_dc = f_lstm.asDiagonal();
    const Eigen::MatrixXd dy_dc = dy_dc_lstm_next * dc_dc;

    Eigen::MatrixXd dc_dh =
      dc_du_f.asDiagonal() *
      weight_lstm_hh_.block(h_dim_, 0, h_dim_, h_dim_);
    dc_dh += dc_du_g.asDiagonal() *
            weight_lstm_hh_.block(2 * h_dim_, 0, h_dim_, h_dim_);
    dc_dh += dc_du_i.asDiagonal() * weight_lstm_hh_.block(0, 0, h_dim_, h_dim_);
    const Eigen::VectorXd dh_dc = dh_dc_next.array() * f_lstm.array();

    Eigen::MatrixXd dh_dh =
      dh_du_o.asDiagonal() *
      weight_lstm_hh_.block(3 * h_dim_, 0, h_dim_, h_dim_);

    dh_dh += dh_dc_next.asDiagonal() * dc_dh;

    const Eigen::MatrixXd dy_dh = dy_d_lstm * dh_dh;
    Eigen::MatrixXd dc_dh1 =
      dc_du_f.asDiagonal() * weight_lstm_ih_.block(h_dim_, 0, h_dim_, h1_size);
    dc_dh1 += dc_du_g.asDiagonal() *
              weight_lstm_ih_.block(2 * h_dim_, 0, h_dim_, h1_size);
    dc_dh1 += dc_du_i.asDiagonal() * weight_lstm_ih_.block(0, 0, h_dim_, h1_size);

    Eigen::MatrixXd dh_dh1 =
      dh_du_o.asDiagonal() *
      weight_lstm_ih_.block(3 * h_dim_, 0, h_dim_, h1_size);
    dh_dh1 += dh_dc_next.asDiagonal() * dc_dh1;

    Eigen::MatrixXd dh_d_acc_layer_1 =
      d_relu_product(dh_dh1.block(0, 1, h_dim_, u_acc_layer_2.size()), u_acc_layer_2) *
      weight_acc_layer_2_;
    Eigen::MatrixXd dh_d_steer_layer_1 =
      d_relu_product(
        dh_dh1.block(0, 1 + u_acc_layer_2.size(), h_dim_, u_steer_layer_2.size()),
        u_steer_layer_2) *
      weight_steer_layer_2_;
    Eigen::MatrixXd dh_d_acc_sub =
      d_relu_product(dh_d_acc_layer_1, u_acc_layer_1) * weight_acc_layer_1_;
    Eigen::MatrixXd dh_d_steer_sub =
      d_relu_product(dh_d_steer_layer_1, u_steer_layer_1) * weight_steer_layer_1_;


    Eigen::MatrixXd dh_dx = Eigen::MatrixXd::Zero(h_dim_, x.size());
    dh_dx.col(vel_index_) += vel_scaling_ * (dh_dh1.col(0) + dh_d_acc_sub.col(0) + dh_d_steer_sub.col(0));
    dh_dx.col(acc_index_) += dh_d_acc_sub.col(1);
    dh_dx.col(steer_index_) += dh_d_steer_sub.col(1);
    dh_dx.block(0, acc_input_start_index_, h_dim_, acc_queue_size_ + predict_step_) +=
      dh_d_acc_sub.block(0, 2, h_dim_, acc_queue_size_ + predict_step_);
    dh_dx.block(0, steer_input_start_index_, h_dim_, steer_queue_size_ + predict_step_) +=
      dh_d_steer_sub.block(0, 2, h_dim_, steer_queue_size_ + predict_step_);

    Eigen::MatrixXd dc_d_acc_layer_1 =
      d_relu_product(dc_dh1.block(0, 1, h_dim_, u_acc_layer_2.size()), u_acc_layer_2) *
      weight_acc_layer_2_;
    Eigen::MatrixXd dc_d_steer_layer_1 =
      d_relu_product(
        dc_dh1.block(0, 1 + u_acc_layer_2.size(), h_dim_, u_steer_layer_2.size()),
        u_steer_layer_2) *
      weight_steer_layer_2_;
    Eigen::MatrixXd dc_d_acc_sub =
      d_relu_product(dc_d_acc_layer_1, u_acc_layer_1) * weight_acc_layer_1_;
    Eigen::MatrixXd dc_d_steer_sub =
      d_relu_product(dc_d_steer_layer_1, u_steer_layer_1) * weight_steer_layer_1_;

    Eigen::MatrixXd dc_dx = Eigen::MatrixXd::Zero(h_dim_, x.size());
    dc_dx.col(vel_index_) += vel_scaling_ * (dc_dh1.col(0) + dc_d_acc_sub.col(0) + dc_d_steer_sub.col(0));
    dc_dx.col(acc_index_) += dc_d_acc_sub.col(1);
    dc_dx.col(steer_index_) += dc_d_steer_sub.col(1);
    dc_dx.block(0, acc_input_start_index_, h_dim_, acc_queue_size_ + predict_step_) +=
      dc_d_acc_sub.block(0, 2, h_dim_, acc_queue_size_ + predict_step_);
    dc_dx.block(0, steer_input_start_index_, h_dim_, steer_queue_size_ + predict_step_) +=
      dc_d_steer_sub.block(0, 2, h_dim_, steer_queue_size_ + predict_step_);

    dy_dhc.block(0, 0, y.size(), h_dim_) = dy_dh;
    dy_dhc.block(0, h_dim_, y.size(), h_dim_) = dy_dc;
    dhc_dhc.block(0, 0, h_dim_, h_dim_) = dh_dh;
    dhc_dhc.block(h_dim_, 0, h_dim_, h_dim_) = dc_dh;
    dhc_dhc.block(0, h_dim_, h_dim_, h_dim_) = dh_dc.asDiagonal();
    dhc_dhc.block(h_dim_, h_dim_, h_dim_, h_dim_) = dc_dc;
    dhc_dx.block(0, 0, h_dim_, x.size()) = dh_dx;
    dhc_dx.block(h_dim_, 0, h_dim_, x.size()) = dc_dx;
  }

 GetInitialHidden::GetInitialHidden() {}
  GetInitialHidden::~GetInitialHidden() {}
  void GetInitialHidden::set_initial_hidden_params(
    const Eigen::MatrixXd & weight_initial_hidden_acc_layer_1, const Eigen::MatrixXd & weight_initial_hidden_steer_layer_1,
    const Eigen::MatrixXd & weight_initial_hidden_acc_layer_2, const Eigen::MatrixXd & weight_initial_hidden_steer_layer_2,
    const Eigen::MatrixXd & weight_initial_hidden_gru_ih, const Eigen::MatrixXd & weight_initial_hidden_gru_hh,
    const Eigen::MatrixXd & weight_initial_hidden_final_layer,
    const Eigen::MatrixXd & weight_initial_hidden_fusion_layer, const Eigen::MatrixXd & weight_initial_hidden_only_online,
    const Eigen::VectorXd & bias_initial_hidden_acc_layer_1, const Eigen::VectorXd & bias_initial_hidden_steer_layer_1,
    const Eigen::VectorXd & bias_initial_hidden_acc_layer_2, const Eigen::VectorXd & bias_initial_hidden_steer_layer_2,
    const Eigen::VectorXd & bias_initial_hidden_gru_ih, const Eigen::VectorXd & bias_initial_hidden_gru_hh,
    const Eigen::VectorXd & bias_initial_hidden_final_layer,
    const Eigen::VectorXd & bias_initial_hidden_fusion_layer, const Eigen::VectorXd & bias_initial_hidden_only_online)
  {
    weight_initial_hidden_acc_layer_1_ = weight_initial_hidden_acc_layer_1;
    weight_initial_hidden_steer_layer_1_ = weight_initial_hidden_steer_layer_1;
    weight_initial_hidden_acc_layer_2_ = weight_initial_hidden_acc_layer_2;
    weight_initial_hidden_steer_layer_2_ = weight_initial_hidden_steer_layer_2;
    weight_initial_hidden_gru_ih_ = weight_initial_hidden_gru_ih;
    weight_initial_hidden_gru_hh_ = weight_initial_hidden_gru_hh;
    weight_initial_hidden_final_layer_ = weight_initial_hidden_final_layer;
    weight_initial_hidden_fusion_layer_ = weight_initial_hidden_fusion_layer;
    weight_initial_hidden_only_online_ = weight_initial_hidden_only_online;
    bias_initial_hidden_acc_layer_1_ = bias_initial_hidden_acc_layer_1;
    bias_initial_hidden_steer_layer_1_ = bias_initial_hidden_steer_layer_1;
    bias_initial_hidden_acc_layer_2_ = bias_initial_hidden_acc_layer_2;
    bias_initial_hidden_steer_layer_2_ = bias_initial_hidden_steer_layer_2;
    bias_initial_hidden_gru_ih_ = bias_initial_hidden_gru_ih;
    bias_initial_hidden_gru_hh_ = bias_initial_hidden_gru_hh;
    bias_initial_hidden_final_layer_ = bias_initial_hidden_final_layer;
    bias_initial_hidden_fusion_layer_ = bias_initial_hidden_fusion_layer;
    bias_initial_hidden_only_online_ = bias_initial_hidden_only_online;
    h_gru_dim_ = weight_initial_hidden_gru_hh_.cols();
  }
  void GetInitialHidden::set_offline_features(const Eigen::VectorXd & offline_features)
  {
    offline_features_ = offline_features;
    use_offline_features_ = true;
  }
  void GetInitialHidden::update_gru(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_gru, Eigen::VectorXd & h_gru_next)
  {
    Eigen::VectorXd acc_sub(acc_queue_size_ + predict_step_ + 2);
    Eigen::VectorXd steer_sub(steer_queue_size_ + predict_step_ + 2);
    acc_sub << x[vel_index_], x[acc_index_], x.segment(acc_input_start_index_, acc_queue_size_ + predict_step_);
    steer_sub << x[vel_index_], x[steer_index_], x.segment(steer_input_start_index_, steer_queue_size_ + predict_step_);
    const Eigen::VectorXd acc_layer_1 = relu(weight_initial_hidden_acc_layer_1_ * acc_sub + bias_initial_hidden_acc_layer_1_);
    const Eigen::VectorXd steer_layer_1 = relu(weight_initial_hidden_steer_layer_1_ * steer_sub + bias_initial_hidden_steer_layer_1_);
    const Eigen::VectorXd acc_layer_2 = relu(weight_initial_hidden_acc_layer_2_ * acc_layer_1 + bias_initial_hidden_acc_layer_2_);
    const Eigen::VectorXd steer_layer_2 = relu(weight_initial_hidden_steer_layer_2_ * steer_layer_1 + bias_initial_hidden_steer_layer_2_);
    Eigen::VectorXd h1 = Eigen::VectorXd(1 + acc_layer_2.size() + steer_layer_2.size());
    h1 << x[vel_index_], acc_layer_2, steer_layer_2;
    Eigen::VectorXd gru_ih_part = weight_initial_hidden_gru_ih_ * h1 + bias_initial_hidden_gru_ih_;
    Eigen::VectorXd gru_hh_part = weight_initial_hidden_gru_hh_ * h_gru + bias_initial_hidden_gru_hh_;
    Eigen::VectorXd r = sigmoid(gru_ih_part.segment(0, h_gru_dim_) + gru_hh_part.segment(0, h_gru_dim_));
    Eigen::VectorXd z = sigmoid(gru_ih_part.segment(h_gru_dim_, h_gru_dim_) + gru_hh_part.segment(h_gru_dim_, h_gru_dim_));
    Eigen::VectorXd u_n = gru_ih_part.segment(2 * h_gru_dim_, h_gru_dim_).array() + r.array() * gru_hh_part.segment(2 * h_gru_dim_, h_gru_dim_).array();
    Eigen::VectorXd n = tanh(u_n);
    h_gru_next = (1 - z.array()) * n.array() + z.array() * h_gru.array();  
  }
  Eigen::VectorXd GetInitialHidden::get_initial_hidden(Eigen::VectorXd & h_gru)
  {
    Eigen::VectorXd online_out = tanh(weight_initial_hidden_final_layer_ * h_gru + bias_initial_hidden_final_layer_);
    if (use_offline_features_)
    {
      Eigen::VectorXd fused_out(offline_features_.size() + online_out.size());
      fused_out << offline_features_, online_out;
      return tanh(weight_initial_hidden_fusion_layer_ * fused_out + bias_initial_hidden_fusion_layer_);
    }
    else
    {
      return tanh(weight_initial_hidden_only_online_ * online_out + bias_initial_hidden_only_online_);
    }
  }

  //GetInitialHidden
  /*
  GetInitialHidden::GetInitialHidden() {}
  GetInitialHidden::~GetInitialHidden() {}
  void GetInitialHidden::set_attention_params(
    const Eigen::MatrixXd & weight_initial_hidden_acc_layer_1, const Eigen::MatrixXd & weight_initial_hidden_steer_layer_1,
    const Eigen::MatrixXd & weight_initial_hidden_acc_layer_2, const Eigen::MatrixXd & weight_initial_hidden_steer_layer_2,
    const Eigen::MatrixXd & weight_initial_hidden_gru_ih, const Eigen::MatrixXd & weight_initial_hidden_gru_hh,
    const Eigen::MatrixXd & initial_hidden_query, const Eigen::MatrixXd & weight_initial_hidden_key_layer,
    const Eigen::MatrixXd & weight_initial_hidden_value_layer, const Eigen::MatrixXd & weight_initial_hidden_final_layer,
    const Eigen::VectorXd & bias_initial_hidden_acc_layer_1, const Eigen::VectorXd & bias_initial_hidden_steer_layer_1,
    const Eigen::VectorXd & bias_initial_hidden_acc_layer_2, const Eigen::VectorXd & bias_initial_hidden_steer_layer_2,
    const Eigen::VectorXd & bias_initial_hidden_gru_ih, const Eigen::VectorXd & bias_initial_hidden_gru_hh,
    const Eigen::VectorXd & bias_initial_hidden_value_layer, const Eigen::VectorXd & bias_initial_hidden_final_layer,
    const int num_heads, const int key_size, const int value_size, const int mean_steps
  )
  {
    weight_initial_hidden_acc_layer_1_ = weight_initial_hidden_acc_layer_1;
    weight_initial_hidden_steer_layer_1_ = weight_initial_hidden_steer_layer_1;
    weight_initial_hidden_acc_layer_2_ = weight_initial_hidden_acc_layer_2;
    weight_initial_hidden_steer_layer_2_ = weight_initial_hidden_steer_layer_2;
    weight_initial_hidden_gru_ih_ = weight_initial_hidden_gru_ih;
    weight_initial_hidden_gru_hh_ = weight_initial_hidden_gru_hh;
    initial_hidden_query_ = initial_hidden_query;
    weight_initial_hidden_key_layer_ = weight_initial_hidden_key_layer;
    weight_initial_hidden_value_layer_ = weight_initial_hidden_value_layer;
    weight_initial_hidden_final_layer_ = weight_initial_hidden_final_layer;
    bias_initial_hidden_acc_layer_1_ = bias_initial_hidden_acc_layer_1;
    bias_initial_hidden_steer_layer_1_ = bias_initial_hidden_steer_layer_1;
    bias_initial_hidden_acc_layer_2_ = bias_initial_hidden_acc_layer_2;
    bias_initial_hidden_steer_layer_2_ = bias_initial_hidden_steer_layer_2;
    bias_initial_hidden_gru_ih_ = bias_initial_hidden_gru_ih;
    bias_initial_hidden_gru_hh_ = bias_initial_hidden_gru_hh;
    bias_initial_hidden_value_layer_ = bias_initial_hidden_value_layer;
    bias_initial_hidden_final_layer_ = bias_initial_hidden_final_layer;
    num_heads_ = num_heads;
    key_size_ = key_size;
    value_size_ = value_size;
    mean_steps_ = mean_steps;
    h_gru_dim_ = weight_initial_hidden_gru_hh_.cols();
  }
  void GetInitialHidden::update_gru(
    const Eigen::VectorXd & x, const Eigen::VectorXd & h_gru, Eigen::VectorXd & h_gru_next)
  {
    Eigen::VectorXd acc_sub(acc_queue_size_ + predict_step_ + 2);
    Eigen::VectorXd steer_sub(steer_queue_size_ + predict_step_ + 2);
    acc_sub << x[vel_index_], x[acc_index_], x.segment(acc_input_start_index_, acc_queue_size_ + predict_step_);
    steer_sub << x[vel_index_], x[steer_index_], x.segment(steer_input_start_index_, steer_queue_size_ + predict_step_);
    const Eigen::VectorXd acc_layer_1 = relu(weight_initial_hidden_acc_layer_1_ * acc_sub + bias_initial_hidden_acc_layer_1_);
    const Eigen::VectorXd steer_layer_1 = relu(weight_initial_hidden_steer_layer_1_ * steer_sub + bias_initial_hidden_steer_layer_1_);
    const Eigen::VectorXd acc_layer_2 = relu(weight_initial_hidden_acc_layer_2_ * acc_layer_1 + bias_initial_hidden_acc_layer_2_);
    const Eigen::VectorXd steer_layer_2 = relu(weight_initial_hidden_steer_layer_2_ * steer_layer_1 + bias_initial_hidden_steer_layer_2_);
    Eigen::VectorXd h1 = Eigen::VectorXd(1 + acc_layer_2.size() + steer_layer_2.size());
    h1 << x[vel_index_], acc_layer_2, steer_layer_2;
    Eigen::VectorXd gru_ih_part = weight_initial_hidden_gru_ih_ * h1 + bias_initial_hidden_gru_ih_;
    Eigen::VectorXd gru_hh_part = weight_initial_hidden_gru_hh_ * h_gru + bias_initial_hidden_gru_hh_;
    Eigen::VectorXd r = sigmoid(gru_ih_part.segment(0, h_gru_dim_) + gru_hh_part.segment(0, h_gru_dim_));
    Eigen::VectorXd z = sigmoid(gru_ih_part.segment(h_gru_dim_, h_gru_dim_) + gru_hh_part.segment(h_gru_dim_, h_gru_dim_));
    Eigen::VectorXd u_n = gru_ih_part.segment(2 * h_gru_dim_, h_gru_dim_).array() + r.array() * gru_hh_part.segment(2 * h_gru_dim_, h_gru_dim_).array();
    Eigen::VectorXd n = tanh(u_n);
    h_gru_next = (1 - z.array()) * n.array() + z.array() * h_gru.array();  
  }
  void GetInitialHidden::initialize_x_queue(){
    x_queue_.clear();
  }
  void GetInitialHidden::update_x_queue(const Eigen::VectorXd & x){
    x_queue_.push_back(x);
  }
  void GetInitialHidden::initialize_attention_info(){
    attention_queue_.clear();
    value_queue_.clear();
  }
  void GetInitialHidden::erase_attention_info_begin(){
    attention_queue_.erase(attention_queue_.begin());
    value_queue_.erase(value_queue_.begin());
  }
  void GetInitialHidden::add_weight_and_value(const Eigen::VectorXd & h_gru){
    Eigen::VectorXd key = weight_initial_hidden_key_layer_ * h_gru;
    Eigen::VectorXd attention_weight = Eigen::VectorXd::Zero(num_heads_);
    for (int i = 0; i < num_heads_; i++){
      Eigen::VectorXd query = initial_hidden_query_.row(i).transpose();
      attention_weight[i] = query.dot(key.segment(i * key_size_, key_size_)) / std::sqrt(key_size_);
    }
    Eigen::VectorXd attention_weight_exp = attention_weight.array().exp(); 
    Eigen::VectorXd value = weight_initial_hidden_value_layer_ * h_gru + bias_initial_hidden_value_layer_;
    attention_queue_.push_back(attention_weight_exp);
    value_queue_.push_back(value);
  }
  void GetInitialHidden::update_attention_info(){
    Eigen::VectorXd h_gru_mean = Eigen::VectorXd::Zero(h_gru_dim_);
    Eigen::VectorXd h_gru = Eigen::VectorXd::Zero(h_gru_dim_);
    for (int i = 0; i < int(x_queue_.size()); i++){
      Eigen::VectorXd h_gru_next;
      update_gru(x_queue_[i], h_gru, h_gru_next);
      h_gru = h_gru_next;
      if (i >= int(x_queue_.size()) - mean_steps_){
        h_gru_mean += h_gru/mean_steps_;
      }
    }
    add_weight_and_value(h_gru_mean);
  }
  Eigen::VectorXd GetInitialHidden::get_initial_hidden(){
    Eigen::VectorXd context = Eigen::VectorXd::Zero(value_size_ * num_heads_);
    Eigen::VectorXd attention_weight_exp_sum = Eigen::VectorXd::Zero(num_heads_);
    for (int i = 0; i < int(attention_queue_.size()); i++){
      attention_weight_exp_sum += attention_queue_[i];
    }
    for (int i = 0; i < int(attention_queue_.size()); i++){
      for (int j = 0; j < num_heads_; j++){
        context.segment(j * value_size_, value_size_) += attention_queue_[i][j] / attention_weight_exp_sum[j] * value_queue_[i].segment(j * value_size_, value_size_);
      }
    }
    Eigen::VectorXd h_initial_hidden = tanh(weight_initial_hidden_final_layer_ * context + bias_initial_hidden_final_layer_);
    return h_initial_hidden;    
  }
  */