#ifndef VEHICLE_ADAPTOR_UTILS_H
#define VEHICLE_ADAPTOR_UTILS_H
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cmath>
#include "inputs_prediction.h"
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

Eigen::VectorXd tanh(const Eigen::VectorXd & v);
Eigen::VectorXd d_tanh(const Eigen::VectorXd & v);
Eigen::VectorXd sigmoid(const Eigen::VectorXd & v);
Eigen::VectorXd relu(const Eigen::VectorXd & x);
Eigen::VectorXd d_relu(const Eigen::VectorXd & x);
Eigen::MatrixXd d_relu_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x);
Eigen::MatrixXd d_tanh_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x);
Eigen::VectorXd d_tanh_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x);
Eigen::MatrixXd d_sigmoid_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x);
Eigen::VectorXd d_sigmoid_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x);
Eigen::VectorXd rotate_data(Eigen::VectorXd states, double yaw);
Eigen::VectorXd vector_power(Eigen::VectorXd vec, int power);
double double_power(double val, int power);
double calc_table_value(double domain_value, std::vector<double> domain_table, std::vector<double> target_table);
std::string get_param_dir_path();
Eigen::MatrixXd read_csv(std::string file_path);
std::vector<std::string> read_string_csv(std::string file_path);
Eigen::VectorXd interpolate_eigen(Eigen::VectorXd y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new);
std::vector<Eigen::VectorXd> interpolate_vector(std::vector<Eigen::VectorXd> y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new);
#endif // VEHICLE_ADAPTOR_UTILS_H