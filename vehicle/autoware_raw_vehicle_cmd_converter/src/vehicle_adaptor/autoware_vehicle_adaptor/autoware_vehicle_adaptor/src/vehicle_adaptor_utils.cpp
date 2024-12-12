#include "vehicle_adaptor_utils.h"

Eigen::VectorXd tanh(const Eigen::VectorXd & v)
{
  return v.array().tanh();
}
Eigen::VectorXd d_tanh(const Eigen::VectorXd & v)
{
  return 1 / (v.array().cosh() * v.array().cosh());
}
Eigen::VectorXd sigmoid(const Eigen::VectorXd & v)
{
  return 0.5 * (0.5 * v).array().tanh() + 0.5;
}
Eigen::VectorXd relu(const Eigen::VectorXd & x)
{
  Eigen::VectorXd x_ = x;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] < 0) {
      x_[i] = 0;
    }
  }
  return x_;
}
Eigen::VectorXd d_relu(const Eigen::VectorXd & x)
{
  Eigen::VectorXd result = Eigen::VectorXd::Ones(x.size());
  for (int i = 0; i < x.size(); i++) {
    if (x[i] < 0) {
      result[i] = 0;
    }
  }
  return result;
}
Eigen::MatrixXd d_relu_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x)
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m.rows(), m.cols());
  for (int i = 0; i < m.cols(); i++) {
    if (x[i] >= 0) {
      result.col(i) = m.col(i);
    }
  }
  return result;
}
Eigen::MatrixXd d_tanh_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x)
{
  Eigen::MatrixXd result = Eigen::MatrixXd(m.rows(), m.cols());
  for (int i = 0; i < m.cols(); i++) {
    result.col(i) = m.col(i) / (std::cosh(x[i]) * std::cosh(x[i]));
  }
  return result;
}
Eigen::VectorXd d_tanh_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x)
{
  Eigen::VectorXd result = Eigen::VectorXd(v.size());
  for (int i = 0; i < v.size(); i++) {
    result[i] = v[i] / (std::cosh(x[i]) * std::cosh(x[i]));
  }
  return result;
}
Eigen::MatrixXd d_sigmoid_product(const Eigen::MatrixXd & m, const Eigen::VectorXd & x)
{
  Eigen::MatrixXd result = Eigen::MatrixXd(m.rows(), m.cols());
  for (int i = 0; i < m.cols(); i++) {
    result.col(i) = 0.25 * m.col(i) / (std::cosh(0.5 * x[i]) * std::cosh(0.5 * x[i]));
  }
  return result;
}
Eigen::VectorXd d_sigmoid_product_vec(const Eigen::VectorXd & v, const Eigen::VectorXd & x)
{
  Eigen::VectorXd result = Eigen::VectorXd(v.size());
  for (int i = 0; i < v.size(); i++) {
    result[i] = 0.25 * v[i] / (std::cosh(0.5 * x[i]) * std::cosh(0.5 * x[i]));
  }
  return result;
}

Eigen::VectorXd rotate_data(Eigen::VectorXd states, double yaw)
{
  Eigen::VectorXd states_rotated = states;
  double cos_yaw = std::cos(yaw);
  double sin_yaw = std::sin(yaw);
  states_rotated[0] = states[0] * cos_yaw + states[1] * sin_yaw;
  states_rotated[1] = -states[1] * sin_yaw + states[0] * cos_yaw;
  return states_rotated;
}

Eigen::VectorXd vector_power(Eigen::VectorXd vec, int power)
{
  Eigen::VectorXd result = vec;
  for (int i = 0; i < power - 1; i++) {
    result = result.array() * vec.array();
  }
  return result;
}

double double_power(double val, int power)
{
  double result = val;
  for (int i = 0; i < power - 1; i++) {
    result = result * val;
  }
  return result;
}
double calc_table_value(double domain_value, std::vector<double> domain_table, std::vector<double> target_table)
{
  if (domain_value <= domain_table[0]) {
    return target_table[0];
  }
  if (domain_value >= domain_table[domain_table.size() - 1]) {
    return target_table[target_table.size() - 1];
  }
  for (int i = 0; i < int(domain_table.size()) - 1; i++) {
    if (domain_value >= domain_table[i] && domain_value <= domain_table[i + 1]) {
      return target_table[i] +
             (target_table[i + 1] - target_table[i]) /
               (domain_table[i + 1] - domain_table[i]) *
               (domain_value - domain_table[i]);
    }
  }
  return 0.0;
}

std::string get_param_dir_path()
{
  std::string build_path = BUILD_PATH;
  std::string param_dir_path = build_path + "/autoware_vehicle_adaptor/param";
  return param_dir_path;
}
Eigen::MatrixXd read_csv(std::string file_path)
{
  std::string build_path = BUILD_PATH;
  std::string file_path_abs = build_path + "/" + file_path;
  std::vector<std::vector<double>> csv_data;
  std::ifstream ifs(file_path_abs);
  if (!ifs) {
    std::cerr << "Failed to open file." << std::endl;
    return Eigen::MatrixXd(0, 0);
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream stream(line);
    std::string field;
    std::vector<double> fields;
    while (std::getline(stream, field, ',')) {
      fields.push_back(std::stod(field));
    }
    csv_data.push_back(fields);
  }
  ifs.close();
  Eigen::MatrixXd data = Eigen::MatrixXd(csv_data.size(), csv_data[0].size());
  for (int i = 0; i < int(csv_data.size()); i++) {
    for (int j = 0; j < int(csv_data[0].size()); j++) {
      data(i, j) = csv_data[i][j];
    }
  }
  return data;
}
std::vector<std::string> read_string_csv(std::string file_path)
{
  std::string build_path = BUILD_PATH;
  std::string file_path_abs = build_path + "/" + file_path;
  std::vector<std::string> csv_data;
  std::ifstream ifs(file_path_abs);
  if (!ifs) {
    std::cerr << "Failed to open file." << std::endl;
    return csv_data;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) {
      field.erase(field.find_last_not_of(" \n\r\t") + 1);
      field.erase(0, field.find_first_not_of(" \n\r\t"));
      csv_data.push_back(field);
    }
  }
  ifs.close();
  return csv_data;
}
Eigen::VectorXd interpolate_eigen(Eigen::VectorXd y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new)
{
  Eigen::VectorXd y_new = Eigen::VectorXd(time_stamp_new.size());
  int lower_bound_index = 0;
  for (int i = 0; i < int(time_stamp_new.size()); i++) {
    if (time_stamp_new[i] >= time_stamp_obs[time_stamp_obs.size() - 1]) {
      y_new[i] = y[time_stamp_obs.size() - 1] + (y[time_stamp_obs.size() - 1] - y[time_stamp_obs.size() - 2]) / (time_stamp_obs[time_stamp_obs.size() - 1] - time_stamp_obs[time_stamp_obs.size() - 2]) * (time_stamp_new[i] - time_stamp_obs[time_stamp_obs.size() - 1]);
      continue;
    }
    for (int j = lower_bound_index; j < int(time_stamp_obs.size()) - 1; j++) {
      if (time_stamp_new[i] >= time_stamp_obs[j] && time_stamp_new[i] <= time_stamp_obs[j + 1]) {
        y_new[i] = y[j] + (y[j + 1] - y[j]) / (time_stamp_obs[j + 1] - time_stamp_obs[j]) * (time_stamp_new[i] - time_stamp_obs[j]);
        lower_bound_index = j;
        break;
      }
    }
  }
  return y_new;
}
std::vector<Eigen::VectorXd> interpolate_vector(std::vector<Eigen::VectorXd> y, std::vector<double> time_stamp_obs, std::vector<double> time_stamp_new)
{
  std::vector<Eigen::VectorXd> y_new;
  int lower_bound_index = 0;
  for (int i = 0; i < int(time_stamp_new.size()); i++) {
    if (time_stamp_new[i] >= time_stamp_obs[time_stamp_obs.size() - 1]) {
      y_new.push_back(y[time_stamp_obs.size() - 1] + (y[time_stamp_obs.size() - 1] - y[time_stamp_obs.size() - 2]) / (time_stamp_obs[time_stamp_obs.size() - 1] - time_stamp_obs[time_stamp_obs.size() - 2]) * (time_stamp_new[i] - time_stamp_obs[time_stamp_obs.size() - 1]));
      continue;
    }
    for (int j = lower_bound_index; j < int(time_stamp_obs.size()) - 1; j++) {
      if (time_stamp_new[i] >= time_stamp_obs[j] && time_stamp_new[i] <= time_stamp_obs[j + 1]) {
        y_new.push_back(y[j] + (y[j + 1] - y[j]) / (time_stamp_obs[j + 1] - time_stamp_obs[j]) * (time_stamp_new[i] - time_stamp_obs[j]));
        lower_bound_index = j;
        break;
      }
    }
  }
  return y_new;
}
