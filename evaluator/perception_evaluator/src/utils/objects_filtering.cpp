// Copyright 2024 TIER IV, Inc.
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

#include "perception_evaluator/utils/objects_filtering.hpp"

// clang-format off
namespace {
std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens; std::string token; int p = 0;
  for (char c : s) {
    if (c == '(') p++; else if (c == ')') p--;
    if (c == delimiter && p == 0) { tokens.push_back(token); token.clear(); } else token += c;
  }
  if (!token.empty()) tokens.push_back(token);
  return tokens;
}

template <typename T> void view(const std::string &n, T e) { std::cerr << n << ": " << e << ", "; }
template <typename T> void view(const std::string &n, const std::vector<T> &v) { std::cerr << n << ":"; for (const auto &e : v) std::cerr << " " << e; std::cerr << ", "; }
template <typename First, typename... Rest> void view_multi(const std::vector<std::string> &n, First f, Rest... r) { view(n[0], f); if constexpr (sizeof...(r) > 0) view_multi(std::vector<std::string>(n.begin() + 1, n.end()), r...); }

template <typename... Args> void debug_helper(const char *f, int l, const char *n, Args... a) {
  std::cerr << f << ": " << l << ", "; auto nl = split(n, ',');
  for (auto &nn : nl) { nn.erase(nn.begin(), std::find_if(nn.begin(), nn.end(), [](int ch) { return !std::isspace(ch); })); nn.erase(std::find_if(nn.rbegin(), nn.rend(), [](int ch) { return !std::isspace(ch); }).base(), nn.end()); }
  view_multi(nl, a...); std::cerr << std::endl;
}

#define debug(...) debug_helper(__func__, __LINE__, #__VA_ARGS__, __VA_ARGS__)
#define line() { std::cerr << "(" << __FILE__ << ") " << __func__ << ": " << __LINE__ << std::endl; }
} // namespace
// clang-format on

namespace perception_diagnostics
{
ObjectTypesToCheck getDeviationCheckObjectTypes(
  const std::unordered_map<uint8_t, ObjectParameter> & params)
{
  ObjectTypesToCheck object_types_to_check;
  for (const auto & [object_class, object_param] : params) {
    switch (object_class) {
      case ObjectClassification::CAR:
        object_types_to_check.check_car = object_param.check_deviation;
        break;
      case ObjectClassification::TRUCK:
        object_types_to_check.check_truck = object_param.check_deviation;
        break;
      case ObjectClassification::BUS:
        object_types_to_check.check_bus = object_param.check_deviation;
        break;
      case ObjectClassification::TRAILER:
        object_types_to_check.check_trailer = object_param.check_deviation;
        break;
      case ObjectClassification::BICYCLE:
        object_types_to_check.check_bicycle = object_param.check_deviation;
        break;
      case ObjectClassification::MOTORCYCLE:
        object_types_to_check.check_motorcycle = object_param.check_deviation;
        break;
      case ObjectClassification::PEDESTRIAN:
        object_types_to_check.check_pedestrian = object_param.check_deviation;
        break;
      case ObjectClassification::UNKNOWN:
        object_types_to_check.check_unknown = object_param.check_deviation;
        break;
      default:
        break;
    }
  }
  return object_types_to_check;
}

std::uint8_t getHighestProbLabel(const std::vector<ObjectClassification> & classification)
{
  std::uint8_t label = ObjectClassification::UNKNOWN;
  float highest_prob = 0.0;
  for (const auto & _class : classification) {
    if (highest_prob < _class.probability) {
      highest_prob = _class.probability;
      label = _class.label;
    }
  }

  return label;
}

bool isTargetObjectType(
  const PredictedObject & object, const ObjectTypesToCheck & target_object_types)
{
  const auto t = getHighestProbLabel(object.classification);

  return (
    (t == ObjectClassification::CAR && target_object_types.check_car) ||
    (t == ObjectClassification::TRUCK && target_object_types.check_truck) ||
    (t == ObjectClassification::BUS && target_object_types.check_bus) ||
    (t == ObjectClassification::TRAILER && target_object_types.check_trailer) ||
    (t == ObjectClassification::UNKNOWN && target_object_types.check_unknown) ||
    (t == ObjectClassification::BICYCLE && target_object_types.check_bicycle) ||
    (t == ObjectClassification::MOTORCYCLE && target_object_types.check_motorcycle) ||
    (t == ObjectClassification::PEDESTRIAN && target_object_types.check_pedestrian));
}

void filterObjectsByClass(
  PredictedObjects & objects, const ObjectTypesToCheck & target_object_types)
{
  const auto filter = [&](const auto & object) {
    return isTargetObjectType(object, target_object_types);
  };

  filterObjects(objects, filter);
}

void filterDeviationCheckObjects(
  PredictedObjects & objects, const std::unordered_map<uint8_t, ObjectParameter> & params)
{
  const auto object_types = getDeviationCheckObjectTypes(params);
  filterObjectsByClass(objects, object_types);
}

}  // namespace perception_diagnostics