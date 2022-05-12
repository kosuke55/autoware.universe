// Copyright 2022 The Autoware Foundation
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

#ifndef TRAJECTORY_FOLLOWER__LATERAL_CONTROLLER_HPP_
#define TRAJECTORY_FOLLOWER__LATERAL_CONTROLLER_HPP_

#include "trajectory_follower/lateral_controller.hpp"
#include "trajectory_follower/sync_data.hpp"

namespace autoware
{
namespace motion
{
namespace control
{
namespace trajectory_follower
{
class LateralController
{
public:
  virtual LateralSyncData run() = 0;
  void sync(LongitudinalSyncData longitudinal_sync_data);

protected:
  LongitudinalSyncData longitudinal_sync_data_;
};

}  // namespace trajectory_follower
}  // namespace control
}  // namespace motion
}  // namespace autoware

#endif  // TRAJECTORY_FOLLOWER__LATERAL_CONTROLLER_HPP_