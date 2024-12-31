<p align="center">
  <a href="https://proxima-ai-tech.com/">
    <img width="500px" src="./images/proxima_logo.png">
  </a>
</p>

# Autoware Vehicle Adaptor

In order to perform adaptive control that allows various vehicles to operate without detailed parameter calibration, it is placed in the lower part of the controller and corrects the input so that the ideal state based on the internal model of the controller is achieved according to the learned model.
However, if you are not going to do any learning and simply want to drive on Autoware with the default NN model, you do not need to follow this installation procedure, and it is sufficient to just do the normal Autoware setup.



# Provided features

To use all the functions of this package, execute the following setup command on the autoware_vehicle_adaptor directory:

```bash
pip3 install .
```

For the upcoming setup, execute the following command:

```bash
pip3 install -U .
```

For the training and evaluation, we can use the following rosbag record command:
```bash
ros2 bag record /localization/kinematic_state /localization/acceleration /vehicle/status/steering_status /control/command/control_cmd /control/trajectory_follower/lane_departure_checker_node/debug/deviation/lateral /system/operation_mode/state /vehicle/raw_vehicle_cmd_converter/debug/compensated_control_cmd /external/selected/control_cmd /control/command/actuation_cmd 
```
Not all of the topics listed here are always published.
The following topics are always required:
```bash
/localization/kinematic_state /localization/acceleration /vehicle/status/steering_status /system/operation_mode/state
```
Either of the following two is required, and the former is published when the vehicle is automatically driven by a `trajectory_follower_node`, and the latter is published when the data collection tool in the `autoware_tools` is used.
```bash
/control/command/control_cmd /external/selected/control_cmd
```
The following topic is required if you want to evaluate lateral deviation:
```bash
/control/trajectory_follower/lane_departure_checker_node/debug/deviation/lateral
```
The following topic is required if you want to evaluate the driving data with the vehicle_adaptor or include the driving data with the vehicle_adaptor in the training data.
```bash
/vehicle/raw_vehicle_cmd_converter/debug/compensated_control_cmd
```
The following topic is required if you want to use the NN based accel brake map calibrator or evaluate the accel brake inputs:
```bash
/control/command/actuation_cmd
```

To convert a rosbag file into a form that can be used for our training and evaluation, execute the following command:

```python
from autoware_vehicle_adaptor.data_analyzer import rosbag_to_csv
rosbag_to_csv.rosbag_to_csv(rosbag_dir)
```
Here, `rosbag_dir` represents the rosbag directory.
As a result, the necessary CSV files are generated in `rosbag_dir`.

## Trajectory following with vehicle adaptor

To use vehicle adaptor, you need set `use_vehicle_adaptor: true` and `enable_control_cmd_horizon_pub: true`.
When you start up autoware and perform automatic driving, the controller's input values are corrected based on the NN (ensemble) models saved in [vehicle_models](./vehicle_models).
By default, the models trained based on drive with nominal parameter are saved.
If you want to train models based on specific vehicle data and make changes, please refer to the next section.
The parameters in [controller_param.yaml](./autoware_vehicle_adaptor/param/controller_param.yaml), which includes following, need to be adjusted to match those of the controller.

| Parameter                                        | Type  | Description                    |
| ------------------------------------------------ | ----- | ------------------------------ |
| controller_parameter:vehicle_info:wheel_base        | float | wheel base [m]                 |
| controller_parameter:acceleration:acc_time_delay    | float | acceleration time delay [s]    |
| controller_parameter:acceleration:acc_time_constant | float | acceleration time constant [s] |
| controller_parameter:steering:steer_time_delay      | float | steer time delay [s]           |
| controller_parameter:steering:steer_time_constant   | float | steer time constant [s]        |

We test our vehicle adaptor with the following route:

<p style="text-align: center;">
    <img src="images/sample_map_root.png" width="712px">
</p>

The parameter `optimization_parameter:autoware_alignment:use_vehicle_adaptor` in [optimization_param.yaml](./autoware_vehicle_adaptor/param/optimization_param.yaml) is switched between `false` and `true` to compare driving without using the Vehicle Adaptor and driving with it.
Here, we use the trained [sample model](./vehicle_models).
The performance of the drive can be evaluated using [driving_log_plotter.py](./autoware_vehicle_adaptor/data_analyzer/driving_log_plotter.py).
Please refer to [test_drive_log_plotter.ipynb](./autoware_vehicle_adaptor/data_analyzer/test_drive_log_plotter.ipynb) for sample code for performance evaluation.
We will only display the resulting graphs for lateral deviation and steering here.

The following shows a nominal autoware mpc drive without using Vehicle Adaptor, with a lateral deviation of about 40 cm.

<p style="text-align: center;">
    <img src="images/sample_map_sim_nominal_controller.png" width="712px">
</p>

By using Vehicle Adaptor, the lateral deviation was reduced to about 30 cm, and driving was improved.

<p style="text-align: center;">
    <img src="images/sample_map_sim_vehicle_adaptor.png" width="712px">
</p>


## Training and its evaluation

[test_vehicle_adaptor_model_training.ipynb](./autoware_vehicle_adaptor/data_analyzer/test_vehicle_adaptor_model_training.ipynb) is a useful reference for training the models.
The training is done using the gray box model, and the parameters of the nominal model are specified in [nimonal_param.yaml](./autoware_vehicle_adaptor/param/nominal_param.yaml).
The parameters that can be specified are the same as for [controller_param.yaml](./autoware_vehicle_adaptor/param/controller_param.yaml).
This parameter needs to be consistent during training and driving, but it does not need to match [controller_param.yaml](./autoware_vehicle_adaptor/param/controller_param.yaml), and [controller_param.yaml](./autoware_vehicle_adaptor/param/controller_param.yaml) can be changed to match the controller when driving even after training has been performed.

In order to train the models, training and validation data are required.
With the paths of the rosbag directories used for training and validation, `dir_train_0`, `dir_train_1`, `dir_train_2`,..., `dir_val_0`, `dir_val_1`, `dir_val_2`,... and the directory `save_dir` where you save the models, the model can be trained and saved in the python environment as follows:

```python
from autoware_vehicle_adaptor.data_analyzer import driving_learners
model_trainer = driving_learners.train_error_prediction_NN()
model_trainer.add_data_from_csv(dir_train_0, add_mode="as_train")
model_trainer.add_data_from_csv(dir_train_1, add_mode="as_train")
model_trainer.add_data_from_csv(dir_train_2, add_mode="as_train")
...
model_trainer.add_data_from_csv(dir_val_0, add_mode="as_val")
model_trainer.add_data_from_csv(dir_val_1, add_mode="as_val")
model_trainer.add_data_from_csv(dir_val_2, add_mode="as_val")
...
model_trainer.get_trained_model()
model_trainer.save_models(save_dir)
model_trainer.get_trained_ensemble_models(batch_sizes=[100],ensemble_size=5)
paths = [save_dir + "/vehicle_model_" + str(i+1) + ".pth" for i in range(5)]
model_trainer.save_ensemble_models(paths=paths)
```
If you rename `save_dir` to `vehicle_models` and place it in `autoware_vehicle_adaptor`, it will be loaded when using vehicle adaptor.

In some cases, training once may not work because it falls into the local minimum, but [test_vehicle_adaptor_model_training_with_relearn.ipynb](./autoware_vehicle_adaptor/data_analyzer/test_vehicle_adaptor_model_training_with_relearn.ipynb) can be used as a reference when repeating training to avoid this.

[test_NN_model_evaluator.ipynb](./autoware_vehicle_adaptor/data_analyzer/test_NN_model_evaluator.ipynb) is useful for evaluating how well the trained model can predict the dynamics of the driving data.
The default model is evaluated as follows.
<p style="text-align: center;">
    <img src="images/NN_evaluator_result.png" width="712px">
</p>

## Brief simulation using a python simulator

First, to give the steer time constant in the python simulator, create the following file and save it in [autoware_vehicle_adaptor/python_simulator/supporting_data](./autoware_vehicle_adaptor/python_simulator/supporting_data) with the name `sim_setting.json`:

```json
{ "steer_time_constant": 0.5}
```

Next, after moving to [autoware_vehicle_adaptor/python_simulator](./autoware_vehicle_adaptor/python_simulator), run the following commands to test the slalom driving on the python simulator with the nominal control:

```bash
python3 run_vehicle_adaptor.py nominal_test (nominal_dir)
```

The results are saved in a directory that the user can specify, here `python_simulator/log_data/test_python_nominal_sim_(nominal_dir)`.

The following results were obtained.

<p style="text-align: center;">
    <img src="images/nominal_sim_steer_time_constant_0_5.png" width="712px">
</p>

The center of the upper row represents the lateral deviation.

To perform training using a figure eight driving and driving with Vehicle Adaptor based on the obtained model, run the following commands:

```bash
python3 run_vehicle_adaptor.py (trained_dir)
```

The result of the driving is stored in `python_simulator/log_data/test_python_vehicle_adaptor_sim_(trained_dir)`.


The following results were obtained.

<p style="text-align: center;">
    <img src="images/vehicle_adaptor_sim_steer_time_constant_0_5.png" width="712px">
</p>

The following are some of the parameters that can be specified:

| Parameter                | Type        | Description                                                                                                                                                                                                                                                                                  |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| steer_dead_band          | float       | steer dead band [rad]                                                                                                                                                                                                                                                                        |
| acc_time_delay           | float       | acceleration time delay [s]                                                                                                                                                                                                                                                                  |
| steer_time_delay         | float       | steer time delay [s]                                                                                                                                                                                                                                                                         |
| acc_time_constant        | float       | acceleration time constant [s]                                                                                                                                                                                                                                                               |
| steer_time_constant      | float       | steer time constant [s]                                                                                                                                                                                                                                                                      |

For example, to give the simulation side 0.5 [sec] of steer time constant and 0.001 [rad] of steer dead band, edit the `sim_setting.json` as follows.

```json
{ "steer_time_constant": 0.5, "steer_dead_band": 0.001 }
```

Please refer to [parameter_change_utils.py](./autoware_vehicle_adaptor/python_simulator/utils/parameter_change_utils.py) for other parameters.
The test method in [run_auto_parameter_change_sim.py](./autoware_vehicle_adaptor/python_simulator/run_auto_parameter_change_sim.py)　is also helpful.

# Parameter description

The important parameters in the Vehicle Adaptor's changeable parameter file [optimization_param.yaml](./autoware_vehicle_adaptor/param/optimization_param.yaml) are as follows:


| Parameter                                  | Type        | Description                                                                                                                                                                                                                                        |
| ------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| weight_parameter:x_cost                  | float         | longitudinal error stage cost weight                                                                                                               |
| weight_parameter:y_cost                  | float         | lateral error stage cost weight                                                                                                               |
| weight_parameter:vel_cost                  | float         | velocity error stage cost weight                                                                                                               |
| weight_parameter:yaw_cost                  | float         | yaw angle error stage cost weight                                                                                                               |
| weight_parameter:acc_cost                  | float         | acceleration error stage cost weight                                                                                                               |
| weight_parameter:steer_cost                  | float         | steer angle error stage cost weight                                                                                                               |
| weight_parameter:x_terminal_cost                  | float         | longitudinal error terminal cost weight                                                                                                               |
| weight_parameter:y_terminal_cost                  | float         | lateral error stage terminal weight                                                                                                               |
| weight_parameter:vel_terminal_cost                  | float         | velocity error stage terminal weight                                                                                                               |
| weight_parameter:yaw_terminal_cost                  | float         | yaw angle error stage terminal weight                                                                                                               |
| weight_parameter:acc_terminal_cost                  | float         | acceleration error stage terminal weight                                                                                                               |
| weight_parameter:steer_terminal_cost                  | float         | steer angle error stage terminal weight                                                                                                               |
| weight_parameter:intermediate_cost_index                  | int         | The intermediate point horizon number that increases the weight                                                                  |
| weight_parameter:x_intermediate_cost                  | float         | longitudinal error intermediate cost weight                                                                                                               |
| weight_parameter:y_intermediate_cost                  | float         | lateral error stage intermediate weight                                                                                                               |
| weight_parameter:vel_intermediate_cost                  | float         | velocity error stage intermediate weight                                                                                                               |
| weight_parameter:yaw_intermediate_cost                  | float         | yaw angle error stage intermediate weight                                                                                                               |
| weight_parameter:acc_intermediate_cost                  | float         | acceleration error stage intermediate weight                                                                                                               |
| weight_parameter:steer_intermediate_cost                  | float         | steer angle error stage intermediate weight                                                                                                               |


# Limitation

* The performance when the horizon of the steering input is not given has not been verified.
* If the dynamics given to the controller are accurate, the performance may degrade.
