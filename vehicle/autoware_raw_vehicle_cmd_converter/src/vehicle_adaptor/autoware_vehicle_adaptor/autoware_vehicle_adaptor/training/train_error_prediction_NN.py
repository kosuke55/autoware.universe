from autoware_vehicle_adaptor.training import add_data_from_csv
from autoware_vehicle_adaptor.training import error_prediction_NN
from autoware_vehicle_adaptor.training import convert_model_to_csv
from autoware_vehicle_adaptor.training.early_stopping import EarlyStopping
from autoware_vehicle_adaptor.training.training_utils import TrainErrorPredictionNNFunctions, plot_relearned_vs_original_prediction_error
from autoware_vehicle_adaptor.param import parameters
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
import json
import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import yaml
import random
import os
import types
from sklearn.linear_model import Lasso
#torch.autograd.set_detect_anomaly(True)

get_loss = TrainErrorPredictionNNFunctions.get_loss
validate_in_batches = TrainErrorPredictionNNFunctions.validate_in_batches
get_each_component_loss = TrainErrorPredictionNNFunctions.get_each_component_loss
get_losses = TrainErrorPredictionNNFunctions.get_losses
get_signed_prediction_error = TrainErrorPredictionNNFunctions.get_signed_prediction_error
get_sequence_data = TrainErrorPredictionNNFunctions.get_sequence_data

prediction_length = parameters.prediction_length
past_length = parameters.past_length
integration_length = parameters.integration_length
integration_weight = parameters.integration_weight
add_position_to_prediction = parameters.add_position_to_prediction
add_vel_to_prediction = parameters.add_vel_to_prediction
add_yaw_to_prediction = parameters.add_yaw_to_prediction
integrate_states = parameters.integrate_states
integrate_vel = parameters.integrate_vel
integrate_yaw = parameters.integrate_yaw

state_component_predicted = parameters.state_component_predicted
state_component_predicted_index = parameters.state_component_predicted_index
state_name_to_predicted_index = parameters.state_name_to_predicted_index

acc_queue_size = parameters.acc_queue_size
steer_queue_size = parameters.steer_queue_size
control_dt = parameters.control_dt
acc_delay_step = parameters.acc_delay_step
steer_delay_step = parameters.steer_delay_step
acc_time_constant = parameters.acc_time_constant
steer_time_constant = parameters.steer_time_constant
wheel_base = parameters.wheel_base
vel_index = 0
acc_index = 1
steer_index = 2
prediction_step = parameters.mpc_predict_step
acc_input_indices_nom =  np.arange(3 + acc_queue_size -acc_delay_step, 3 + acc_queue_size -acc_delay_step + prediction_step)
steer_input_indices_nom = np.arange(3 + acc_queue_size + prediction_step + steer_queue_size -steer_delay_step, 3 + acc_queue_size + steer_queue_size -steer_delay_step + 2 * prediction_step)



class train_error_prediction_NN(add_data_from_csv.add_data_from_csv):
    """Class for training the error prediction NN."""

    def __init__(self, max_iter=10000, tol=1e-5, alpha_1=0.1**7, alpha_2=0.1**7, alpha_jacobian=0.1**4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_jacobian = alpha_jacobian
        self.models = None
        self.weights_for_dataloader = None

        self.past_length = past_length
        self.prediction_length = prediction_length
        self.prediction_step = prediction_step
        self.acc_queue_size = acc_queue_size
        self.steer_queue_size = steer_queue_size
        self.adaptive_weight = torch.ones(len(state_component_predicted_index)).to(self.device)
    def train_model(
        self,
        model: error_prediction_NN.ErrorPredictionNN,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        batch_sizes: list,
        learning_rates: list,
        patience: int,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        Z_val: torch.Tensor,
        fix_lstm: bool = False,
        randomize_fix_lstm: float = 0.001,
        integration_prob: float = 0.1,
        X_replay: torch.Tensor | None = None,
        Y_replay: torch.Tensor | None = None,
        Z_replay: torch.Tensor | None = None,
        replay_data_rate: float = 0.05,
    ):
        """Train the error prediction NN."""

        model = model.to(self.device)
        print("sample_size: ", X_train.shape[0] + X_val.shape[0])
        print("patience: ", patience)
        # Define the loss function.
        criterion = nn.L1Loss()
        # Fix the LSTM
        if fix_lstm:
            self.fix_lstm(model,randomize=randomize_fix_lstm)
        # save the original adaptive weight
        original_adaptive_weight = self.adaptive_weight.clone()
        print("original_adaptive_weight: ", original_adaptive_weight)
        # Define the optimizer.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        # Define the initial loss.
        initial_loss = validate_in_batches(model,criterion,X_val, Y_val, Z_val, adaptive_weight=self.adaptive_weight)
        print("initial_loss: ", initial_loss)
        batch_size = batch_sizes[0]
        print("batch_size: ", batch_size)
        # Define the early stopping object.
        early_stopping = EarlyStopping(initial_loss, tol=self.tol, patience=patience)
        # Data Loader
        if self.weights_for_dataloader is None:
            weighted_sampler = None
            train_dataset = DataLoader(
                TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, shuffle=True
            )
        else:
            weighted_sampler = WeightedRandomSampler(weights=self.weights_for_dataloader, num_samples=len(self.weights_for_dataloader), replacement=True)
            train_dataset = DataLoader(
                TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, sampler=weighted_sampler
            )
        # learning_rate index
        learning_rate_index = 0
        # batch_size index
        # Print learning rate
        print("learning rate: ", learning_rates[learning_rate_index])
        # Train the model.
        for i in range(self.max_iter):
            model.train()

            for X_batch, Y_batch, Z_batch in train_dataset:
                optimizer.zero_grad()
                # outputs = model(X_batch)
                loss = get_loss(criterion, model, X_batch, Y_batch,Z_batch, adaptive_weight=self.adaptive_weight,integral_prob=integration_prob,alpha_jacobian=self.alpha_jacobian)
                for w in model.parameters():
                    loss += self.alpha_1 * torch.norm(w, 1) + self.alpha_2 * torch.norm(w, 2) ** 2
                if X_replay is not None:
                    replay_mask = torch.rand(X_replay.size(0)) < replay_data_rate * X_train.size(0) / X_replay.size(0)
                    X_replay_batch = X_replay[replay_mask]
                    Y_replay_batch = Y_replay[replay_mask]
                    Z_replay_batch = Z_replay[replay_mask]
                    loss += get_loss(criterion, model, X_replay_batch, Y_replay_batch,Z_replay_batch, adaptive_weight=self.adaptive_weight,integral_prob=integration_prob,alpha_jacobian=self.alpha_jacobian)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = validate_in_batches(model,criterion,X_val, Y_val, Z_val,adaptive_weight=self.adaptive_weight)
            val_loss_with_original_weight = validate_in_batches(model,criterion,X_val, Y_val, Z_val,adaptive_weight=original_adaptive_weight)
            if i % 10 == 1:
                print("epoch: ", i)
                print("val_loss with original weight: ", val_loss_with_original_weight)
                print("val_loss: ", val_loss)
            if early_stopping(val_loss):
                learning_rate_index += 1
                batch_size = batch_sizes[min(learning_rate_index, len(batch_sizes) - 1)]
                if learning_rate_index >= len(learning_rates):
                    break
                batch_size = batch_sizes[min(learning_rate_index, len(batch_sizes) - 1)]
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rates[learning_rate_index]
                )
                print("update learning rate to ", learning_rates[learning_rate_index])
                print("batch size:", batch_size)
                if self.weights_for_dataloader is None:
                    train_dataset = DataLoader(
                        TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, shuffle=True
                    )
                else:
                    train_dataset = DataLoader(
                        TensorDataset(X_train, Y_train, Z_train), batch_size=batch_size, sampler=weighted_sampler
                    )
                early_stopping.reset()
                if learning_rates[learning_rate_index - 1] < 3e-4:
                    self.update_adaptive_weight(model,X_train,Y_train)
                print("adaptive_weight: ", self.adaptive_weight)

    def relearn_model(
        self,
        model: error_prediction_NN.ErrorPredictionNN,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        batch_sizes: list,
        learning_rates: list,
        patience: int,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        Z_val: torch.Tensor,
        fix_lstm: bool = False,
        randomize_fix_lstm: float = 0.001,
        integration_prob: float = 0.1,
        randomize: float = 0.001,
        reset_weight: bool = False,
        X_test=None,
        Y_test=None,
        Z_test=None,
        X_replay=None,
        Y_replay=None,
        Z_replay=None,
        replay_data_rate=0.05,
        plt_save_dir=None,
        window_size=10,
        save_path=None,
        always_update_model=False,
    ):
        print("randomize: ", randomize)
        self.update_adaptive_weight(model,X_train,Y_train)
        original_adaptive_weight = self.adaptive_weight.clone()
        criterion = nn.L1Loss()
        original_train_loss, original_each_component_train_loss, Y_train_pred_origin = get_losses(
            model, criterion, X_train, Y_train, Z_train, adaptive_weight=original_adaptive_weight)
        original_val_loss, original_each_component_val_loss, Y_val_pred_origin = get_losses(
            model, criterion, X_val, Y_val, Z_val, adaptive_weight=original_adaptive_weight)
        if X_test is not None:
            original_test_loss, original_each_component_test_loss, Y_test_pred_origin = get_losses(
                model, criterion, X_test, Y_test, Z_test, adaptive_weight=original_adaptive_weight)
        else:
            original_test_loss = None
            original_each_component_test_loss = None
            Y_test_pred_origin = None
        if reset_weight:
            relearned_model = error_prediction_NN.ErrorPredictionNN(prediction_length=prediction_length,state_component_predicted=state_component_predicted).to(self.device)
        else:
            relearned_model = copy.deepcopy(model)
            relearned_model.lstm_encoder.flatten_parameters()
            relearned_model.lstm.flatten_parameters()
            with torch.no_grad():
                if fix_lstm:
                    relearned_model.complimentary_layer[0].weight += randomize * torch.randn_like(model.complimentary_layer[0].weight)
                    relearned_model.complimentary_layer[0].bias += randomize * torch.randn_like(model.complimentary_layer[0].bias)
                    relearned_model.linear_relu[0].weight += randomize * torch.randn_like(model.linear_relu[0].weight)
                    relearned_model.linear_relu[0].bias += randomize * torch.randn_like(model.linear_relu[0].bias)
                    relearned_model.final_layer.weight += randomize * torch.randn_like(model.final_layer.weight)
                    relearned_model.final_layer.bias += randomize * torch.randn_like(model.final_layer.bias)
                else:
                    for w in relearned_model.parameters():
                        w += randomize * torch.randn_like(w)
        self.train_model(
            relearned_model,
            X_train,
            Y_train,
            Z_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            Z_val,
            fix_lstm=fix_lstm,
            randomize_fix_lstm=randomize_fix_lstm,
            integration_prob=integration_prob,
            X_replay=X_replay,
            Y_replay=Y_replay,
            Z_replay=Z_replay,
            replay_data_rate=replay_data_rate,
        )
        relearned_train_loss, relearned_each_component_train_loss, Y_train_pred_relearned = get_losses(
            relearned_model, criterion, X_train, Y_train, Z_train, adaptive_weight=original_adaptive_weight)
        relearned_val_loss, relearned_each_component_val_loss, Y_val_pred_relearned = get_losses(
            relearned_model, criterion, X_val, Y_val, Z_val, adaptive_weight=original_adaptive_weight)
        if X_test is not None:
            relearned_test_loss, relearned_each_component_test_loss, Y_test_pred_relearned = get_losses(
                relearned_model, criterion, X_test, Y_test, Z_test, adaptive_weight=original_adaptive_weight)
        else:
            relearned_test_loss = None
            relearned_each_component_test_loss = None
            Y_test_pred_relearned = None

        
        nominal_signed_train_prediction_error, original_signed_train_prediction_error, relearned_signed_train_prediction_error = get_signed_prediction_error(
            model, relearned_model, X_train, Y_train, window_size
        )
        nominal_signed_val_prediction_error, original_signed_val_prediction_error, relearned_signed_val_prediction_error = get_signed_prediction_error(
            model, relearned_model, X_val, Y_val, window_size
        )

        if X_test is not None:
            nominal_signed_test_prediction_error, original_signed_test_prediction_error, relearned_signed_test_prediction_error = get_signed_prediction_error(
                model, relearned_model, X_test, Y_test, window_size
            )
        else:
            nominal_signed_test_prediction_error = None
            original_signed_test_prediction_error = None
            relearned_signed_test_prediction_error = None   

        plot_relearned_vs_original_prediction_error(
                window_size,
                original_train_loss,
                relearned_train_loss,
                original_each_component_train_loss,
                relearned_each_component_train_loss,
                nominal_signed_train_prediction_error,
                original_signed_train_prediction_error,
                relearned_signed_train_prediction_error,
                Y_train,
                Y_train_pred_origin,
                Y_train_pred_relearned,
                original_val_loss,
                relearned_val_loss,
                original_each_component_val_loss,
                relearned_each_component_val_loss,
                nominal_signed_val_prediction_error,
                original_signed_val_prediction_error,
                relearned_signed_val_prediction_error,
                Y_val,
                Y_val_pred_origin,
                Y_val_pred_relearned,
                original_test_loss,
                relearned_test_loss,
                original_each_component_test_loss,
                relearned_each_component_test_loss,
                nominal_signed_test_prediction_error,
                original_signed_test_prediction_error,
                relearned_signed_test_prediction_error,
                Y_test,
                Y_test_pred_origin,
                Y_test_pred_relearned,
                plt_save_dir
            )
        if save_path is not None:
            self.save_given_model(relearned_model, save_path)
        if relearned_val_loss < original_val_loss or always_update_model:
            return relearned_model, True
        else:
            return model, False

    def get_trained_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100]):
        print("state_component_predicted: ", state_component_predicted)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        self.model = error_prediction_NN.ErrorPredictionNN(
            prediction_length=prediction_length, state_component_predicted=state_component_predicted
        ).to(self.device)
        self.update_adaptive_weight(None,torch.tensor(X_train_np, dtype=torch.float32, device=self.device),torch.tensor(Y_train_np, dtype=torch.float32, device=self.device))
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
    def get_relearned_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100],reset_weight=False, randomize=0.001,plt_save_dir=None,save_path=None, use_replay_data=False, replay_data_rate=0.05, always_update_model=False):
        self.model.to(self.device)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        if len(self.X_test_list) > 0:
            X_test_np, Y_test_np, Z_test_np = get_sequence_data(self.X_test_list, self.Y_test_list, self.Z_test_list,self.division_indices_test)
            X_test = torch.tensor(X_test_np, dtype=torch.float32, device=self.device)
            Y_test = torch.tensor(Y_test_np, dtype=torch.float32, device=self.device)
            Z_test = torch.tensor(Z_test_np, dtype=torch.float32, device=self.device)
        else:
            X_test = None
            Y_test = None
            Z_test = None
        if use_replay_data and len(self.X_replay_list) > 0:
            X_replay_np, Y_replay_np, Z_replay_np = get_sequence_data(self.X_replay_list, self.Y_replay_list, self.Z_replay_list,self.division_indices_replay)
            X_replay = torch.tensor(X_replay_np, dtype=torch.float32, device=self.device)
            Y_replay = torch.tensor(Y_replay_np, dtype=torch.float32, device=self.device)
            Z_replay = torch.tensor(Z_replay_np, dtype=torch.float32, device=self.device)
        else:
            X_replay = None
            Y_replay = None
            Z_replay = None
            
        self.model, updated = self.relearn_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
            randomize=randomize,
            X_test=X_test,
            Y_test=Y_test,
            Z_test=Z_test,
            reset_weight=reset_weight,
            X_replay=X_replay,
            Y_replay=Y_replay,
            Z_replay=Z_replay,
            replay_data_rate=replay_data_rate,
            plt_save_dir=plt_save_dir,
            save_path=save_path,
            always_update_model=always_update_model
        )
        return updated
    def initialize_ensemble_models(self):
        self.models = [self.model]
    def get_updated_temp_model(self,learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100], use_replay_data=False, replay_data_rate=0.05,randomize_fix_lstm=0.0):
        self.temp_model = copy.deepcopy(self.model)
        X_train_np, Y_train_np, Z_train_np = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)    
        X_val_np, Y_val_np, Z_val_np = get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        if use_replay_data and len(self.X_replay_list) > 0:
            X_replay_np, Y_replay_np, Z_replay_np = get_sequence_data(self.X_replay_list, self.Y_replay_list, self.Z_replay_list,self.division_indices_replay)
            X_replay = torch.tensor(X_replay_np, dtype=torch.float32, device=self.device)
            Y_replay = torch.tensor(Y_replay_np, dtype=torch.float32, device=self.device)
            Z_replay = torch.tensor(Z_replay_np, dtype=torch.float32, device=self.device)
        else:
            X_replay = None
            Y_replay = None
            Z_replay = None
        self.temp_model.to(self.device)
        self.update_adaptive_weight(self.temp_model,torch.tensor(X_train_np, dtype=torch.float32, device=self.device),torch.tensor(Y_train_np, dtype=torch.float32, device=self.device))
        self.train_model(self.temp_model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
            fix_lstm=True,
            randomize_fix_lstm=randomize_fix_lstm,
            X_replay=X_replay,
            Y_replay=Y_replay,
            Z_replay=Z_replay,
            replay_data_rate=replay_data_rate
        )
    def relearn_temp_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100], randomize=0.001,plt_save_dir=None,save_path=None, use_replay_data=False, replay_data_rate=0.05,randomize_fix_lstm=0.0):
        self.temp_model.to(self.device)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        if len(self.X_test_list) > 0:
            X_test_np, Y_test_np, Z_test_np = get_sequence_data(self.X_test_list, self.Y_test_list, self.Z_test_list,self.division_indices_test)
            X_test = torch.tensor(X_test_np, dtype=torch.float32, device=self.device)
            Y_test = torch.tensor(Y_test_np, dtype=torch.float32, device=self.device)
            Z_test = torch.tensor(Z_test_np, dtype=torch.float32, device=self.device)
        else:
            X_test = None
            Y_test = None
            Z_test = None
        if use_replay_data and len(self.X_replay_list) > 0:
            X_replay_np, Y_replay_np, Z_replay_np = get_sequence_data(self.X_replay_list, self.Y_replay_list, self.Z_replay_list,self.division_indices_replay)
            X_replay = torch.tensor(X_replay_np, dtype=torch.float32, device=self.device)
            Y_replay = torch.tensor(Y_replay_np, dtype=torch.float32, device=self.device)
            Z_replay = torch.tensor(Z_replay_np, dtype=torch.float32, device=self.device)
        else:
            X_replay = None
            Y_replay = None
            Z_replay = None
            
            
        self.temp_model, updated = self.relearn_model(
            self.temp_model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
            randomize=randomize,
            X_test=X_test,
            Y_test=Y_test,
            Z_test=Z_test,
            X_replay=X_replay,
            Y_replay=Y_replay,
            Z_replay=Z_replay,
            replay_data_rate=replay_data_rate,
            plt_save_dir=plt_save_dir,
            save_path=save_path,
            fix_lstm=True,
            randomize_fix_lstm=randomize_fix_lstm
        )
        return updated
    def add_temp_model_to_ensemble(self):
        self.models.append(self.temp_model)
    def get_trained_ensemble_models(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100], ensemble_size=5):
        print("state_component_predicted: ", state_component_predicted)
        # Define Time Series Data
        X_train_np, Y_train_np, Z_train_np = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)    
        X_val_np, Y_val_np, Z_val_np = get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        self.model = error_prediction_NN.ErrorPredictionNN(
            prediction_length=prediction_length, state_component_predicted=state_component_predicted
        ).to(self.device)
        print("______________________________")
        print("ensemble number: ", 0)
        print("______________________________")
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )
        self.models = [self.model]
        for i in range(ensemble_size - 1):
            print("______________________________")
            print("ensemble number: ", i + 1)
            print("______________________________")
            temp_model = copy.deepcopy(self.model)
            self.train_model(temp_model,
                torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
                torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
                torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
                batch_sizes,
                learning_rates,
                patience,
                torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
                torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
                torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
                fix_lstm=True
            )
            self.models.append(temp_model)
    def update_saved_model(
        self, path, learning_rates=[1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100]
    ):
        X_train_np, Y_train_np, Z_train_np = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_val_np, Y_val_np, Z_val_np = get_sequence_data(self.X_val_list, self.Y_val_list, self.Z_val_list,self.division_indices_val)
        self.model = torch.load(path)
        self.model.to(self.device)
        self.train_model(
            self.model,
            torch.tensor(X_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_train_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_train_np, dtype=torch.float32, device=self.device),
            batch_sizes,
            learning_rates,
            patience,
            torch.tensor(X_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Y_val_np, dtype=torch.float32, device=self.device),
            torch.tensor(Z_val_np, dtype=torch.float32, device=self.device),
        )

    def save_model(self, path="vehicle_model.pth"):
        self.model.to("cpu")
        torch.save(self.model, path)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_model_to_csv(self.model, save_dir)
    def save_given_model(self, model, path="vehicle_model.pth"):
        model.to("cpu")
        torch.save(model, path)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_model_to_csv(model, save_dir)
    def save_ensemble_models(self, paths):
        for i in range(len(paths)):
            temp_model = self.models[i]
            temp_model.to("cpu")
            torch.save(temp_model, paths[i])
            save_dir = paths[i].replace(".pth", "")
            convert_model_to_csv.convert_model_to_csv(temp_model, save_dir)
    def fix_lstm(self,model,randomize=0.001,):
        # freeze the encoder layers
        for param in model.acc_encoder_layer_1.parameters():
            param.requires_grad = False
        for param in model.acc_encoder_layer_2.parameters():
            param.requires_grad = False
        for param in model.steer_encoder_layer_1.parameters():
            param.requires_grad = False
        for param in model.steer_encoder_layer_2.parameters():
            param.requires_grad = False
        for param in model.lstm_encoder.parameters():
            param.requires_grad = False
        # freeze shallow layers of the decoder
        for param in model.acc_layer_1.parameters():
            param.requires_grad = False
        for param in model.steer_layer_1.parameters():
            param.requires_grad = False
        for param in model.acc_layer_2.parameters():
            param.requires_grad = False
        for param in model.steer_layer_2.parameters():
            param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False
        
        #lb = -randomize
        #ub = randomize
        #nn.init.uniform_(model.complimentary_layer[0].weight, a=lb, b=ub)
        #nn.init.uniform_(model.complimentary_layer[0].bias, a=lb, b=ub)
        #nn.init.uniform_(model.linear_relu[0].weight, a=lb, b=ub)
        #nn.init.uniform_(model.linear_relu[0].bias, a=lb, b=ub)
        #nn.init.uniform_(model.final_layer.weight, a=lb, b=ub)
        #nn.init.uniform_(model.final_layer.bias, a=lb, b=ub)
        with torch.no_grad():
            model.complimentary_layer[0].weight += randomize * torch.randn_like(model.complimentary_layer[0].weight)
            model.complimentary_layer[0].bias += randomize * torch.randn_like(model.complimentary_layer[0].bias)
            model.linear_relu[0].weight += randomize * torch.randn_like(model.linear_relu[0].weight)
            model.linear_relu[0].bias += randomize * torch.randn_like(model.linear_relu[0].bias)
            model.final_layer.weight += randomize * torch.randn_like(model.final_layer.weight)
            model.final_layer.bias += randomize * torch.randn_like(model.final_layer.bias)
        

    def extract_features_from_data(self,X,division_indices,window_size=10):
        X_train_np, _, _ = get_sequence_data(X, self.Y_train_list, self.Z_train_list,division_indices)
        X_extracted = []
        for i in range(X_train_np.shape[0]):
            X_tmp = X_train_np[i][past_length:past_length + window_size]
            vel = X_tmp[:, 0].mean()
            acc = X_tmp[:, 1].mean()
            steer = X_tmp[:, 2].mean()
            acc_change = X_tmp[:, 1].max() - X_tmp[:, 1].min()
            steer_change = X_tmp[:, 2].max() - X_tmp[:, 2].min()
            X_extracted.append([vel, acc, steer, acc_change, steer_change])
        return X_extracted
    def extract_features_for_trining_data(self,window_size=10, vel_scale=0.1,acc_scale=1.0,steer_scale=1.5,acc_change_scale=5.0,steer_change_scale=5.0,bandwidth=0.3):
        self.X_extracted = np.array(self.extract_features_from_data(self.X_train_list,self.division_indices_train,window_size))
        self.scaling = [vel_scale, acc_scale, steer_scale, acc_change_scale, steer_change_scale]
        self.X_extracted_scaled = self.scaling * self.X_extracted
        self.kde_acc = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.X_extracted_scaled[:,[0,1,3]])
        self.kde_steer = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.X_extracted_scaled[:,[0,2,4]])
        self.kde_acc_steer = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.X_extracted_scaled)

    def plot_extracted_features(self,show_flag=True,save_dir=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_extracted[:,0],self.X_extracted[:,1],self.X_extracted[:,3])
        ax.set_xlabel('vel')
        ax.set_ylabel('acc')
        ax.set_zlabel('acc_change')
        ax.set_title('vel acc acc_change')
        if show_flag:
            plt.show()
        if save_dir:
            plt.savefig(save_dir + "/vel_acc_acc_change.png")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_extracted[:,0],self.X_extracted[:,2],self.X_extracted[:,4])
        ax.set_xlabel('vel')
        ax.set_ylabel('steer')
        ax.set_zlabel('steer_change')
        ax.set_title('vel steer steer_change')
        if show_flag:
            plt.show()
        if save_dir:
            plt.savefig(save_dir + "/vel_steer_steer_change.png")
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24,15), tight_layout=True)
        fig.suptitle("sequential plot")
        axes[0,0].plot(self.X_extracted[:,1],label="acc")
        axes[0,0].set_title("acc")
        axes[0,0].legend()
        axes[0,1].plot(self.X_extracted[:,3],label="acc_change")
        axes[0,1].set_title("acc_change")
        axes[0,1].legend()
        axes[1,0].plot(self.X_extracted[:,2],label="steer")
        axes[1,0].set_title("steer")
        axes[1,0].legend()
        axes[1,1].plot(self.X_extracted[:,4],label="steer_change")
        axes[1,1].set_title("steer_change")
        axes[1,1].legend()
        if show_flag:
            plt.show()
        if save_dir:
            plt.savefig(save_dir + "/extracted_data_sequential_plot.png")
    def calc_prediction_error_and_std(self, window_size=10, error_smoothing_window=100, show_density=False):
        if self.models is None:
            print("models are not trained")
            return
        X_test_np, Y_test_np, _ = get_sequence_data(self.X_test_list, self.Y_test_list, self.Z_test_list,self.division_indices_test)
        result_dict = {}
        nominal_prediction = []
        prediction = []
        for model in self.models:
            prediction_tmp, nominal_prediction = self.get_acc_steer_prediction(model,torch.tensor(X_test_np, dtype=torch.float32),torch.tensor(Y_test_np, dtype=torch.float32),window_size)
            
            prediction.append(prediction_tmp)
        result_dict["prediction_by_models"] = prediction
        mean_prediction = np.mean(prediction,axis=0)
        std_prediction = np.std(prediction,axis=0)
        result_dict["mean_prediction"] = mean_prediction
        result_dict["std_prediction"] = std_prediction
        result_dict["nominal_prediction"] = nominal_prediction
        result_dict["true_value"] = X_test_np[:,past_length+window_size,[1,2]]
        signed_prediction_error = X_test_np[:,past_length+window_size,[1,2]] - mean_prediction
        signed_nominal_prediction_error = X_test_np[:,past_length+window_size,[1,2]] - nominal_prediction
        if show_density:
            X_test_extracted = np.array(self.extract_features_from_data(self.X_test_list,self.division_indices_test,window_size))
            X_test_extracted_scaled = self.scaling * X_test_extracted
            prob_acc = np.exp(self.kde_acc.score_samples(X_test_extracted_scaled[:,[0,1,3]]))
            prob_steer = np.exp(self.kde_steer.score_samples(X_test_extracted_scaled[:,[0,2,4]]))
            prob_acc_steer = np.exp(self.kde_acc_steer.score_samples(X_test_extracted_scaled))
            w = np.ones(error_smoothing_window) / error_smoothing_window
            residual_acc_error_ratio = np.convolve(np.abs(signed_prediction_error[:,0]),w,mode="same") / (np.convolve(np.abs(signed_nominal_prediction_error[:,0]),w,mode="same") + 1e-3)
            residual_steer_error_ratio = np.convolve(np.abs(signed_prediction_error[:,1]),w,mode="same") / (np.convolve(np.abs(signed_nominal_prediction_error[:,1]),w,mode="same") + 1e-3)
            residual_acc_error_ratio = np.clip(residual_acc_error_ratio,0.0,1.0)
            residual_steer_error_ratio = np.clip(residual_steer_error_ratio,0.0,1.0)
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24,15), tight_layout=True)
            fig.suptitle("acc steer prediction error and std")
            axes[0,0].plot(signed_prediction_error[:,0],label="trained")
            axes[0,0].plot(signed_nominal_prediction_error[:,0],label="nominal")
            axes[0,0].set_title("acc error")
            axes[0,0].legend()
            axes[0,1].plot(std_prediction[:,0],label="acc")
            axes[0,1].set_title("acc std")
            axes[0,1].legend()
            axes[1,0].plot(signed_prediction_error[:,1],label="trained")
            axes[1,0].plot(signed_nominal_prediction_error[:,1],label="nominal")
            axes[1,0].set_title("steer error")
            axes[1,0].legend()
            axes[1,1].plot(std_prediction[:,1],label="steer")
            axes[1,1].set_title("steer std")
            axes[1,1].legend()
            axes[0,2].plot(prob_acc,label="acc_prob")
            axes[0,2].set_title("acc prob")
            axes[0,2].legend()
            axes[0,3].plot(prob_acc*prob_steer,label="acc*steer")
            axes[0,3].set_title("acc*steer prob")
            axes[0,3].legend()
            axes[1,2].plot(prob_steer,label="steer_prob")
            axes[1,2].set_title("steer prob")
            axes[1,2].legend()
            axes[1,3].plot(prob_acc_steer,label="acc_steer")
            axes[1,3].set_title("acc_steer prob")
            plt.show()
            #plt.plot(np.abs(signed_prediction_error)[:,0]/np.abs(signed_prediction_error[:,0]).max(),label="abs_trained_error/" + str(np.abs(signed_prediction_error[:,0]).max()))
            plt.plot(residual_acc_error_ratio,label="residual_acc_error_ratio")
            plt.plot(std_prediction[:,0]/std_prediction[:,0].max(),label="acc_std/"+str(std_prediction[:,0].max()))
            plt.plot(prob_acc.min()/prob_acc,label=str(prob_acc.min()) + "/acc_prob")
            plt.title("acc comparison")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,1],label="acc")
            plt.title("acc")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,3],label="acc_change")
            plt.title("acc_change")
            plt.legend()
            plt.show()
            #plt.plot(np.abs(signed_prediction_error)[:,1]/np.abs(signed_prediction_error[:,1]).max(),label="abs_trained_error/" + str(np.abs(signed_prediction_error[:,1]).max()))
            plt.plot(residual_steer_error_ratio,label="residual_steer_error_ratio")
            plt.plot(std_prediction[:,1]/std_prediction[:,1].max(),label="steer_std/"+str(std_prediction[:,1].max()))
            plt.plot(prob_steer.min()/prob_steer,label=str(prob_steer.min())+"/steer_prob")
            plt.title("steer comparison")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,2],label="steer")
            plt.title("steer")
            plt.legend()
            plt.show()
            plt.plot(X_test_extracted[:,4],label="steer_change")
            plt.title("steer_change")
            plt.legend()
            plt.show()
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24,15), tight_layout=True)
            fig.suptitle("acc steer prediction error and std")
            axes[0,0].plot(signed_prediction_error[:,0],label="trained")
            axes[0,0].plot(signed_nominal_prediction_error[:,0],label="nominal")
            axes[0,0].set_title("acc error")
            axes[0,0].legend()
            axes[0,1].plot(std_prediction[:,0],label="acc")
            axes[0,1].set_title("acc std")
            axes[0,1].legend()
            axes[1,0].plot(signed_prediction_error[:,1],label="trained")
            axes[1,0].plot(signed_nominal_prediction_error[:,1],label="nominal")
            axes[1,0].set_title("steer error")
            axes[1,0].legend()
            axes[1,1].plot(std_prediction[:,1],label="steer")
            axes[1,1].set_title("steer std")
            axes[1,1].legend()
        return result_dict
        

    def get_acc_steer_prediction(self,model,X,Y,window_size=10,batch_size=3000):
        model.eval()
        model.to("cpu")
        num_batches = (X.size(0) + batch_size - 1) // batch_size
        prediction = []
        nominal_prediction = []
        for k in range(num_batches):
            _, hc = model(X[k*batch_size:(k+1)*batch_size], previous_error=Y[k*batch_size:(k+1)*batch_size, :past_length, -2:], mode="get_lstm_states")
            states_tmp=X[k*batch_size:(k+1)*batch_size, past_length, [vel_index, acc_index, steer_index]].unsqueeze(1)
            nominal_states_tmp = states_tmp.clone()
            for i in range(window_size):
                states_tmp_with_input_history = torch.cat((states_tmp, X[k*batch_size:(k+1)*batch_size,[past_length+i], 3:]), dim=2)
                predicted_error, hc =  model(states_tmp_with_input_history, hc=hc, mode="predict_with_hc")
                states_tmp[:,0,0] = X[k*batch_size:(k+1)*batch_size,past_length + i + 1, 0]
                nominal_states_tmp[:,0,0] = X[k*batch_size:(k+1)*batch_size,past_length + i + 1, 0]
                for j in range(prediction_step):
                    states_tmp[:,0,1] = states_tmp[:,0,1] + (X[k*batch_size:(k+1)*batch_size,past_length + i, acc_input_indices_nom[j]] - states_tmp[:,0,1]) * (1 - np.exp(- control_dt / acc_time_constant))
                    states_tmp[:,0,2] = states_tmp[:,0,2] + (X[k*batch_size:(k+1)*batch_size,past_length + i, steer_input_indices_nom[j]] - states_tmp[:,0,2]) * (1 - np.exp(- control_dt / steer_time_constant))
                    nominal_states_tmp[:,0,1] = nominal_states_tmp[:,0,1] + (X[k*batch_size:(k+1)*batch_size,past_length + i, acc_input_indices_nom[j]] - nominal_states_tmp[:,0,1]) * (1 - np.exp(- control_dt / acc_time_constant))
                    nominal_states_tmp[:,0,2] = nominal_states_tmp[:,0,2] + (X[k*batch_size:(k+1)*batch_size,past_length + i, steer_input_indices_nom[j]] - nominal_states_tmp[:,0,2]) * (1 - np.exp(- control_dt / steer_time_constant))
                states_tmp[:,0,1] = states_tmp[:,0,1] + predicted_error[:,0,state_name_to_predicted_index["acc"]] * control_dt * prediction_step
                states_tmp[:,0,2] = states_tmp[:,0,2] + predicted_error[:,0,state_name_to_predicted_index["steer"]] * control_dt * prediction_step
                

            prediction.append(states_tmp[:,0,[1,2]].detach().numpy())
            nominal_prediction.append(nominal_states_tmp[:,0,[1,2]].detach().numpy())
        prediction = np.concatenate(prediction,axis=0)
        nominal_prediction = np.concatenate(nominal_prediction,axis=0)
        return prediction, nominal_prediction
    def calc_dataloader_weights(self, window_size=10, vel_scale=0.1,acc_scale=1.0,steer_scale=1.5,acc_change_scale=5.0,steer_change_scale=5.0,bandwidth=0.3, maximum_weight_by_acc_density=3.0,maximum_weight_by_steer_density=3.0, maximum_weight_by_small_steer=3.0, maximum_weight_by_small_steer_change=10.0,small_steer_threshold=0.01,small_steer_change_threshold=0.01,steer_bins=[0.01,0.1]):
        if len(self.X_train_list) == 0:
            print("no data")
            return
        if len(self.X_val_list) == 0:
            print("no validation data")
            return
        self.extract_features_for_trining_data(window_size, vel_scale,acc_scale,steer_scale,acc_change_scale,steer_change_scale,bandwidth)
        weights = []
        acc_density = np.exp(self.kde_acc.score_samples(self.X_extracted_scaled[:,[0,1,3]]))
        steer_density = np.exp(self.kde_steer.score_samples(self.X_extracted_scaled[:,[0,2,4]]))
        for i in range(self.X_extracted_scaled.shape[0]):
            weight_by_acc_density = 1.0 / (acc_density[i] + 1.0/maximum_weight_by_acc_density)
            weight_by_steer_density = 1.0 / (steer_density[i] + 1.0/maximum_weight_by_steer_density)
            weight_by_small_steer = (1.0 + 1.0/maximum_weight_by_small_steer) / (np.abs(self.X_extracted[i,4]) / small_steer_threshold + 1/maximum_weight_by_small_steer)
            #weight_by_small_steer_change = (1.0 + 1.0/maximum_weight_by_small_steer_change) / (np.abs(self.X_extracted[i,3]) / small_steer_change_threshold + 1/maximum_weight_by_small_steer_change)
            #weight_by_small_steer_and_change = max(min(weight_by_small_steer,weight_by_small_steer_change),1.0)
            weights.append(weight_by_acc_density * weight_by_steer_density * weight_by_small_steer)# * weight_by_small_steer_and_change)
        self.weights_for_dataloader = np.array(weights)
        return
        steer_positive_indices = []
        steer_negative_indices = []
        steer_positive_indices.append(np.where((self.X_extracted[:,2] > 0.0) & (self.X_extracted[:,2] < steer_bins[0]))[0])
        steer_negative_indices.append(np.where((self.X_extracted[:,2] < 0.0) & (self.X_extracted[:,2] > -steer_bins[0]))[0])
        for i in range(len(steer_bins)-1):
            steer_positive_indices.append(np.where((self.X_extracted[:,2] > steer_bins[i]) & (self.X_extracted[:,2] < steer_bins[i+1]))[0])
            steer_negative_indices.append(np.where((self.X_extracted[:,2] < -steer_bins[i]) & (self.X_extracted[:,2] > -steer_bins[i+1]))[0])
        steer_positive_indices.append(np.where(self.X_extracted[:,2] > steer_bins[-1])[0])
        steer_negative_indices.append(np.where(self.X_extracted[:,2] < -steer_bins[-1])[0])
        self.weights_for_dataloader = np.array(weights)
        for i in range(len(steer_positive_indices)):
            positive_ind = steer_positive_indices[i]
            negative_ind = steer_negative_indices[i]
            if len(positive_ind) == 0 or len(negative_ind) == 0:
                continue
            positive_weight_sum = np.sum(self.weights_for_dataloader[positive_ind])
            negative_weight_sum = np.sum(self.weights_for_dataloader[negative_ind])
            positive_weight_coef = 0.5  + 0.5 * (negative_weight_sum + 1e+3) / (positive_weight_sum + 1e-3)
            negative_weight_coef = 0.5  + 0.5 * (positive_weight_sum + 1e+3) / (negative_weight_sum + 1e-3)
            self.weights_for_dataloader[positive_ind] *= positive_weight_coef
            self.weights_for_dataloader[negative_ind] *= negative_weight_coef

    """
    def get_linear_regression_matrices(self, save_dir, batch_size=10000,mode="single_model"):
        X_train_np = transform_to_sequence_data(
            np.array(self.X_train_list)[:, 3:],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        Y_train_np = transform_to_sequence_data(
            np.array(self.Y_train_list)[:, state_component_predicted_index],
            past_length + prediction_length,
            self.division_indices_train,
            prediction_step,
        )
        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32, device=self.device)
        num_batches = (X_train.size(0) + batch_size - 1) // batch_size
        x_dim = 3
        feature_size = 1 + x_dim * parameters.x_history_len_for_linear_compensation + acc_queue_size + steer_queue_size + 2*prediction_step
        if mode == "single_model":
            h_dim = self.model.lstm_hidden_size
        elif mode == "ensemble":
            h_dim = self.models[0].lstm_hidden_size
        else:
            print("mode is not correct")
            return
        feature_size += 2 * h_dim
        if parameters.fit_yaw_for_linear_compensation:
            target_size = 3
        else:
            target_size = 2
        XXT = np.zeros((feature_size,feature_size))
        YXT = np.zeros((target_size,feature_size))
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_train.size(0))
                if mode == "single_model":
                    self.model.to(self.device)
                    _, hc = self.model(X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :past_length, -2:], mode="get_lstm_states")
                elif mode == "ensemble":
                    for i in range(len(self.models)):
                        self.models[i].to(self.device)
                    _, hc = self.models[0](X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :past_length, -2:], mode="get_lstm_states")
                for j in range(parameters.compensation_lstm_len):
                    x_input = np.zeros((end_idx - start_idx, feature_size))
                    x_input[:,0] = 1.0
                    x_input[:,-2*h_dim:-h_dim] = hc[0][0].to("cpu").detach().numpy()
                    x_input[:,-h_dim:] = hc[1][0].to("cpu").detach().numpy()
                    x_input[:, 1 + x_dim * parameters.x_history_len_for_linear_compensation:- 2 * h_dim] = (
                        X_train_np[start_idx:end_idx,past_length + j, 3:]
                    )
                    for k in range(parameters.x_history_len_for_linear_compensation):
                        x_input[:,1 + x_dim * k] = parameters.vel_scale_for_linear_compensation * X_train_np[start_idx:end_idx,past_length + j - parameters.x_history_len_for_linear_compensation + k, 0]
                        x_input[:,2 + x_dim * k] = X_train_np[start_idx:end_idx,past_length + j - parameters.x_history_len_for_linear_compensation + k, 1]
                        x_input[:,3 + x_dim * k] = X_train_np[start_idx:end_idx,past_length + j - parameters.x_history_len_for_linear_compensation + k, 2]
                    XXT += (x_input.T @ x_input) / ((end_idx - start_idx) * parameters.compensation_lstm_len)
                    if mode == "single_model":
                        Y_pred, hc = self.model(X_train[start_idx:end_idx, [past_length + j]], hc=hc, mode="predict_with_hc")
                    elif mode == "ensemble":
                        Y_pred, hc_update = self.models[0](X_train[start_idx:end_idx, [past_length + j]], hc=hc, mode="predict_with_hc")
                        for k in range(1,len(self.models)):
                            Y_pred_tmp, _ = self.models[k](X_train[start_idx:end_idx, [past_length + j]], hc=hc, mode="predict_with_hc")
                            Y_pred += Y_pred_tmp
                        Y_pred /= len(self.models)
                        hc = hc_update
                    prediction_error = Y_train[start_idx:end_idx, past_length + j] - Y_pred.squeeze()
                    prediction_error_np = prediction_error.to("cpu").detach().numpy()
                    if parameters.fit_yaw_for_linear_compensation:
                        YXT += prediction_error_np[:,-3:].T @ x_input / ((end_idx - start_idx) * parameters.compensation_lstm_len)
                    else:
                        YXT += prediction_error_np[:,-2:].T @ x_input / ((end_idx - start_idx) * parameters.compensation_lstm_len)
        XXT = XXT/num_batches
        YXT = YXT/num_batches
        np.savetxt(save_dir + "/XXT.csv",XXT,delimiter=",")
        np.savetxt(save_dir + "/YXT.csv",YXT,delimiter=",")

    """
    def get_linear_regression_matrices(self,save_dir, batch_size=3000,mode="single_model"):
        speed_threshold = 6.0
        steer_threshold = 0.05
        acc_threshold = 0.1
        lasso_alpha=1e-4
        max_cumulative_error = 0.1
        max_projection_dim = 10
        X_train_np, Y_train_np, _ = get_sequence_data(self.X_train_list, self.Y_train_list, self.Z_train_list,self.division_indices_train)
        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=self.device)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32, device=self.device)
        num_batches = (X_train.size(0) + batch_size - 1) // batch_size
        x_dim = 3
        feature_size = 1 + x_dim * parameters.x_history_len_for_linear_compensation + self.acc_queue_size + self.steer_queue_size + 2*self.prediction_step
        if mode == "single_model":
            h_dim = self.model.lstm_hidden_size
        elif mode == "ensemble":
            h_dim = self.models[0].lstm_hidden_size
        else:
            print("mode is not correct")
            return
        feature_size += 2 * h_dim
        if parameters.fit_yaw_for_linear_compensation:
            target_size = 3
        else:
            target_size = 2
        
        speed_names = ["high_speed","low_speed"]
        steer_names = ["left","right","straight"]
        acc_names = ["accelerate","decelerate","constant"]
        X_dict = {}
        Y_dict = {}
        linear_models = {}
        for speed_name in speed_names:
            for steer_name in steer_names:
                for acc_name in acc_names:
                    X_dict[speed_name + "_" + steer_name + "_" + acc_name] = []
                    Y_dict[speed_name + "_" + steer_name + "_" + acc_name] = []
                    linear_models[speed_name + "_" + steer_name + "_" + acc_name] = Lasso(alpha=lasso_alpha)
        X_train_np = X_train.to("cpu").detach().numpy()

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_train.size(0))
                if mode == "single_model":
                    self.model.to(self.device)
                    _, hc = self.model(X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :self.past_length, -2:], mode="get_lstm_states")
                elif mode == "ensemble":
                    for i in range(len(self.models)):
                        self.models[i].to(self.device)
                    _, hc = self.models[0](X_train[start_idx:end_idx], previous_error=Y_train[start_idx:end_idx, :self.past_length, -2:], mode="get_lstm_states")
                for j in range(parameters.compensation_lstm_len):
                    x_input = np.zeros((end_idx - start_idx, feature_size-1))
                    x_input[:,-2*h_dim:-h_dim] = hc[0][0].to("cpu").detach().numpy()
                    x_input[:,-h_dim:] = hc[1][0].to("cpu").detach().numpy()
                    x_input[:, x_dim * parameters.x_history_len_for_linear_compensation:- 2 * h_dim] = (
                        X_train_np[start_idx:end_idx,self.past_length + j, 3:]
                    )
                    for k in range(parameters.x_history_len_for_linear_compensation):
                        x_input[:,x_dim * k] = parameters.vel_scale_for_linear_compensation * X_train_np[start_idx:end_idx,self.past_length + j - parameters.x_history_len_for_linear_compensation + k, 0]
                        x_input[:,x_dim * k] = X_train_np[start_idx:end_idx,self.past_length + j - parameters.x_history_len_for_linear_compensation + k, 1]
                        x_input[:,x_dim * k] = X_train_np[start_idx:end_idx,self.past_length + j - parameters.x_history_len_for_linear_compensation + k, 2]
                    if mode == "single_model":
                        Y_pred, hc = self.model(X_train[start_idx:end_idx, [self.past_length + j]], hc=hc, mode="predict_with_hc")
                    elif mode == "ensemble":
                        Y_pred, hc_update = self.models[0](X_train[start_idx:end_idx, [self.past_length + j]], hc=hc, mode="predict_with_hc")
                        for k in range(1,len(self.models)):
                            Y_pred_tmp, _ = self.models[k](X_train[start_idx:end_idx, [self.past_length + j]], hc=hc, mode="predict_with_hc")
                            Y_pred += Y_pred_tmp
                        Y_pred /= len(self.models)
                        hc = hc_update
                    prediction_error = Y_train[start_idx:end_idx, self.past_length + j] - Y_pred.squeeze()
                    prediction_error_np = prediction_error.to("cpu").detach().numpy()
                    if parameters.fit_yaw_for_linear_compensation:
                        prediction_error_np = prediction_error_np[:, -3:]
                    else:
                        prediction_error_np = prediction_error_np[:, -2:]
                    for k in range(end_idx - start_idx):
                        speed_idx = 0 if X_train_np[start_idx + k, self.past_length + j, 0] > speed_threshold else 1
                        steer_idx = 0 if X_train_np[start_idx + k, self.past_length + j, 1] < -steer_threshold else 1 if X_train_np[start_idx + k, self.past_length + j, 1] > steer_threshold else 2
                        acc_idx = 0 if X_train_np[start_idx + k, self.past_length + j, 2] > acc_threshold else 1 if X_train_np[start_idx + k, self.past_length + j, 2] < -acc_threshold else 2
                        X_dict[speed_names[speed_idx] + "_" + steer_names[steer_idx] + "_" + acc_names[acc_idx]].append(x_input[k])
                        Y_dict[speed_names[speed_idx] + "_" + steer_names[steer_idx] + "_" + acc_names[acc_idx]].append(prediction_error_np[k])
        coef_matrix_stacked = []
        for speed_name in speed_names:
            for steer_name in steer_names:
                for acc_name in acc_names:
                    X_dict[speed_name + "_" + steer_name + "_" + acc_name] = np.array(X_dict[speed_name + "_" + steer_name + "_" + acc_name])
                    Y_dict[speed_name + "_" + steer_name + "_" + acc_name] = np.array(Y_dict[speed_name + "_" + steer_name + "_" + acc_name])
                    error_std = np.std(Y_dict[speed_name + "_" + steer_name + "_" + acc_name], axis=0)
                    linear_models[speed_name + "_" + steer_name + "_" + acc_name].fit(X_dict[speed_name + "_" + steer_name + "_" + acc_name], Y_dict[speed_name + "_" + steer_name + "_" + acc_name])
                    coef_matrix_stacked.append((linear_models[speed_name + "_" + steer_name + "_" + acc_name].coef_.T / (error_std + 1e-9)).T)
        coef_matrix_stacked = np.vstack(coef_matrix_stacked)
        U, S, VT = np.linalg.svd(coef_matrix_stacked)

        cumulative_sum = np.cumsum(S)
        cumulative_ratio = cumulative_sum / cumulative_sum[-1]
        print("cumulative_ratio:", cumulative_ratio)
        projection_dim = max(2,min(np.where(cumulative_ratio > 1 - max_cumulative_error)[0][0] + 1, max_projection_dim))
        P = VT[:projection_dim]
        np.savetxt(save_dir + "/Projection.csv", P, delimiter=",")
    def update_adaptive_weight(self,model, X, Y, batch_size=3000):
        if model is not None:
            model.to(self.device)
            model.eval()
        num_batches = (X.size(0) + batch_size - 1) // batch_size
        prediction_error = torch.zeros(self.adaptive_weight.shape[0], device=self.device)
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.size(0))
                
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]
                if model is not None:
                    Y_pred, _ = model(X_batch, previous_error=Y_batch[:, :past_length, -2:], mode="get_lstm_states")
                    prediction_error += torch.mean(torch.abs(Y_pred - Y_batch[:,past_length:]),dim=(0,1)) * (end_idx - start_idx)
                else:
                    prediction_error += torch.mean(torch.abs(Y_batch[:,past_length:]),dim=(0,1)) * (end_idx - start_idx)
            prediction_error /= X.size(0)
        print("prediction_error:", prediction_error)
        self.adaptive_weight = 1.0 / (prediction_error + 1e-4)
        for i in range(len(self.adaptive_weight)):
            if self.adaptive_weight[i] > torch.max(self.adaptive_weight[-2:]):
                self.adaptive_weight[i] = torch.max(self.adaptive_weight[-2:]) # acc and steer are respected
        self.adaptive_weight = self.adaptive_weight / torch.mean(self.adaptive_weight)
