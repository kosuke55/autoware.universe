from autoware_vehicle_adaptor.training import add_data_from_csv
from autoware_vehicle_adaptor.training import error_prediction_NN
from autoware_vehicle_adaptor.training import get_initial_hidden_NN
from autoware_vehicle_adaptor.training import convert_model_to_csv
from autoware_vehicle_adaptor.training.training_utils import plot_relearned_vs_original_prediction_error, get_random_memory_bank
from autoware_vehicle_adaptor.training.training_utils import TrainErrorPredictionNNWithMemoryFunctions
from autoware_vehicle_adaptor.training.early_stopping import EarlyStopping
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
from multiprocessing import Process
import tempfile

get_loss = TrainErrorPredictionNNWithMemoryFunctions.get_loss
validate_in_batches = TrainErrorPredictionNNWithMemoryFunctions.validate_in_batches
get_each_component_loss = TrainErrorPredictionNNWithMemoryFunctions.get_each_component_loss
get_losses = TrainErrorPredictionNNWithMemoryFunctions.get_losses
get_signed_prediction_error = TrainErrorPredictionNNWithMemoryFunctions.get_signed_prediction_error
get_sequence_data_tensor = TrainErrorPredictionNNWithMemoryFunctions.get_sequence_data_tensor


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

prob_update_memory_bank = 0.03
memory_bank_size = 100
memory_bank_element_len = 10


class train_error_prediction_NN_with_memory_bank(add_data_from_csv.add_data_from_csv):
    """Class for training the error prediction NN."""

    def __init__(
            self,
            max_iter=10000,
            tol=1e-5,
            alpha_1=0.1**7,
            alpha_2=0.1**7,
            alpha_jacobian=0.1**4,
            alpha_jacobian_encoder=0.1**5,
            alpha_jacobian_gru_attention=0.1**4,
            alpha_hessian=0.1**14,
            alpha_hessian_encoder=0.1**14,
            alpha_hessian_gru_attention=0.1**14
        ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.alpha_jacobian = alpha_jacobian
        self.alpha_jacobian_encoder = alpha_jacobian_encoder
        self.alpha_jacobian_gru_attention = alpha_jacobian_gru_attention
        self.alpha_hessian = alpha_hessian
        self.alpha_hessian_encoder = alpha_hessian_encoder
        self.alpha_hessian_gru_attention = alpha_hessian_gru_attention

        self.models = None
        self.model_for_initial_hidden = None

        self.past_length = past_length
        self.prediction_length = prediction_length
        self.prediction_step = prediction_step
        self.acc_queue_size = acc_queue_size
        self.steer_queue_size = steer_queue_size
        self.adaptive_weight = torch.ones(len(state_component_predicted_index)).to(self.device)
    def train_model(
        self,
        model,
        model_for_initial_hidden,
        X_train,
        Y_train,
        memory_bank_train,
        batch_sizes: list,
        learning_rates: list,
        patience: int,
        X_val,
        Y_val,
        memory_bank_val,
        freeze_shallow_layers: bool = False,
        randomize_remaining_layers: float = 0.001,
        integration_prob: float = 0.1,
        X_replay=None,
        Y_replay=None,
        memory_bank_replay=None,
        replay_data_rate=0.05
    ):
        # Initialize the model
        model = model.to(self.device)
        model_for_initial_hidden = model_for_initial_hidden.to(self.device)

        print("device:", self.device)
        print("total_sample size:", X_train.size(0) + X_val.size(0))
        print("train_sample size:", X_train.size(0))
        criterion = nn.L1Loss()
        # freeze the shallow layers
        if freeze_shallow_layers:
            self.freeze_shallow_layers(model,model_for_initial_hidden,randomize=randomize_remaining_layers)
        # save the original adaptive weight
        original_adaptive_weight = self.adaptive_weight.clone()
        print("original_adaptive_weight: ", original_adaptive_weight)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_for_initial_hidden.parameters()), lr=learning_rates[0])
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0])
        
        # Initial loss
        initial_loss = validate_in_batches(model, model_for_initial_hidden, criterion, X_val, Y_val, memory_bank_val, adaptive_weight=self.adaptive_weight)
        print("Initial loss:", initial_loss)

        batch_size = batch_sizes[0]
        print("batch_size:", batch_size)
        # Initialize the early stopping
        early_stopping = EarlyStopping(initial_loss, tol=self.tol, patience=patience)
        #Data loader
        train_dataset = TensorDataset(X_train, Y_train, memory_bank_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        # learning_rate index
        learning_rate_index = 0
        print("learning rate:", learning_rates[learning_rate_index])

        # Train the model
        for i in range(self.max_iter):
            model.train()
            model_for_initial_hidden.train()
            for X_batch, Y_batch, memory_bank_batch in train_loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)
                memory_bank_batch = memory_bank_batch.to(self.device)
                optimizer.zero_grad()
                loss = get_loss(
                    criterion, model, model_for_initial_hidden,
                    X_batch, Y_batch, memory_bank_batch, self.adaptive_weight,
                    integral_prob=integration_prob,
                    alpha_jacobian=self.alpha_jacobian,alpha_jacobian_encoder=self.alpha_jacobian_encoder,
                    alpha_jacobian_gru_attention=self.alpha_jacobian_gru_attention,
                    alpha_hessian=self.alpha_hessian,alpha_hessian_encoder=self.alpha_hessian_encoder,
                    alpha_hessian_gru_attention=self.alpha_hessian_gru_attention)
                for w in model.parameters():
                    loss += self.alpha_1 * torch.norm(w, 1) + self.alpha_2 * torch.norm(w, 2) ** 2
                for w in model_for_initial_hidden.parameters():
                    loss += self.alpha_1 * torch.norm(w, 1) + self.alpha_2 * torch.norm(w, 2) ** 2
                if X_replay is not None:
                    replay_indices = torch.randperm(X_replay.size(0))[:int(replay_data_rate * batch_size)]
                    X_replay_batch = X_replay[replay_indices].pin_memory().to(self.device)
                    Y_replay_batch = Y_replay[replay_indices].pin_memory().to(self.device)
                    memory_bank_replay_batch = memory_bank_replay[replay_indices].pin_memory().to(self.device)
                    loss += get_loss(
                        criterion, model, model_for_initial_hidden,
                        X_replay_batch, Y_replay_batch, memory_bank_replay_batch, self.adaptive_weight,
                        integral_prob=integration_prob,
                        alpha_jacobian=self.alpha_jacobian,alpha_jacobian_encoder=self.alpha_jacobian_encoder,
                        alpha_jacobian_gru_attention=self.alpha_jacobian_gru_attention,
                        alpha_hessian=self.alpha_hessian,alpha_hessian_encoder=self.alpha_hessian_encoder,
                        alpha_hessian_gru_attention=self.alpha_hessian_gru_attention)

                loss.backward()
                optimizer.step()
            # Validate the model
            model.eval()
            val_loss = validate_in_batches(
                model, model_for_initial_hidden, criterion, X_val, Y_val, memory_bank_val,
                adaptive_weight=self.adaptive_weight)
            val_loss_with_original_weight = validate_in_batches(
                model, model_for_initial_hidden, criterion, X_val, Y_val, memory_bank_val,
                adaptive_weight=original_adaptive_weight)
            if i % 10 == 1:
                print("epoch:", i)
                print("val_loss_with_original_weight:", val_loss_with_original_weight)
                print("val_loss:", val_loss)
            if early_stopping(val_loss):
                learning_rate_index += 1
                if learning_rate_index >= len(learning_rates):
                    break
                batch_size = batch_sizes[min(learning_rate_index, len(batch_sizes) - 1)]
                optimizer = torch.optim.Adam(list(model.parameters()) + list(model_for_initial_hidden.parameters()), lr=learning_rates[learning_rate_index])
                #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[learning_rate_index])
                
                print("update learning rate to ", learning_rates[learning_rate_index])
                print("batch_size:", batch_size)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                early_stopping.reset()
                if learning_rates[learning_rate_index - 1] < 3e-4:
                    self.update_adaptive_weight(model, model_for_initial_hidden, X_train, Y_train, memory_bank_train)
                print("adaptive_weight: ", self.adaptive_weight)
                
    def relearn_model(
        self,
        model,
        model_for_initial_hidden,
        X_train,
        Y_train,
        memory_bank_train,
        batch_sizes: list,
        learning_rates: list,
        patience: int,
        X_val,
        Y_val,
        memory_bank_val,
        freeze_shallow_layers: bool = False,
        integration_prob: float = 0.1,
        randomize=0.001,
        reset_weight=False,
        X_test=None,
        Y_test=None,
        memory_bank_test=None,
        X_replay=None,
        Y_replay=None,
        memory_bank_replay=None,
        replay_data_rate=0.05,
        plt_save_dir=None,
        window_size=10,
        model_save_path=None,
        always_update_model=False
    ):
        print("randomize:", randomize)
        self.update_adaptive_weight(model, model_for_initial_hidden, X_train, Y_train, memory_bank_train)
        original_adaptive_weight = self.adaptive_weight.clone()
        criterion = nn.L1Loss()
        original_train_loss, original_each_component_train_loss, Y_train_pred_origin = get_losses(
            model, model_for_initial_hidden, criterion, X_train, Y_train, memory_bank_train, adaptive_weight=original_adaptive_weight)
        original_val_loss, original_each_component_val_loss, Y_val_pred_origin = get_losses(
            model, model_for_initial_hidden, criterion, X_val, Y_val, memory_bank_val, adaptive_weight=original_adaptive_weight)
        if X_test is not None:
            original_test_loss, original_each_component_test_loss, Y_test_pred_origin = get_losses(
                model, model_for_initial_hidden, criterion, X_test, Y_test, memory_bank_test, adaptive_weight=original_adaptive_weight)
        else:
            original_test_loss = None
            original_each_component_test_loss = None
            Y_test_pred_origin = None
        if reset_weight:
            relearned_model = error_prediction_NN.ErrorPredictionNN(
                prediction_length=prediction_length, state_component_predicted=state_component_predicted
            )
            relearned_model_for_initial_hidden = get_initial_hidden_NN.GetInitialHiddenNN(
                output_size=2*relearned_model.lstm_hidden_total_size
            )
        else:
            relearned_model = copy.deepcopy(model)
            relearned_model.lstm_encoder.flatten_parameters()
            relearned_model.lstm.flatten_parameters()
            relearned_model_for_initial_hidden = copy.deepcopy(model_for_initial_hidden)
            with torch.no_grad():
                if not freeze_shallow_layers:
                    for w in relearned_model.parameters():
                        w += randomize * torch.randn_like(w)
                    for w in relearned_model_for_initial_hidden.parameters():
                        w += randomize * torch.randn_like(w)
        self.train_model(
            relearned_model,
            relearned_model_for_initial_hidden,
            X_train,
            Y_train,
            memory_bank_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            memory_bank_val,
            freeze_shallow_layers=freeze_shallow_layers,
            randomize_remaining_layers=randomize,
            integration_prob=integration_prob,
            X_replay=X_replay,
            Y_replay=Y_replay,
            memory_bank_replay=memory_bank_replay,
            replay_data_rate=replay_data_rate
        )
        relearned_train_loss, relearned_each_component_train_loss, Y_train_pred_relearned = get_losses(
            relearned_model, relearned_model_for_initial_hidden, criterion, X_train, Y_train, memory_bank_train, adaptive_weight=original_adaptive_weight)
        relearned_val_loss, relearned_each_component_val_loss, Y_val_pred_relearned = get_losses(
            relearned_model, relearned_model_for_initial_hidden, criterion, X_val, Y_val, memory_bank_val, adaptive_weight=original_adaptive_weight)
        if X_test is not None:
            relearned_test_loss, relearned_each_component_test_loss, Y_test_pred_relearned = get_losses(
                relearned_model, relearned_model_for_initial_hidden, criterion, X_test, Y_test, memory_bank_test, adaptive_weight=original_adaptive_weight)
        else:
            relearned_test_loss = None
            relearned_each_component_test_loss = None
            Y_test_pred_relearned = None

        nominal_signed_train_prediction_error, original_signed_train_prediction_error, relearned_signed_train_prediction_error = get_signed_prediction_error(
            model, model_for_initial_hidden, relearned_model, relearned_model_for_initial_hidden,
            X_train, Y_train, memory_bank_train, window_size
        )
        nominal_signed_val_prediction_error, original_signed_val_prediction_error, relearned_signed_val_prediction_error = get_signed_prediction_error(
            model, model_for_initial_hidden, relearned_model, relearned_model_for_initial_hidden,
            X_val, Y_val, memory_bank_val, window_size
        )

        if X_test is not None:
            nominal_signed_test_prediction_error, original_signed_test_prediction_error, relearned_signed_test_prediction_error = get_signed_prediction_error(
                model, model_for_initial_hidden, relearned_model, relearned_model_for_initial_hidden,
                X_test, Y_test, memory_bank_test, window_size
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
        if model_save_path is not None:
            self.save_given_model(relearned_model,relearned_model_for_initial_hidden,model_save_path,model_save_path.replace(".pth","_for_initial_hidden.pth"))
        if relearned_val_loss < original_val_loss or always_update_model:
            return relearned_model, relearned_model_for_initial_hidden, True
        else:
            return model, model_for_initial_hidden, False
        
    def get_trained_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100]):
        print("state_component_predicted: ", state_component_predicted)
        # Define Time Series Data
        X_train, Y_train, memory_bank_train = get_sequence_data_tensor(self.X_train_list, self.Y_train_list,self.division_indices_train)
        X_val, Y_val, memory_bank_val = get_sequence_data_tensor(self.X_val_list, self.Y_val_list,self.division_indices_val)

        self.model = error_prediction_NN.ErrorPredictionNN(
            prediction_length=prediction_length, state_component_predicted=state_component_predicted
        )
        self.model_for_initial_hidden = get_initial_hidden_NN.GetInitialHiddenNN(
            output_size=2*self.model.lstm_hidden_total_size
        )
        # self.update_adaptive_weight(None,None,X_train,Y_train,memory_bank_train)
        self.train_model(
            self.model,
            self.model_for_initial_hidden,
            X_train,
            Y_train,
            memory_bank_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            memory_bank_val
        )
    def get_relearned_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100],
                            reset_weight=False, randomize=0.001,plt_save_dir=None,model_save_path=None,
                            use_replay_data=False, replay_data_rate=0.05, always_update_model=False, freeze_shallow_layers=False):
        self.model.to(self.device)
        # Define Time Series Data
        X_train, Y_train, memory_bank_train = get_sequence_data_tensor(self.X_train_list, self.Y_train_list,self.division_indices_train)
        X_val, Y_val, memory_bank_val = get_sequence_data_tensor(self.X_val_list, self.Y_val_list,self.division_indices_val)
        if len(self.X_test_list) > 0:
            X_test, Y_test, memory_bank_test = get_sequence_data_tensor(self.X_test_list, self.Y_test_list,self.division_indices_test)
        else:
            X_test = None
            Y_test = None
            memory_bank_test = None
        if use_replay_data and len(self.X_replay_list) > 0:
            X_replay, Y_replay, memory_bank_replay = get_sequence_data_tensor(self.X_replay_list, self.Y_replay_list,self.division_indices_replay)
        else:
            X_replay = None
            Y_replay = None
            memory_bank_replay = None
            
        self.model, self.model_for_initial_hidden, updated = self.relearn_model(
            self.model,
            self.model_for_initial_hidden,
            X_train,
            Y_train,
            memory_bank_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            memory_bank_val,
            randomize=randomize,
            X_test=X_test,
            Y_test=Y_test,
            memory_bank_test=memory_bank_test,
            reset_weight=reset_weight,
            X_replay=X_replay,
            Y_replay=Y_replay,
            memory_bank_replay=memory_bank_replay,
            replay_data_rate=replay_data_rate,
            plt_save_dir=plt_save_dir,
            model_save_path=model_save_path,
            always_update_model=always_update_model,
            freeze_shallow_layers=freeze_shallow_layers
        )
        return updated
    def initialize_ensemble_models(self):
        self.models = [self.model]
        self.models_for_initial_hidden = [self.model_for_initial_hidden]
    def get_updated_temp_model(self,learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100,10,100],
                               use_replay_data=False, replay_data_rate=0.05,randomize_remaining_layers=0.0):
        self.temp_model = copy.deepcopy(self.model)
        self.temp_model_for_initial_hidden = copy.deepcopy(self.model_for_initial_hidden)
        X_train, Y_train, memory_bank_train = get_sequence_data_tensor(self.X_train_list, self.Y_train_list,self.division_indices_train)
        X_val, Y_val, memory_bank_val = get_sequence_data_tensor(self.X_val_list, self.Y_val_list,self.division_indices_val)
        if use_replay_data and len(self.X_replay_list) > 0:
            X_replay, Y_replay, memory_bank_replay = get_sequence_data_tensor(self.X_replay_list, self.Y_replay_list,self.division_indices_replay)
        else:
            X_replay = None
            Y_replay = None
            memory_bank_replay = None
        self.temp_model.to(self.device)
        self.update_adaptive_weight(self.temp_model, self.temp_model_for_initial_hidden,
                                    X_train, Y_train, memory_bank_train)
        self.train_model(self.temp_model,
            X_train,
            Y_train,
            memory_bank_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            memory_bank_val,
            freeze_shallow_layers=True,
            randomize_remaining_layers=randomize_remaining_layers,
            X_replay=X_replay,
            Y_replay=Y_replay,
            memory_bank_replay=memory_bank_replay,
            replay_data_rate=replay_data_rate
        )
    def relearn_temp_model(self, learning_rates=[1e-3, 1e-4, 1e-5, 1e-6], patience=10, batch_sizes=[100],
                           plt_save_dir=None,model_save_path=None, use_replay_data=False,
                           replay_data_rate=0.05,randomize_remaining_layers=0.0):
        self.temp_model.to(self.device)
        self.temp_model_for_initial_hidden.to(self.device)
        # Define Time Series Data
        X_train, Y_train, memory_bank_train = get_sequence_data_tensor(self.X_train_list, self.Y_train_list,self.division_indices_train)
        X_val, Y_val, memory_bank_val = get_sequence_data_tensor(self.X_val_list, self.Y_val_list,self.division_indices_val)
        if len(self.X_test_list) > 0:
            X_test, Y_test, memory_bank_test = get_sequence_data_tensor(self.X_test_list, self.Y_test_list,self.division_indices_test)
        else:
            X_test = None
            Y_test = None
            memory_bank_test = None
        if use_replay_data and len(self.X_replay_list) > 0:
            X_replay, Y_replay, memory_bank_replay = get_sequence_data_tensor(self.X_replay_list, self.Y_replay_list,self.division_indices_replay)
        else:
            X_replay = None
            Y_replay = None
            memory_bank_replay = None
        self.temp_model, self.temp_model_for_initial_hidden, updated = self.relearn_model(
            self.temp_model,
            self.temp_model_for_initial_hidden,
            X_train,
            Y_train,
            memory_bank_train,
            batch_sizes,
            learning_rates,
            patience,
            X_val,
            Y_val,
            memory_bank_val,
            X_test=X_test,
            Y_test=Y_test,
            memory_bank_test=memory_bank_test,
            X_replay=X_replay,
            Y_replay=Y_replay,
            memory_bank_replay=memory_bank_replay,
            replay_data_rate=replay_data_rate,
            plt_save_dir=plt_save_dir,
            model_save_path=model_save_path,
            freeze_shallow_layers=True,
            randomize=randomize_remaining_layers,
        )
        return updated
    def add_temp_model_to_ensemble(self):
        self.models.append(self.temp_model)
        self.models_for_initial_hidden.append(self.temp_model_for_initial_hidden)
    def save_model(self, path="vehicle_model.pth",path_for_initial_hidden="vehicle_model_for_initial_hidden.pth"):
        self.model.to("cpu")
        self.model_for_initial_hidden.to("cpu")
        torch.save(self.model, path)
        torch.save(self.model_for_initial_hidden, path_for_initial_hidden)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_model_to_csv(self.model, save_dir)
        convert_model_to_csv.convert_initial_hidden_model_to_csv(self.model_for_initial_hidden, save_dir)
    def save_given_model(self, model,model_for_initial_hidden, path="vehicle_model.pth",path_for_initial_hidden="vehicle_model_for_initial_hidden.pth"):
        model.to("cpu")
        model_for_initial_hidden.to("cpu")
        torch.save(model, path)
        torch.save(model_for_initial_hidden, path_for_initial_hidden)
        save_dir = path.replace(".pth", "")
        convert_model_to_csv.convert_model_to_csv(model, save_dir)
        convert_model_to_csv.convert_initial_hidden_model_to_csv(model_for_initial_hidden, save_dir)
    def save_ensemble_models(self, paths):
        for i in range(len(paths)):
            temp_model = self.models[i]
            temp_model.to("cpu")
            torch.save(temp_model, paths[i])
            temp_model_for_initial_hidden = self.models_for_initial_hidden[i]
            temp_model_for_initial_hidden.to("cpu")
            torch.save(temp_model_for_initial_hidden, paths[i].replace(".pth", "_for_initial_hidden.pth"))
            save_dir = paths[i].replace(".pth", "")
            convert_model_to_csv.convert_model_to_csv(temp_model, save_dir)
            convert_model_to_csv.convert_initial_hidden_model_to_csv(temp_model_for_initial_hidden, save_dir)
    def get_initial_gru_bank(self,path="gru_bank.csv"):
        random_memory_bank_np = get_random_memory_bank(np.array(self.X_train_list))
        random_memory_bank = torch.from_numpy(random_memory_bank_np).to(torch.float32).unsqueeze(0)
        if random_memory_bank.device != next(self.model_for_initial_hidden.parameters()).device:
            random_memory_bank = random_memory_bank.to(next(self.model_for_initial_hidden.parameters()).device)
        gru_bank = self.model_for_initial_hidden(random_memory_bank, only_gru=True).squeeze(0)
        gru_bank_np = gru_bank.cpu().detach().numpy()
        np.savetxt(path, gru_bank_np, delimiter=",")
    def freeze_shallow_layers(self,model,model_for_initial_hidden,randomize=0.001):
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
        for param in model.acc_layer_2.parameters():
            param.requires_grad = False
        for param in model.steer_layer_1.parameters():
            param.requires_grad = False
        for param in model.steer_layer_2.parameters():
            param.requires_grad = False
        for param in model.lstm.parameters():
            param.requires_grad = False

        # freeze model for initial hidden
        for param in model_for_initial_hidden.parameters():
            param.requires_grad = False

        # randomize the remaining parameters
        with torch.no_grad():
            model.complimentary_layer[0].weight += randomize * torch.randn_like(model.complimentary_layer[0].weight)
            model.complimentary_layer[0].bias += randomize * torch.randn_like(model.complimentary_layer[0].bias)
            model.linear_relu[0].weight += randomize * torch.randn_like(model.linear_relu[0].weight)
            model.linear_relu[0].bias += randomize * torch.randn_like(model.linear_relu[0].bias)
            model.final_layer.weight += randomize * torch.randn_like(model.final_layer.weight)
            model.final_layer.bias += randomize * torch.randn_like(model.final_layer.bias)
        
    def update_adaptive_weight(self,model, model_for_initial_hidden, X, Y, memory_bank,batch_size=1000):
        if model is not None:
            model.to(self.device)
            model_for_initial_hidden.to(self.device)
            model.eval()
            model_for_initial_hidden.eval()
        num_batches = (X.size(0) + batch_size - 1) // batch_size
        prediction_error = torch.zeros(self.adaptive_weight.shape[0], device=self.device)
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.size(0))
                if X.device != self.device:
                    X_batch = X[start_idx:end_idx].pin_memory().to(self.device, non_blocking=True)
                    Y_batch = Y[start_idx:end_idx].pin_memory().to(self.device, non_blocking=True)
                    memory_bank_batch = memory_bank[start_idx:end_idx].pin_memory().to(self.device, non_blocking=True)
                else:
                    X_batch = X[start_idx:end_idx]
                    Y_batch = Y[start_idx:end_idx]
                    memory_bank_batch = memory_bank[start_idx:end_idx]
                if model is not None:
                    hc_initial_concat = model_for_initial_hidden(memory_bank_batch)
                    hc_initial = (hc_initial_concat[:, :model.lstm_hidden_total_size].unsqueeze(0).contiguous(), hc_initial_concat[:, model.lstm_hidden_total_size:].unsqueeze(0).contiguous())
                    Y_pred, _ = model(X_batch, previous_error=Y_batch[:, :past_length, -2:],hc=hc_initial, mode="get_lstm_states")
                    # Y_pred, _ = model(X_batch, previous_error=Y_batch[:, :past_length, -2:], mode="get_lstm_states")
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
