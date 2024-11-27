from multiprocessing import Process
from autoware_vehicle_adaptor.training import train_error_prediction_NN_with_memory_bank
import torch

def get_trained_model_wrapper(learning_rates, batch_sizes, train_dirs, val_dirs, train_reverse_steers, val_reverse_steers, model_save_dir):
    model_trainer = train_error_prediction_NN_with_memory_bank.train_error_prediction_NN_with_memory_bank()
    for i in range(len(train_dirs)):
        model_trainer.add_data_from_csv(train_dirs[i], add_mode = "as_train", reverse_steer=train_reverse_steers[i])
    for i in range(len(val_dirs)):
        model_trainer.add_data_from_csv(val_dirs[i], add_mode = "as_val", reverse_steer=val_reverse_steers[i])
    model_trainer.get_trained_model(learning_rates=learning_rates, batch_sizes=batch_sizes)
    model_trainer.save_model(path=model_save_dir+"/vehicle_model.pth", path_for_initial_hidden=model_save_dir+"/vehicle_model_for_initial_hidden.pth")

def get_relearned_model_wrapper(
        learning_rates,
        batch_sizes,
        train_dirs,
        val_dirs,
        test_dirs,
        replay_dirs,
        train_reverse_steers,
        val_reverse_steers,
        test_reverse_steers,
        replay_reverse_steers,       
        model_load_dir,
        model_save_dir,
        randomize=0.01,
        patience=10,
        replay_data_rate=0.05,
        reset_weight=False,
        always_update_model=False,
        freeze_shallow_layers=False):
    model_trainer = train_error_prediction_NN_with_memory_bank.train_error_prediction_NN_with_memory_bank()
    for i in range(len(train_dirs)):
        model_trainer.add_data_from_csv(train_dirs[i], add_mode = "as_train", reverse_steer=train_reverse_steers[i])
    for i in range(len(val_dirs)):
        model_trainer.add_data_from_csv(val_dirs[i], add_mode = "as_val", reverse_steer=val_reverse_steers[i])
    for i in range(len(test_dirs)):
        model_trainer.add_data_from_csv(test_dirs[i], add_mode = "as_test", reverse_steer=test_reverse_steers[i])
    if replay_dirs is not None:
        for i in range(len(replay_dirs)):
            model_trainer.add_data_from_csv(replay_dirs[i], add_mode = "as_replay", reverse_steer=replay_reverse_steers[i])
        use_replay_data = True
    else:
        use_replay_data = False
    model_trainer.model = torch.load(model_load_dir+"/vehicle_model.pth")
    model_trainer.model_for_initial_hidden = torch.load(model_load_dir+"/vehicle_model_for_initial_hidden.pth")
    model_trainer.get_relearned_model(
        learning_rates=learning_rates, batch_sizes=batch_sizes, patience=patience,
        use_replay_data=use_replay_data, randomize=randomize,
        replay_data_rate=replay_data_rate, reset_weight=reset_weight,
        always_update_model=always_update_model,
        freeze_shallow_layers=freeze_shallow_layers,
        plt_save_dir=model_save_dir, model_save_path=model_save_dir+"/tmp_vehicle_model.pth")
    model_trainer.save_model(path=model_save_dir+"/vehicle_model.pth", path_for_initial_hidden=model_save_dir+"/vehicle_model_for_initial_hidden.pth")
def get_trained_model(learning_rates, batch_sizes, train_dirs, val_dirs, train_reverse_steers, val_reverse_steers, model_save_dir):
    p = Process(target=get_trained_model_wrapper, args=(learning_rates, batch_sizes, train_dirs, val_dirs, train_reverse_steers, val_reverse_steers, model_save_dir))
    p.start()
    p.join()
def get_relearned_model(
        learning_rates,
        batch_sizes,
        train_dirs,
        val_dirs,
        test_dirs,
        replay_dirs,
        train_reverse_steers,
        val_reverse_steers,
        test_reverse_steers,
        replay_reverse_steers,       
        model_load_dir,
        model_save_dir,
        randomize=0.01,
        patience=10,
        replay_data_rate=0.05,
        reset_weight=False,
        always_update_model=False,
        freeze_shallow_layers=False):
    p = Process(target=get_relearned_model_wrapper, args=(
        learning_rates,
        batch_sizes,
        train_dirs,
        val_dirs,
        test_dirs,
        replay_dirs,
        train_reverse_steers,
        val_reverse_steers,
        test_reverse_steers,
        replay_reverse_steers,       
        model_load_dir,
        model_save_dir,
        randomize,
        patience,
        replay_data_rate,
        reset_weight,
        always_update_model,
        freeze_shallow_layers))
    p.start()
    p.join()

    




