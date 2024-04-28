import os
from train import *

# datasets["num_work"] = 1

dataset_name = "abalone"
datasets[dataset_name]["c"] = 3.0
datasets[dataset_name]["optim_rate"] = 0.001
datasets[dataset_name]["epoch"] = 1000
load_data(dataset_name)

datasets[dataset_name]["slow-star"] = True
datasets["decrease"] = False
datasets["slow"] = 200
datasets["stop_c"] = 0
datasets["inf"] = 0

datasets["weight_decay"] = 0
datasets["drop_out"] = 0
model_type = "RwR_mlp"
loss_type = "RwR_loss_mae"

metrics = ["RwR_Risk_Evaluation", "A_loss", "R_loss", "Reject_Rate", "R_A", "A_R"]

best_model = train_dataset_model(dataset_name, loss_type, model_type=model_type,
                                 metrics=metrics,
                                 print_show=True)