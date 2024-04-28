import mindspore
import numpy as np
import os
from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis
from train_and_validation.train_with_dp import  DPtrain
from train_and_validation.validation import validation
import data.util.dataloader as dl
from data.util.dividing_validation_data import dividing_validation_set
from privacy_analysis.dp_utils import scatter_normalization
from privacy_analysis.RDP.compute_rdp import compute_rdp
from privacy_analysis.RDP.rdp_convert_dp import compute_eps
from data.util.scatter import get_scatter_transform, get_scattered_dataset
from model.CNN import MNIST_CNN_Tanh
from privacy_analysis.RDP.get_MaxSigma_or_MaxSteps import get_max_steps, get_min_sigma
from datetime import datetime
def DPSUR(train_data_name, test_data_name, argModel, l2_norm_clip, learning_rate,
          batch_size, epsilon_budget, delta, sigma, device, size_valid,
          use_scattering, input_norm, bn_noise_multiplier, C_v, beta):

    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]

    train_dataset, valid_dataset, test_dataset = dividing_validation_set(train_data_name, size_valid)

    train_data, train_label = train_dataset
    valid_data, valid_label = valid_dataset
    test_data, test_label = test_dataset

    # For shuffle
    train_indicies = np.arange(train_data.shape[0])
    np.random.shuffle(train_indicies)
    valid_indices = np.arange(valid_data.shape[0])
    np.random.shuffle(valid_indices)

    train_data_shuffled = train_data[train_indicies]
    train_label_shuffled = train_label[train_indicies]
    valid_data_shuffled = valid_data[valid_indices]
    valid_label_shuffled = valid_label[valid_indices]
    # For batching
    train_batches_num = int (np.ceil(train_data.shape[0] / batch_size))
    train_data_batched = np.array_split(train_data_shuffled, train_batches_num)
    train_label_batched = np.array_split(train_label_shuffled, train_batches_num)

    valid_batches_num = int (np.ceil(valid_data.shape[0] / batch_size))
    valid_data_batched = np.array_split(valid_data_shuffled, valid_batches_num)
    valid_label_batched = np.array_split(valid_label_shuffled, valid_batches_num)

    test_batches_num = int(np.ceil(test_data.shape[0] / batch_size))
    test_data_batched = np.array_split(test_data, test_batches_num)
    test_label_batched = np.array_split(test_label, test_batches_num)

    train_dataset = (train_data_batched, train_label_batched)
    valid_dataset = (valid_data_batched, valid_label_batched)
    test_dataset = (test_data_batched, test_label_batched)

    if use_scattering:
        scattering, K, _ = get_scatter_transform(train_data_name)
    else:
        scattering = None
        K = 3 if train_data_name == "Cifar10" else 1

    rdp_norm = 0
    if input_norm == "BN":
        save_dir = f"bn_stats/{train_data_name}"
        os.makedirs(save_dir, exist_ok=True)
        bn_stats, rdp_norm = scatter_normalization(train_dataset,
                                                   scattering,
                                                   K,
                                                   device,
                                                   len(train_data_shuffled),
                                                   len(train_data_shuffled),
                                                   noise_multiplier=bn_noise_multiplier,
                                                   orders=orders,
                                                   save_dir=save_dir)
        model = MNIST_CNN_Tanh(K, input_norm = "BN", bn_stats = bn_stats, size = None)
    train_data_scattered = get_scattered_dataset(train_dataset, scattering, device, len(train_data_shuffled))
    valid_data_scattered = get_scattered_dataset(valid_dataset, scattering, device, len(valid_data_shuffled))
    test_data_scattered = get_scattered_dataset(test_dataset, scattering, device, len(test_data))
    test_dl = dl.load_test_data(is_dpsur = True, data = test_data_scattered)

    last_valid_loss = 100000.0
    last_accept_test_acc = 0.
    last_model = model
    t = 0
    iter = 1
    best_iter = -1
    epsilon = 0.
    len_train_dataset =0
    len_train_data = 0
    len_valid_data = 0
    best_test_acc = 0.
    if train_data_name == "MNIST":
        len_train_dataset = 60000.0
        len_train_data = 55000.0
        len_valid_data = 5000.0

    epsilon_budget_for_train = epsilon_budget / (len_train_data / len_train_dataset)
    print("privacy budget of training set:", epsilon_budget_for_train)

    epsilon_budget_for_valid_in_all_updates = epsilon_budget / (len_valid_data / len_train_dataset)
    print("epsilon_budget_for_valid_in_all_updates:", epsilon_budget_for_valid_in_all_updates)

    max_number_of_updates = get_max_steps(epsilon_budget_for_train, delta, batch_size / len_train_data, sigma, orders)
    print("max_number_of_updates:", max_number_of_updates)

    epsilon_budget_for_train_in_one_iter, _ = apply_dp_sgd_analysis(batch_size / len_train_data, sigma, 1, orders,
                                                                    delta)
    print("epsilon_budget_for_train_in_one_iter:", epsilon_budget_for_train_in_one_iter)

    sigma_for_valid = get_min_sigma(epsilon_budget_for_valid_in_all_updates,
                                    len_train_data / len_valid_data * epsilon_budget_for_train_in_one_iter, delta,
                                    batch_size / len_valid_data, max_number_of_updates, orders)
    while epsilon < epsilon_budget_for_train:
        if input_norm == "BN":
            rdp = rdp = compute_rdp(batch_size / len_train_data, sigma, t, orders)
            epsilon, best_alpha = compute_eps(orders, rdp + rdp_norm, delta)
        else:
            epsilon, best_alpha = apply_dp_sgd_analysis(batch_size / len_train_data, sigma, t, orders, delta)

        train_dl = dl.load_train_data_minibatch(data = train_data_scattered, minibatch_size = batch_size)
        valid_dl = dl.load_train_data_minibatch(data = valid_data_scattered, minibatch_size = batch_size)

        dp_train = DPtrain(model, train_dl, l2_norm_clip, sigma, batch_size, learning_rate, device)
        train_loss, train_accuracy = dp_train.train_with_dp()
        valid_loss, valid_accuracy = validation(model, valid_dl)
        test_loss, test_accuracy = validation(model, test_dl)

        deltaE = valid_loss - last_valid_loss
        print("Delta E:", deltaE)

        deltaE = np.clip(deltaE, -C_v, C_v)
        deltaE_after_dp = 2 * C_v * sigma_for_valid * np.random.normal(0, 1) + deltaE

        print("Delta E after dp:", deltaE_after_dp)

        if deltaE_after_dp < beta * C_v or iter == 1:
            last_valid_loss = valid_loss
            mindspore.save_checkpoint(model, "last_model.ckpt")
            t = t + 1
            print("accept updates，the number of updates t：", format(t))
            last_accept_test_acc = test_accuracy
            if last_accept_test_acc > best_test_acc:
                best_test_acc = last_accept_test_acc
                best_iter = t

        else:
            print("reject updates")
            param_dict = mindspore.load_checkpoint("last_model.ckpt")
            mindspore.load_param_into_net(model, param_dict)
        print(
            f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' Test set: Average loss: {test_loss:.4f},'f' Accuracy:({test_accuracy:.2f}%)')

        iter += 1

    print("------ finished ------")
    return last_accept_test_acc, t, best_test_acc, best_iter
