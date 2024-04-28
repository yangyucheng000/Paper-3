import torch
import torch.optim as optim
from losses import *
from model import *
from parameter import *
import copy

RwRM = ["RwR_mlp", "RwR_linear", "RwR_ResNet", "RwR_one_ResNet"]


def train_dataset_model(dataset_name, loss_type, model_type="RwR_mlp", metrics=None, print_show=False, out_dim=1):
    if metrics is None:
        metrics = ["RwR_Risk_Evaluation"]
    if model_type == "RwR_mlp":
        model = RwRModel(len(datasets[dataset_name]["train_dataset"].__getitem__(0)[0]), out_dim=out_dim)
    elif model_type == "RwR_linear":
        model = RwRLinear(len(datasets[dataset_name]["train_dataset"].__getitem__(0)[0]), out_dim=out_dim)
    elif model_type == "mlp":
        model = MlpModel(len(datasets[dataset_name]["train_dataset"].__getitem__(0)[0]), out_dim=out_dim)
    elif model_type == "linear":
        model = LinearModel(len(datasets[dataset_name]["train_dataset"].__getitem__(0)[0]), out_dim=out_dim)
    elif model_type == "RwR_ResNet":
        model = RwR_ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim, dropout=datasets["drop_out"])
    elif model_type == "RwR_one_ResNet":
        model = RwR_one_ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim, dropout=datasets["drop_out"])
    elif model_type == "ResNet":
        model = ResNet(Bottleneck, [3, 4, 6, 3], out_dim=out_dim, dropout=datasets["drop_out"])
    else:
        print("invalid model")
        sys.exit()
    if (model_type in RwRM) and \
            datasets[dataset_name][
                "slow-star"]:
        if torch.cuda.device_count() > 1:
            for i in model.module.r.parameters():
                i.requires_grad = False
        else:
            for i in model.r.parameters():
                i.requires_grad = False
    optimizer = optim.Adam(model.parameters(), learning_rate=datasets[dataset_name]["optim_rate"], beta1=0.9, beta2 = 0.999, eps=1e-08,
                           weight_decay=datasets["weight_decay"])
    best_model = {}
    max_loss = {}
    for metric in metrics:
        max_loss[metric] = MAX_INT
        if metric in ["RwR_Risk_Evaluation", "real"]:
            max_loss[metric] = MAX_INT
            best_model[metric] = copy.deepcopy(model)
        elif metric in []:
            max_loss[metric] = 0

    if datasets["decrease"]:
        c = 150
    else:
        c = datasets[dataset_name]["c"]

    if datasets[dataset_name]["slow-star"]:
        s = True
    else:
        s = False

    for iter_epoch in range(datasets[dataset_name]["epoch"]):
        if (iter_epoch == datasets["slow"]) and (
                model_type in RwRM) and \
                datasets[dataset_name][
                    "slow-star"]:
            if torch.cuda.device_count() > 1:
                for i in model.module.r.parameters():
                    i.requires_grad = True
            else:
                for i in model.r.parameters():
                    i.requires_grad = True

        if print_show:
            print('Epoch {}/{}'.format(iter_epoch + 1, datasets[dataset_name]["epoch"]))
        losses_train = 0
        for i, (feature, target, _) in enumerate(
                torch.utils.data.DataLoader(datasets[dataset_name]["train_dataset"],
                                            batch_size=datasets["batch_size"],
                                            num_workers=datasets["num_work"], pin_memory=datasets["pin_memory"])):
            feature, target = feature.cuda(non_blocking=datasets["non_blocking"]), target.cuda(
                non_blocking=datasets["non_blocking"])

            model.train()
            optimizer.zero_grad()
            if model_type in RwRM:
                pre, reject = model(feature)

                if not (datasets[dataset_name]["slow-star"] and (iter_epoch <= datasets["slow"])):
                    s = False
            else:
                pre = model(feature)

            if loss_type == "real":
                loss = real_loss(pre, target)
            elif loss_type == "RwR_loss_sigmoid":
                loss = RwR_loss_sigmoid(pre, reject, target, c, datasets["inf"], s)
            elif loss_type == "RwR_loss_logistic":
                loss = RwR_loss_logistic(pre, reject, target, c, datasets["inf"], s)
            elif loss_type == "RwR_loss_mse":
                loss = RwR_loss_mse(pre, reject, target, c, datasets["inf"], s)
            elif loss_type == "RwR_loss_mae":
                loss = RwR_loss_mae(pre, reject, target, c, datasets["inf"], s)
            elif loss_type == "RwR_loss_hinge":
                loss = RwR_loss_hinge(pre, reject, target, c, datasets["inf"], s)
            else:
                print("invalid loss function")
                sys.exit()
            loss.backward()
            optimizer.step()
            model.eval()
            losses_train += loss.item() * feature.shape[0]

        verify = {}
        for metric in metrics:
            verify[metric] = 0
        if model_type in RwRM:
            verify["R_A_m"] = 0
            verify["A_R_m"] = 0

        for i, (feature, target, _) in enumerate(
                torch.utils.data.DataLoader(datasets[dataset_name]["verify_dataset"],
                                            batch_size=datasets["batch_size"],
                                            num_workers=datasets["num_work"], pin_memory=datasets["pin_memory"])):
            feature, target = feature.cuda(non_blocking=datasets["non_blocking"]), target.cuda(
                non_blocking=datasets["non_blocking"])
            with torch.no_grad():
                if model_type in RwRM:
                    pre, reject = model(feature)
                else:
                    pre = model(feature)

                for metric in metrics:
                    if metric == "RwR_Risk_Evaluation":
                        verify[metric] += RwR_Risk_Evaluation(pre, reject, target, datasets[dataset_name]["c"]).item() * \
                                          feature.shape[0]
                    elif metric == "A_loss":
                        verify[metric] += A_loss(pre, reject, target).item() * feature.shape[0]
                    elif metric == "R_loss":
                        verify[metric] += R_loss(pre, reject, target).item() * feature.shape[0]
                    elif metric == "Reject_Rate":
                        verify[metric] += Reject_Rate(reject).item() * feature.shape[0]
                    elif metric == "R_A":
                        n, m = R_A(pre, reject, target, datasets[dataset_name]["c"])
                        verify[metric] += n
                        verify["R_A_m"] += m
                    elif metric == "A_R":
                        n, m = A_R(pre, reject, target, datasets[dataset_name]["c"])
                        verify[metric] += n
                        verify["A_R_m"] += m
                    elif metric == "real":
                        verify[metric] += real_loss(pre, target).item() * feature.shape[0]
        for metric in metrics:
            if metric == "R_A":
                verify[metric] = (verify[metric] + 1) / (verify["R_A_m"] + 1)
            elif metric == "A_R":
                verify[metric] = (verify[metric] + 1) / (verify["A_R_m"] + 1)
            else:
                verify[metric] = verify[metric] / len(datasets[dataset_name]["verify_dataset"])

        l = (losses_train / len(datasets[dataset_name]["train_dataset"]))
        if print_show:
            print('Training loss : \t loss_train = {}\t verify = {}'.format(l, verify))

        up_model = 0
        for metric in metrics:
            if metric in ["RwR_Risk_Evaluation", "real"]:
                if verify[metric] <= max_loss[metric]:
                    best_model[metric] = copy.deepcopy(model)
                    max_loss[metric] = verify[metric]
                    up_model = 1
            elif (metric in ["A_loss", "R_loss", "Reject_Rate", "R_A", "A_R"]) and up_model:
                max_loss[metric] = verify[metric]

        if datasets["decrease"]:
            if iter_epoch > (datasets[dataset_name]["epoch"] * datasets["stop_c"]):
                c = datasets[dataset_name]["c"]
            elif (l > datasets[dataset_name]["c"]) and (c > l):
                c = l

    return best_model


def test_model(model, dataset_name, model_type, metrics):
    model.eval()
    test = {}
    for metric in metrics:
        test[metric] = 0

    if model_type in RwRM:
        test["R_A_m"] = 0
        test["A_R_m"] = 0

    for i, (feature, target, _) in enumerate(
            torch.utils.data.DataLoader(datasets[dataset_name]["test_dataset"],
                                        batch_size=datasets["batch_size"],
                                        num_workers=datasets["num_work"], pin_memory=datasets["pin_memory"])):
        feature, target = feature.cuda(non_blocking=datasets["non_blocking"]), target.cuda(
            non_blocking=datasets["non_blocking"])
        with torch.no_grad():
            if model_type in RwRM:
                pre, reject = model(feature)
            else:
                pre = model(feature)
            for metric in metrics:
                if metric == "RwR_Risk_Evaluation":
                    test[metric] += RwR_Risk_Evaluation(pre, reject, target, datasets[dataset_name]["c"]).item() * \
                                    feature.shape[0]
                elif metric == "A_loss":
                    test[metric] += A_loss(pre, reject, target).item() * feature.shape[0]
                elif metric == "R_loss":
                    test[metric] += R_loss(pre, reject, target).item() * feature.shape[0]
                elif metric == "Reject_Rate":
                    test[metric] += Reject_Rate(reject).item() * feature.shape[0]
                elif metric == "R_A":
                    n, m = R_A(pre, reject, target, datasets[dataset_name]["c"])
                    test[metric] += n
                    test["R_A_m"] += m
                elif metric == "A_R":
                    n, m = A_R(pre, reject, target, datasets[dataset_name]["c"])
                    test[metric] += n
                    test["A_R_m"] += m
                elif metric == "real":
                    test[metric] += real_loss(pre, target).item() * feature.shape[0]
    for metric in metrics:
        if metric == "R_A":
            test[metric] = (test[metric] + 1) / (test["R_A_m"] + 1)
        elif metric == "A_R":
            test[metric] = (test[metric] + 1) / (test["A_R_m"] + 1)
        else:
            test[metric] = test[metric] / len(datasets[dataset_name]["verify_dataset"])
    return test

