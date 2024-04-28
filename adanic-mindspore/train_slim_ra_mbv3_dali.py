import argparse
import math
import random
import shutil
import os
import sys
from datetime import datetime
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch.utils.data import DataLoader
# from torchvision import transforms
# from compressai.datasets import ImageFolder
# from compressai.zoo import models
from models.slim_models import AdaMeanScaleHyperprior, RoutingAgent

import logging
import tensorboardX

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


@pipeline_def
def create_dali_pipeline(data_dir, crop, shard_id, num_shards, dali_cpu=False, is_training=True):
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    images, labels = fn.readers.file(files=file_list,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader",
                                     # labels=0  # give unified class labels (useless for compression)
                                     )
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in dataset for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 16777216 if decoder_device == 'mixed' else 0
    host_memory_padding = 8388608 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the dataset to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0  # not updated for CLIC
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0  # not updated for CLIC

    device_memory_padding = host_memory_padding = preallocate_height_hint = preallocate_width_hint = 0

    if is_training:
        # random generator
        anchor_x_rel = fn.random.uniform(labels, range=[0.0, 1.0])
        anchor_y_rel = fn.random.uniform(labels, range=[0.0, 1.0])
    else:
        # center crop
        anchor_x_rel = 0.5
        anchor_y_rel = 0.5

    images = fn.decoders.image_crop(
        images,
        crop_w=crop[0],  # fixed patch size
        crop_h=crop[1],
        crop_pos_x=anchor_x_rel,
        crop_pos_y=anchor_y_rel,
        device=decoder_device, output_type=types.RGB,
        device_memory_padding=device_memory_padding,
        host_memory_padding=host_memory_padding,
        preallocate_width_hint=preallocate_width_hint,
        preallocate_height_hint=preallocate_height_hint,
        memory_stats=False  # get stats at run-time
    )

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=crop,
                                      mean=[0.0 * 255, 0.0 * 255, 0.0 * 255],
                                      std=[1.0 * 255, 1.0 * 255, 1.0 * 255],
                                      )

    labels = labels.gpu()
    return images, labels


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, cur_lmbda=None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        if not cur_lmbda:
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["loss"] = cur_lmbda * out["mse_loss"] + out["bpp_loss"]

        return out

# Sample-Level Metrics
class SampleCost(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.lmbda = lmbda

    def forward(self, output, target, cur_lmbda=None):
        N, _, H, W = target.size()
        out = {}
        num_pixels = H * W
        with torch.no_grad():
            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum(dim=[1,2,3]) / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()  # "y" & "z"
            )
            out["mse_loss"] = self.mse(output["x_hat"], target).mean(dim=[1,2,3])
            out["psnr"] = -10 * torch.log10(out["mse_loss"])
            if not cur_lmbda:
                out["cost"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
            else:
                out["cost"] = cur_lmbda * out["mse_loss"] + out["bpp_loss"]

        return out


class RoutingLoss(nn.Module):
    def __init__(self, epsilon=2e-2, gamma=0.5, temperature=1.0,
                 distill_weight=0.9, action_reg_loss=False, action_loss_penalty=False):
        super().__init__()
        # JUMC
        self.epsilon = epsilon
        self.gamma = gamma  # balancing factor for two objectives
        # kd
        self.temperature = temperature
        self.distill_weight = distill_weight
        self.kl = nn.KLDivLoss(reduction='batchmean')
        # regression
        self.mse = nn.MSELoss()
        self.mse_nr = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss()
        # classification
        self.ce = nn.CrossEntropyLoss()
        self.ce_nr = nn.CrossEntropyLoss(reduction="none")
        # booleans
        self.action_reg_loss = action_reg_loss
        self.action_loss_penalty = action_loss_penalty if action_loss_penalty != 1.0 else False

    def jusm_by_threshold(self, deltas, epsilon):
        N = deltas.shape[0]
        jusm = torch.ones((N,), device=deltas.device, dtype=torch.int64) * 4  # LongTensor as classification target
        # sample
        for i in range(N):
            for q in range(3, -1, -1):
                if deltas[i, q] >= epsilon:
                    break
                jusm[i] = q
        return jusm

    def forward(self, ra_output, target_rcosts, epsilon=None, out_teacher=None):
        N, _ = target_rcosts.size()
        out = {}

        if not epsilon:
            epsilon = self.epsilon

        # assembly action loss
        out["jusm_true"] = self.jusm_by_threshold(target_rcosts, epsilon=epsilon)
        decision_c = torch.sum(torch.softmax(ra_output["decision"], dim=-1) * torch.arange(5, device=target_rcosts.device), dim=-1)
        decision_d = ra_output["decision"].argmax(dim=-1)
        deg_mask = decision_d < out["jusm_true"]
        if not self.action_loss_penalty:
            out["action_ce"] = self.ce(ra_output["decision"], out["jusm_true"])
            out["action_mse"] = self.mse(decision_c, out["jusm_true"].type(torch.float32))  # casting to floating-point type
        else:
            ce_all = self.ce_nr(ra_output["decision"], out["jusm_true"])
            mse_all = self.mse_nr(decision_c, out["jusm_true"].type(torch.float32))
            out["action_ce"] = torch.mean(self.action_loss_penalty * ce_all * deg_mask + ce_all * ~deg_mask, dim=0)
            out["action_mse"] = torch.mean(self.action_loss_penalty * mse_all * deg_mask + mse_all * ~deg_mask, dim=0)

        out["action_loss"] = out["action_mse"] if self.action_reg_loss else out["action_ce"]  # [optional] + task loss (CAE controlled by action)

        # assembly cost reg loss
        out["reg_loss"] = self.mse(ra_output["cost"], target_rcosts)

        # assembly kd loss
        if out_teacher:
            out["action_kl"] = self.kl(
                F.log_softmax(ra_output["decision"] / self.temperature, dim=1),
                F.softmax(out_teacher["decision"].detach() / self.temperature, dim=1)
            ) * (self.temperature * self.temperature)
            out["reg_mse"] = self.mse(
                ra_output["cost"], out_teacher["cost"]
            )
            out["action_loss"] = (1-self.distill_weight) * out["action_loss"] + self.distill_weight * out["action_kl"]
            out["reg_loss"] = (1-self.distill_weight) * out["reg_loss"] + self.distill_weight * out["reg_mse"]

        # assembly combined loss
        out["loss"] = out["action_loss"] * (1 - self.gamma) + out["reg_loss"] * self.gamma

        with torch.no_grad():
            out["reg_mae"] = self.mae(ra_output["cost"], target_rcosts)
            decision = ra_output["decision"].argmax(dim=-1)
            out["action_mae"] = self.mae(decision.type(torch.float32),
                                         out["jusm_true"].type(torch.float32))  # casting to floating-point type
            out["action_acc"] = (decision == out["jusm_true"]).sum() / N
            out["action_deg"] = (decision < out["jusm_true"]).sum() / N
            decision_reg = self.jusm_by_threshold(ra_output["cost"], epsilon=epsilon)
            out["reg_action_mae"] = self.mae(decision_reg.type(torch.float32),
                                         out["jusm_true"].type(torch.float32))  # casting to floating-point type
            out["reg_action_acc"] = (decision_reg == out["jusm_true"]).sum() / N
            out["reg_action_deg"] = (decision_reg < out["jusm_true"]).sum() / N

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
        model, routing_agent, criterion, patch_criterion, ra_criterion,
        train_dataloader, optimizer, aux_optimizer, ra_optimizer,
        epoch, clip_max_norm, N_list, lmbda_list, writers=None, teacher=None
):
    # get loader len
    total_samples = train_dataloader._size
    total_batches = int(math.ceil(train_dataloader._size / args.batch_size))

    if teacher:
        teacher.eval()
    model.eval()
    routing_agent.train()
    device = next(model.parameters()).device
    loss = collections.defaultdict(AverageMeter)
    mse_loss = collections.defaultdict(AverageMeter)
    bpp_loss = collections.defaultdict(AverageMeter)
    aux_loss = AverageMeter()
    ra_loss = AverageMeter()
    ra_reg_mae = AverageMeter()
    ra_action_mae, ra_action_acc, ra_action_deg, ra_action_kl = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    ra_reg_action_mae, ra_reg_action_acc, ra_reg_action_deg = AverageMeter(), AverageMeter(), AverageMeter()

    # train by batch (by iterating the dataloader, random patches are retrieved)
    for i, d in enumerate(train_dataloader):
        d = d[0]["data"]
        patch_costs = torch.zeros((len(N_list), d.shape[0]), device=device)
        with torch.no_grad():
            # traverse sub-networks, get rate / distortion criterion
            for j, width in enumerate(N_list):
                model.set_running_width(width)
                out_net = model(d)
                # training loss (by item, for stats)
                out_criterion = criterion(out_net, d, lmbda_list[j])
                loss[width].update(out_criterion["loss"].item())
                mse_loss[width].update(out_criterion["mse_loss"].item())
                bpp_loss[width].update(out_criterion["bpp_loss"].item())
                # patch rd, as labels for the routing agent
                out_patch_criterion = patch_criterion(out_net, d, cur_lmbda=args.rcost_lambda)
                patch_costs[j] = out_patch_criterion["cost"]

            # values to deltas
            rcosts = torch.zeros((d.shape[0], len(N_list)-1), device=device)
            for j in range(len(N_list)-1):
                rcosts[:, j] = patch_costs[j] - patch_costs[-1]

        # train agent
        ra_optimizer.zero_grad()
        out_ra = {}
        out_ra["cost"], out_ra["decision"] = routing_agent(d)
        out_teacher = None
        if teacher:
            out_teacher = {}
            with torch.no_grad():
                out_teacher["cost"], out_teacher["decision"] = teacher(d)
        out_ra_criterion = ra_criterion(out_ra, rcosts, epsilon=args.epsilon, out_teacher=out_teacher)
        out_ra_criterion["loss"].backward()
        ra_loss.update(out_ra_criterion["loss"].item())
        ra_reg_mae.update(out_ra_criterion["reg_mae"].item())
        ra_action_mae.update(out_ra_criterion["action_mae"].item())
        ra_action_acc.update(out_ra_criterion["action_acc"].item())
        ra_action_deg.update(out_ra_criterion["action_deg"].item())
        if teacher:
            ra_action_kl.update(out_ra_criterion["action_kl"].item())
        ra_reg_action_mae.update(out_ra_criterion["reg_action_mae"].item())
        ra_reg_action_acc.update(out_ra_criterion["reg_action_acc"].item())
        ra_reg_action_deg.update(out_ra_criterion["reg_action_deg"].item())
        ra_optimizer.step()

        # aux stats
        with torch.no_grad():
            aux_loss_full = model.aux_loss()
        aux_loss.update(aux_loss_full.item())

        if i % 10 == 0:
            full_width = N_list[-1]
            logging.info(
                f"Train epoch {epoch}: ["
                f"{i*args.batch_size+len(d)}/{total_samples}"
                f" ({100. * (i+1) / total_batches:.0f}%)]"
                f'\tLoss: {loss[full_width].val:.3f} ({loss[full_width].avg:.3f}) |'
                # f'\tMSE loss: {mse_loss[full_width].val:.4f} ({mse_loss[full_width].avg:.4f}) |'
                f'\tPSNR: {-10 * math.log10(mse_loss[full_width].val):.2f} ({-10 * math.log10(mse_loss[full_width].avg):.2f}) |'
                f'\tBpp loss: {bpp_loss[full_width].val:.3f} ({bpp_loss[full_width].avg:.3f}) |'
                f"\tAux loss: {aux_loss.val:.2f} ({aux_loss.avg:.2f})"
            )

            logging.info(
                f"Train epoch {epoch}: ["
                f"{i*args.batch_size+len(d)}/{total_samples}"
                f" ({100. * (i+1) / total_batches:.0f}%)]"
                f'\tRA Loss: {ra_loss.val:.3f} ({ra_loss.avg:.3f}) |'
                f'\tCost Reg MAE: {ra_reg_mae.avg:.3f} |'
                f"\tAction MAE: {ra_action_mae.avg:.3f} |"
                f'\tAction Acc: {ra_action_acc.avg*100:.2f}% |'
                f'\tAction Deg: {ra_action_deg.avg*100:.2f}%'
            )

            if teacher:
                logging.info(
                    f"Train epoch {epoch}: ["
                    f"{i*args.batch_size+len(d)}/{total_samples}"
                    f" ({100. * (i+1) / total_batches:.0f}%)]"
                    f"\tAction KL Div: {ra_action_kl.avg:.3f} |"
                )

            logging.info(
                f"Train epoch {epoch}: ["
                f"{i*args.batch_size+len(d)}/{total_samples}"
                f" ({100. * (i+1) / total_batches:.0f}%)]"
                f"\tReg Action MAE: {ra_reg_action_mae.avg:.3f} |"
                f'\tReg Action Acc: {ra_reg_action_acc.avg * 100:.2f}% |'
                f'\tReg Action Deg: {ra_reg_action_deg.avg * 100:.2f}%'
            )

            j_pred = out_ra["decision"].argmax(dim=-1).type(torch.float32)
            j_true = out_ra_criterion["jusm_true"].type(torch.float32)
            logging.info(
                f"Train epoch {epoch}: (std, min, avg, max)"
                f'\tPred: ({j_pred.std():.3f}, {j_pred.min():.3f}, {j_pred.mean():.3f}, {j_pred.max():.3f}) |'
                f'\tGT: ({j_true.std():.3f}, {j_true.min():.3f}, {j_true.mean():.3f}, {j_true.max():.3f})\n'
            )

            step = epoch*total_batches+i
            for j, N in enumerate(N_list):
                writers[j].add_scalar("train/loss", loss[N].avg, step)
                writers[j].add_scalar("train/mse_loss", mse_loss[N].avg, step)
                writers[j].add_scalar("train/psnr", -10 * math.log10(mse_loss[N].avg), step)
                writers[j].add_scalar("train/bpp_loss", bpp_loss[N].avg, step)
            # writer.add_scalars("train/loss", {str(k): v.avg for (k, v) in loss.items()}, step)
            # writer.add_scalars("train/mse_loss", {str(k): v.avg for (k, v) in mse_loss.items()}, step)
            # writer.add_scalars("train/psnr", {str(k): -10 * math.log10(v.avg) for (k, v) in mse_loss.items()}, step)
            # writer.add_scalars("train/bpp_loss", {str(k): v.avg for (k, v) in bpp_loss.items()}, step)
            writers[-1].add_scalar("train/aux_loss", aux_loss.avg, step)
            writers[-1].add_scalar("train/ra_loss", ra_loss.avg, step)
            writers[-1].add_scalar("train/ra_reg_mae", ra_reg_mae.avg, step)
            writers[-1].add_scalar("train/ra_action_mae", ra_action_mae.avg, step)
            writers[-1].add_scalar("train/ra_action_acc", ra_action_acc.avg, step)
            writers[-1].add_scalar("train/ra_action_deg", ra_action_deg.avg, step)
            writers[-1].add_scalar("train/ra_reg_action_mae", ra_reg_action_mae.avg, step)
            writers[-1].add_scalar("train/ra_reg_action_acc", ra_reg_action_acc.avg, step)
            writers[-1].add_scalar("train/ra_reg_action_deg", ra_reg_action_deg.avg, step)
            if teacher:
                writers[-1].add_scalar("train/ra_action_kl", ra_action_kl.avg, step)


def test_epoch(epoch, test_dataloader, model, routing_agent,
               criterion, patch_criterion, ra_criterion,
               N_list, lmbda_list, writers=None):
    # get loader len
    total_samples = test_dataloader._size
    total_batches = int(math.ceil(test_dataloader._size / args.test_batch_size))

    model.eval()
    routing_agent.eval()
    device = next(model.parameters()).device

    loss = collections.defaultdict(AverageMeter)
    mse_loss = collections.defaultdict(AverageMeter)
    bpp_loss = collections.defaultdict(AverageMeter)
    aux_loss = AverageMeter()
    ra_loss = AverageMeter()
    ra_reg_mae = AverageMeter()
    ra_action_mae, ra_action_acc, ra_action_deg = AverageMeter(), AverageMeter(), AverageMeter()
    ra_reg_action_mae, ra_reg_action_acc, ra_reg_action_deg = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d[0]["data"]
            # d = d.to(device)
            # d = test_transforms_gpu(d)
            patch_costs = torch.zeros((len(N_list), d.shape[0]), device=device)
            for j, width in enumerate(N_list):
                model.set_running_width(width)
                out_net = model(d)
                out_criterion = criterion(out_net, d, lmbda_list[j])
                bpp_loss[width].update(out_criterion["bpp_loss"])
                loss[width].update(out_criterion["loss"])
                mse_loss[width].update(out_criterion["mse_loss"])
                # patch rd, as labels for the routing agent
                out_patch_criterion = patch_criterion(out_net, d, cur_lmbda=args.rcost_lambda)
                patch_costs[j] = out_patch_criterion["cost"]
            # values to deltas
            rcosts = torch.zeros((d.shape[0], len(N_list)-1), device=device)
            for j in range(len(N_list)-1):
                rcosts[:, j] = patch_costs[j] - patch_costs[-1]
            aux_loss.update(model.aux_loss())
            out_ra = {}
            out_ra["cost"], out_ra["decision"] = routing_agent(d)
            out_ra_criterion = ra_criterion(out_ra, rcosts, epsilon=args.epsilon)
            ra_loss.update(out_ra_criterion["loss"].item())
            ra_reg_mae.update(out_ra_criterion["reg_mae"].item())
            ra_action_mae.update(out_ra_criterion["action_mae"].item())
            ra_action_acc.update(out_ra_criterion["action_acc"].item())
            ra_action_deg.update(out_ra_criterion["action_deg"].item())
            ra_reg_action_mae.update(out_ra_criterion["reg_action_mae"].item())
            ra_reg_action_acc.update(out_ra_criterion["reg_action_acc"].item())
            ra_reg_action_deg.update(out_ra_criterion["reg_action_deg"].item())

    full_width = N_list[-1]
    logging.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss[full_width].avg:.3f} |"
        f"\tPSNR: {-10 * math.log10(mse_loss[full_width].avg):.2f} |"
        f"\tBpp loss: {bpp_loss[full_width].avg:.3f} |"
        f"\tAux loss: {aux_loss.avg:.2f}"
    )

    logging.info(
        f"Test epoch {epoch}: Average losses:"
        f'\tRA Loss: {ra_loss.avg:.3f} |'
        f'\tCost Reg MAE: {ra_reg_mae.avg:.3f} |'
        f"\tAction MAE: {ra_action_mae.avg:.3f} |"
        f'\tAction Acc: {ra_action_acc.avg * 100:.2f}% |'
        f'\tAction Deg: {ra_action_deg.avg * 100:.2f}%'
    )

    logging.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tReg Action MAE: {ra_reg_action_mae.avg:.3f} |"
        f'\tReg Action Acc: {ra_reg_action_acc.avg * 100:.2f}% |'
        f'\tReg Action Deg: {ra_reg_action_deg.avg * 100:.2f}%'
    )

    j_pred = out_ra["decision"].argmax(dim=-1).type(torch.float32)
    j_true = out_ra_criterion["jusm_true"].type(torch.float32)
    logging.info(
        f"Test epoch {epoch}: (std, min, avg, max)"
        f'\tPred: ({j_pred.std():.3f}, {j_pred.min():.3f}, {j_pred.mean():.3f}, {j_pred.max():.3f}) |'
        f'\tGT: ({j_true.std():.3f}, {j_true.min():.3f}, {j_true.mean():.3f}, {j_true.max():.3f})\n'
    )

    step = (epoch+1)*total_batches
    for i, N in enumerate(N_list):
        writers[i].add_scalar("test/loss", loss[N].avg, step)
        writers[i].add_scalar("test/mse_loss", mse_loss[N].avg, step)
        writers[i].add_scalar("test/psnr", -10 * math.log10(mse_loss[N].avg), step)
        writers[i].add_scalar("test/bpp_loss", bpp_loss[N].avg, step)
    writers[-1].add_scalar("test/aux_loss", aux_loss.avg, step)
    writers[-1].add_scalar("test/ra_loss", ra_loss.avg, step)
    writers[-1].add_scalar("test/ra_reg_mae", ra_reg_mae.avg, step)
    writers[-1].add_scalar("test/ra_action_mae", ra_action_mae.avg, step)
    writers[-1].add_scalar("test/ra_action_acc", ra_action_acc.avg, step)
    writers[-1].add_scalar("test/ra_action_deg", ra_action_deg.avg, step)
    writers[-1].add_scalar("test/ra_reg_action_mae", ra_reg_action_mae.avg, step)
    writers[-1].add_scalar("test/ra_reg_action_acc", ra_reg_action_acc.avg, step)
    writers[-1].add_scalar("test/ra_reg_action_deg", ra_reg_action_deg.avg, step)

    # writer.add_scalars("test/loss", {str(k): v.avg for (k, v) in loss.items()}, step)
    # writer.add_scalars("test/mse_loss", {str(k): v.avg for (k, v) in mse_loss.items()}, step)
    # writer.add_scalars("test/psnr", {str(k): -10 * math.log10(v.avg) for (k, v) in mse_loss.items()}, step)
    # writer.add_scalars("test/bpp_loss", {str(k): v.avg for (k, v) in bpp_loss.items()}, step)
    # writer.add_scalar("test/aux_loss", aux_loss.avg, step)

    return ra_loss.avg


def save_checkpoint(state, is_best, is_interval, path="./", filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, "model_best_loss.pth.tar"))
    if is_interval:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, "model_{}.pth.tar".format(state["epoch"])))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-N",
        "--N-list",
        dest="N_list",
        type=int,
        nargs="*",
        default=[64, 128, 192],
        help="architecture hyper-parameter N of the supernet (default: %(default)s)"
    ),
    parser.add_argument(
        "-M",
        "--M-list",
        dest="M_list",
        type=int,
        nargs="*",
        default=[128, 192, 320],
        help="architecture hyper-parameter M of the supernet (default: %(default)s)"
    ),
    parser.add_argument(
        "-L",
        "--L-list",
        dest="lmbda_list",
        type=float,
        nargs="*",
        default=[1000] * 3,
        help="Lambda list for balancing rate-distortion loss (default: %(default)s)"
    ),
    parser.add_argument(
        "-RL",
        "--rcost-lambda",
        type=float,
        default=11705,
        help="Lambda in the routing cost equation (default: %(default)s)"
    ),
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=2e-2,
        help="Maximum routing cost degradation (default: %(default)s)"
    ),
    parser.add_argument(
        "-G",
        "--gamma",
        type=float,
        default=0.5,
        help="Balancing factor gamma in RA action loss, the proportion of JUSM CE loss (default: %(default)s)"
    ),
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=12000,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-se",
        "--save-interval",
        default=1000,
        type=int,
        help="Checkpoint interval (in number of epochs, default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate for CAE (default: %(default)s)",
    )
    parser.add_argument(
        "-rlr",
        "--ra-learning-rate",
        default=1e-3,
        type=float,
        help="Learning rate for agent (default: %(default)s)",
    )
    parser.add_argument(
        "-lrda",
        "--lr-decay-after",
        default=0,
        type=int,
        help="Apply learning rate decay after some steps (in number of epochs, default: %(default)s)"
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "-tb",
        "--test-batch-size",
        type=int,
        default=256,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "-alr",
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--dali-cpu", action="store_true", help="Use CPU backend of DALI")
    parser.add_argument(
        "-arl",
        "--action_reg_loss", action="store_true", help="Use MSE as action loss"
    )
    parser.add_argument(
        "-alp",
        "--action_loss_penalty", type=float, default=1.0, help="Action loss penalty multiplier"
    )
    parser.add_argument(
        "--mbv3-small",
        action="store_true", help="Use MBV3-Small"
    )
    parser.add_argument(
        "-iha",
        "--in-house",
        action="store_true", help="Use in-house agent architecture"
    )
    parser.add_argument(
        "-mlp",
        action="store_true", help="Use mlp agent architecture"
    )
    parser.add_argument(
        "--low-res",
        action="store_true", help="Use Downsampled Input"
    )
    parser.add_argument(
        "-radp",
        action="store_true", default=True, help="Use DataParallel on RA (default: off, for compatability)"
    )
    # parser.add_argument(
    #     "-kd",
    #     "--distill",
    #     action="store_true", help="Use KDLoss for student training"
    # )
    parser.add_argument(
        "--teacher", type=str, help="Path to RA teacher checkpoint"
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(args, writers, ts):
    # log arguments
    logging.info(args)

    # initialize random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    traindir = os.path.join(args.dataset, 'train')
    testdir = os.path.join(args.dataset, 'test')
    crop_size = args.patch_size

    logging.info("Using NVIDIA DALI for data loading and pre-processing. (Backend: {})".format(
        "cpu" if args.dali_cpu else "mixed"
    ))
    # train
    pipe = create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.num_workers,
                                device_id=0,  # pipeline on a single gpu
                                seed=args.seed,
                                data_dir=traindir,
                                crop=crop_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=0,
                                num_shards=1,
                                is_training=True)
    pipe.build()
    train_dataloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    # test
    pipe = create_dali_pipeline(batch_size=args.test_batch_size,
                                num_threads=args.num_workers,
                                device_id=0,  # pipeline on a single gpu
                                seed=args.seed,
                                data_dir=testdir,
                                crop=crop_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=0,
                                num_shards=1,
                                is_training=False)
    pipe.build()
    test_dataloader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    # initialize models
    N_list = args.N_list
    M_list = args.M_list
    lmbda_list = args.lmbda_list
    logging.info("N_list: {}".format(str(N_list)))
    logging.info("M_list: {}".format(str(M_list)))
    logging.info("\lambda_list: {}".format(str(lmbda_list)))
    net = AdaMeanScaleHyperprior(N_list=N_list, M_list=M_list)
    net = net.to(device)
    if args.in_house:
        ra = RoutingAgent(pred_delta=True, use_mbv3=None, downsample=args.low_res).to(device)
    elif args.mlp:
        ra = RoutingAgent(pred_delta=True, use_mlp=True, downsample=False).to(device)
    else:
        ra = RoutingAgent(pred_delta=True, use_mbv3="large" if not args.mbv3_small else "small",
                          downsample=args.low_res).to(device)
    teacher = None
    if args.teacher:
        assert args.teacher is not None
        teacher = RoutingAgent(pred_delta=True, use_mbv3="large", downsample=False).to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
        if args.radp:
            ra = CustomDataParallel(ra)
            if args.teacher:
                teacher = CustomDataParallel(teacher)

    # initialize optimizers
    optimizer, aux_optimizer = configure_optimizers(net, args)
    ra_optimizer = optim.Adam(ra.parameters(), lr=args.ra_learning_rate)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=200, factor=0.5, min_lr=2.50e-5)
    ra_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ra_optimizer, "min", patience=4000, factor=0.5, min_lr=1e-4)

    # initialize loss func.
    criterion = RateDistortionLoss()
    patch_criterion = SampleCost()
    ra_criterion = RoutingLoss(gamma=args.gamma,
                               action_reg_loss=args.action_reg_loss, action_loss_penalty=args.action_loss_penalty)

    # load checkpoint
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "ra_state_dict" in checkpoint.keys():
            ra.load_state_dict(checkpoint["ra_state_dict"])
            ra_optimizer.load_state_dict(checkpoint["ra_optimizer"])
        if "ra_lr_scheduler" in checkpoint.keys():
            ra_lr_scheduler.load_state_dict(checkpoint["ra_lr_scheduler"])
        if args.teacher:
            tea_ckpt = torch.load(args.teacher, map_location=device)
            if hasattr(teacher, "module"):
                teacher.module.load_state_dict(tea_ckpt["ra_state_dict"])
            else:
                teacher.load_state_dict(tea_ckpt["ra_state_dict"])

    # main loop
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info(f"RA Learning rate: {ra_optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            ra,
            criterion,
            patch_criterion,
            ra_criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            ra_optimizer,
            epoch,
            args.clip_max_norm,
            N_list,
            lmbda_list,
            writers=writers,
            teacher=teacher
        )
        loss = test_epoch(epoch, test_dataloader, net, ra,
                          criterion, patch_criterion, ra_criterion,
                          N_list, lmbda_list, writers=writers)

        train_dataloader.reset()
        test_dataloader.reset()

        # lr_scheduler.step(loss)
        if epoch >= args.lr_decay_after:
            ra_lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "ra_state_dict": ra.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "ra_optimizer": ra_optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "ra_lr_scheduler": ra_lr_scheduler.state_dict()
                },
                is_interval=(epoch % args.save_interval == 0),
                is_best=is_best,
                path="./logs/ra/{}".format(ts)
            )
    [writer.close() for writer in writers]


if __name__ == "__main__":
    # parse arguments
    args = parse_args(sys.argv[1:])
    N_list = args.N_list
    # initialize logger
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    writers = [tensorboardX.SummaryWriter(logdir="./logs/ra/{}/w{}".format(ts, N)) for N in N_list]
    logging.basicConfig(level=logging.INFO, filename="./logs/ra/{}/train.log".format(ts),
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    log_formatter = logging.Formatter("%(asctime)-15s [%(levelname)-8s] %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(console_handler)

    main(args, writers, ts)
