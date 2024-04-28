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
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
from models.slim_models import AdaMeanScaleHyperprior

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
class SampleRateDistortion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = H * W
        with torch.no_grad():
            out["bpp"] = sum(
                (torch.log(likelihoods).sum(dim=[1,2,3]) / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()  # "y" & "z"
            )
            out["mse_loss"] = self.mse(output["x_hat"], target).mean(dim=[1,2,3])
            out["psnr"] = -10 * torch.log10(out["mse_loss"])
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
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, N_list, lmbda_list, writers=None
):
    # get loader len
    total_samples = train_dataloader._size
    total_batches = int(math.ceil(train_dataloader._size / args.batch_size))

    model.train()
    device = next(model.parameters()).device
    loss = collections.defaultdict(AverageMeter)
    mse_loss = collections.defaultdict(AverageMeter)
    bpp_loss = collections.defaultdict(AverageMeter)
    aux_loss = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d[0]["data"]
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        for j, width in enumerate(N_list):
            model.set_running_width(width)
            out_net = model(d)
            out_criterion = criterion(out_net, d, lmbda_list[j])
            loss[width].update(out_criterion["loss"].item())
            mse_loss[width].update(out_criterion["mse_loss"].item())
            bpp_loss[width].update(out_criterion["bpp_loss"].item())
            out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss_full = model.aux_loss()
        aux_loss_full.backward()
        aux_loss.update(aux_loss_full.item())
        aux_optimizer.step()


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
            writers[-1].add_scalar("train/lr", optimizer.param_groups[0]['lr'], step)


def test_epoch(epoch, test_dataloader, model, criterion, N_list, lmbda_list, writers=None):
    # get loader len
    total_samples = test_dataloader._size
    total_batches = int(math.ceil(test_dataloader._size / args.test_batch_size))

    model.eval()
    device = next(model.parameters()).device

    loss = collections.defaultdict(AverageMeter)
    mse_loss = collections.defaultdict(AverageMeter)
    bpp_loss = collections.defaultdict(AverageMeter)
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d[0]["data"]
            for j, width in enumerate(N_list):
                model.set_running_width(width)
                out_net = model(d)
                out_criterion = criterion(out_net, d, lmbda_list[j])
                bpp_loss[width].update(out_criterion["bpp_loss"])
                loss[width].update(out_criterion["loss"])
                mse_loss[width].update(out_criterion["mse_loss"])
            aux_loss.update(model.aux_loss())

    full_width = N_list[-1]
    logging.info(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss[full_width].avg:.3f} |"
        # f"\tMSE loss: {mse_loss[full_width].avg:.4f} |"
        f"\tPSNR: {-10 * math.log10(mse_loss[full_width].avg):.2f} |"
        f"\tBpp loss: {bpp_loss[full_width].avg:.3f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    step = (epoch+1)*total_batches
    if writers:
        for i, N in enumerate(N_list):
            writers[i].add_scalar("test/loss", loss[N].avg, step)
            writers[i].add_scalar("test/mse_loss", mse_loss[N].avg, step)
            writers[i].add_scalar("test/psnr", -10 * math.log10(mse_loss[N].avg), step)
            writers[i].add_scalar("test/bpp_loss", bpp_loss[N].avg, step)
        writers[-1].add_scalar("test/aux_loss", aux_loss.avg, step)

    # writer.add_scalars("test/loss", {str(k): v.avg for (k, v) in loss.items()}, step)
    # writer.add_scalars("test/mse_loss", {str(k): v.avg for (k, v) in mse_loss.items()}, step)
    # writer.add_scalars("test/psnr", {str(k): -10 * math.log10(v.avg) for (k, v) in mse_loss.items()}, step)
    # writer.add_scalars("test/bpp_loss", {str(k): v.avg for (k, v) in bpp_loss.items()}, step)
    # writer.add_scalar("test/aux_loss", aux_loss.avg, step)

    return loss[full_width].avg


def save_checkpoint(state, is_best, is_interval, path="./", filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, "model_best_loss.pth.tar"))
    if is_interval:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, "model_{}.pth.tar".format(state["epoch"])))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     default="mbt2018-mean",
    #     choices=models.keys(),
    #     help="Model architecture (default: %(default)s)",
    # )
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
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=4000,
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
        help="Learning rate (default: %(default)s)",
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
        default=8,
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
        "-lrda",
        "--lr-decay-after",
        default=0,
        type=int,
        help="Apply learning rate decay after some steps (in number of epochs, default: %(default)s)"
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
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test-only mode"
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

    # initialize dataloader
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
    
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    # initialize optimizers
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=200, factor=0.5, min_lr=2.50e-5)

    # initialize loss func.
    criterion = RateDistortionLoss()

    # load checkpoint
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        logging.info("Loading {}".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    if args.test:
        test_epoch(-1, test_dataloader, net, criterion, N_list, lmbda_list, writers=None)
        return

    # main loop
    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            N_list,
            lmbda_list,
            writers=writers
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion, N_list, lmbda_list, writers=writers)

        train_dataloader.reset()
        test_dataloader.reset()

        if epoch >= args.lr_decay_after:
            lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_interval=(epoch % args.save_interval == 0),
                is_best=is_best,
                path="./logs/{}".format(ts)
            )
    [writer.close() for writer in writers]


if __name__ == "__main__":
    # parse arguments
    args = parse_args(sys.argv[1:])
    N_list = args.N_list
    # initialize logger
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    writers = [tensorboardX.SummaryWriter(logdir="./logs/{}/w{}".format(ts, N)) for N in N_list]
    logging.basicConfig(level=logging.INFO, filename="./logs/{}/train.log".format(ts),
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    log_formatter = logging.Formatter("%(asctime)-15s [%(levelname)-8s] %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logging.getLogger().addHandler(console_handler)

    main(args, writers, ts)
