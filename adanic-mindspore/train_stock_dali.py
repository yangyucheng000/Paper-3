import argparse
import math
import random
import shutil
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

# from torchvision.datasets import ImageFolder
from compressai.datasets import ImageFolder
from compressai.zoo import models
from compressai.models import MeanScaleHyperprior
#from models.slim_models import AdaMeanScaleHyperprior

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

    def forward(self, output, target, cur_lmbda=None, clamp=False):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"].clamp(0, 1) if clamp else output["x_hat"], target)
        if not cur_lmbda:
            out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
        else:
            out["loss"] = cur_lmbda * out["mse_loss"] + out["bpp_loss"]

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    # get loader len
    total_samples = train_dataloader._size
    total_batches = int(math.ceil(train_dataloader._size / args.batch_size))

    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d[0]["data"]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*args.batch_size+len(d)}/{total_samples}"
                f" ({100. * (i+1) / total_batches:.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                f'\tPSNR (dB): {-10 * math.log10(out_criterion["mse_loss"].item()):.2f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.3f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    # get loader len
    total_samples = test_dataloader._size
    total_batches = int(math.ceil(test_dataloader._size / args.test_batch_size))

    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d[0]["data"]
            out_net = model(d)
            out_criterion = criterion(out_net, d, clamp=True)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f'\tPSNR (dB): {-10 * math.log10(mse_loss.avg):.2f} |'
        f"\tBpp loss: {bpp_loss.avg:.3f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        dest="quality",
        type=int,
        default=8,
        help="Model's quality factor (default: %(default)s)",
    )
    parser.add_argument(
        "-pt",
        "--pretrained",
        action="store_true", default=True, help="Load pre-trained weights"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
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
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
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


def main(args):

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    traindir = os.path.join(args.dataset, 'train')
    testdir = os.path.join(args.dataset, 'test')
    crop_size = args.patch_size

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

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = models[args.model](quality=args.quality, metric='mse', pretrained=args.pretrained)
    net = net.to(device)
     
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")

    if args.test:
        test_epoch(-1, test_dataloader, net, criterion)
        return

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        train_dataloader.reset()
        test_dataloader.reset()

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
                is_best,
            )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
