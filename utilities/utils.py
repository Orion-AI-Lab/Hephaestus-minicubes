import datetime
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall, AUROC
from torchmetrics.classification import  AveragePrecision
import math
import matplotlib.pyplot as plt
import numpy as np
from losses.focal_loss import FocalLoss
import psutil
import utilities.webdataset_utils as wds_utils
from utilities import augmentations
import einops 
import webdataset as wds
import io
import random
import albumentations as A
import glob
from torch.utils.data import IterableDataset

from dataset.dataset_utils import InSarDataset


def print_memory_info():
    memory_info = psutil.virtual_memory()
    print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Memory percentage: {memory_info.percent}%")


def get_wandb_checkpoint(configs, wandb_id):
    wandb_id = wandb_id.split('/')[-1]
    latest_checkpoint = None
    checkpoint_dir = Path(configs['checkpoint_path']).parent.absolute()
    files = os.listdir(checkpoint_dir)
    files = [os.path.join(checkpoint_dir, f) for f in files] # add path to each file
    
    for file in files:
        if wandb_id in file:
            latest_checkpoint = file
            break
        
    return latest_checkpoint


def ensure_dirs_exist(path):
    """
    Ensure that directories leading to the given path exist.
    If they don't exist, create them.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def initialize_criterion(configs):
    weights = [None, None]
    if configs['num_classes']==2:
        if configs['weighted_loss']:
            raise NotImplementedError('Weighted Loss is not implemented yet')
        if configs['loss_criterion'].lower() == 'crossentropyloss':
            criterion = nn.CrossEntropyLoss()
        elif configs['loss_criterion'].lower() == 'focalloss':
            criterion = FocalLoss(gamma=2,alpha=0.25)
        else:
            raise NotImplementedError(f"{configs['loss_criterion']} is not implemented")
    else:
        criterion = nn.CrossEntropyLoss()
    
    if configs['wandb'] and configs['weighted_loss']:
        configs['loss_weights'] = weights
        configs.update(configs)
    else:
        print('Weights:',weights)
    return criterion 


def create_checkpoint_directory(args, wandb_run=None):
    if wandb_run is None:
        if 'num_classes' not in args:
            checkpoint_path = (
                Path("checkpoints")
                / args["method"].lower()
                / args["architecture"].lower()
                / f'{args["architecture"].lower()}_{str(args["resolution"])}'
                / str(datetime.datetime.now())
            )
        elif args['timeseries']:
            checkpoint_path = (
                Path("checkpoints")
                / "supervised"
                / args["architecture"].lower()
                / str(args["timeseries_length"])
                / str(datetime.datetime.now())
            )
        else:
            checkpoint_path = (
                Path("checkpoints")
                / "supervised"
                / args["architecture"].lower()
                / str(datetime.datetime.now())
            )
    else:
        if 'num_classes' not in args:
            checkpoint_path = (
                Path("checkpoints")
                / args["method"].lower()
                / args["architecture"].lower()
                / f'{args["architecture"].lower()}_{str(args["resolution"])}'
                / str(str(wandb_run.id))
            )
        elif args['timeseries']:
            checkpoint_path = (
                Path("checkpoints")
                / "supervised"
                / args["architecture"].lower()
                / str(args["timeseries_length"])
                / str(wandb_run.id)
            )
        else:
            checkpoint_path = (
                Path("checkpoints")
                / "supervised"
                / args["architecture"].lower()
                / str(str(wandb_run.id))
            )
            
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def prepare_configuration(path):
    # Load configuration files
    base_cfg = json.load(open(path, "r"))
    if not base_cfg["seed"]:
        base_cfg["seed"] = None

    augmentation_cfg = json.load(open(base_cfg["augmentation_config"], "r"))
    base_cfg.update(augmentation_cfg)

    model_cfg = (
        Path("configs/method")
        / base_cfg["method"].lower()
        / f'{base_cfg["method"].lower()}.json'
    )
    with model_cfg.open("r", encoding="UTF-8") as target:
        model_config = json.load(target)
    base_cfg.update(model_config)

    # Create checkpoint path if it does not exist
    checkpoint_path = create_checkpoint_directory(base_cfg)
    base_cfg["checkpoint_path"] = checkpoint_path.as_posix()

    return base_cfg


def get_dataset(config, mode="train", webdataset_write= False, verbose=False):
    return InSarDataset(config, mode, verbose=verbose, webdataset_write=webdataset_write)


def prepare_supervised_learning_loaders(configs, verbose):
    if configs['webdataset']:
        train_loader, val_loader, test_loader = create_webdataset_loaders(configs)
        return train_loader, val_loader, test_loader
    else:
        print('Creating Dataset loaders on zarr files...')
        train_dataset = get_dataset(config=configs, mode='train', verbose=verbose)
        val_dataset = get_dataset(config=configs, mode='val', verbose=verbose)
        test_dataset = get_dataset(config=configs, mode='test', verbose=verbose)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=configs['num_workers'], pin_memory=True,
                                        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=configs['num_workers'], pin_memory=True,
                                    drop_last=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs['batch_size'], shuffle=False, num_workers=configs['num_workers'], pin_memory=True,
                                    drop_last=False)
    
    return train_loader, val_loader, test_loader


def initialize_metrics(configs):
    if configs['task'] == 'segmentation':
        accuracy = Accuracy(task='multiclass', average=configs['metric_strategy'], multidim_average='global', num_classes=configs['num_classes']).to(configs['device'])
        fscore = F1Score(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy'], multidim_average='global').to(configs['device'])
        precision = Precision(task='multiclass', average=configs['metric_strategy'], num_classes=configs['num_classes'], multidim_average='global').to(configs['device'])
        recall = Recall(task='multiclass', average=configs['metric_strategy'], num_classes=configs['num_classes'], multidim_average='global').to(configs['device'])
        avg_precision = AveragePrecision(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy'], thresholds=5).to(configs['device'])
        #auroc = AUROC(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy']).to(configs['device'])
        iou = JaccardIndex(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy']).to(configs['device'])

        metrics = [accuracy, fscore, precision, recall, avg_precision, iou]
        return metrics
    elif configs['task'] == 'classification':
        accuracy = Accuracy(task='multiclass', average=configs['metric_strategy'], multidim_average='global', num_classes=configs['num_classes']).to(configs['device'])
        fscore = F1Score(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy'], multidim_average='global').to(configs['device'])
        precision = Precision(task='multiclass', average=configs['metric_strategy'], num_classes=configs['num_classes'], multidim_average='global').to(configs['device'])
        recall = Recall(task='multiclass', average=configs['metric_strategy'], num_classes=configs['num_classes'], multidim_average='global').to(configs['device'])
        avg_precision = AveragePrecision(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy'], thresholds=5).to(configs['device'])
        auroc = AUROC(task='multiclass', num_classes=configs['num_classes'], average=configs['metric_strategy']).to(configs['device'])

        metrics = [accuracy, fscore, precision, recall, avg_precision, auroc]
        return metrics


def world_info_from_env():
    local_rank = 0
    for v in ("LOCAL_RANK", "SLURM_LOCALID"):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "SLURM_PROCID"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "SLURM_NTASKS"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break
    return local_rank, global_rank, world_size


def is_global_master(args):
    return args["rank"] == 0


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def load_checkpoint(model, optimizer, args):
    if os.path.isfile(args["resume_checkpoint"]):
        print("=> loading checkpoint '{}'".format(args["resume_checkpoint"]))
        checkpoint = torch.load(args["resume_checkpoint"], map_location="cpu")
        args["start_epoch"] = checkpoint["epoch"]

        for key in list(checkpoint["state_dict"].keys()):
            checkpoint["state_dict"][key.replace("module.", "")] = checkpoint[
                "state_dict"
            ][key]
            del checkpoint["state_dict"][key]

        msg = model.load_state_dict(checkpoint["state_dict"])
        print(msg)

        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args["resume_checkpoint"], checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(args["resume_checkpoint"]))


def extract_state_dict_from_ddp_checkpoint(checkpoint_path):
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder_state_dict = {}
    for key in list(checkpoint["state_dict"].keys()):
        checkpoint["state_dict"][key.replace("module.", "")] = checkpoint[
            "state_dict"
        ][key]

        del checkpoint["state_dict"][key]
        if 'encoder_q' in key and 'fc' not in key:
            encoder_state_dict[key.replace("module.encoder_q.","")] = checkpoint['state_dict'][key.replace("module.","")]

    return encoder_state_dict
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args["lr"]
    if args["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["epochs"]))
    else:  # stepwise lr schedule
        for milestone in args["schedule"]:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct[:k].shape)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = (
                correct[:k].reshape(k * correct.shape[1]).float().sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs["warmup_epochs"]:
        lr = configs["lr"] * epoch / configs["warmup_epochs"] 
    else:
        lr = configs["min_lr"] + (configs["lr"] - configs["min_lr"]) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - configs["warmup_epochs"]) / (configs["epochs"] - configs["warmup_epochs"])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize(image_timeseries, config, statistics_path="statistics.json"):
    """
    Normalize only the channels present in the image, based on the config.
    Handles geomorphology and atmospheric channels in primary/secondary pairs.

    Args:
        image_timeseries (Tensor): shape [T, C, H, W]
        config (dict): includes 'geomorphology_channels' and 'atmospheric_channels'
        statistics_path (str): JSON file with per-channel stats

    Returns:
        Tensor: normalized image_timeseries
    """
    statistics = json.load(open(statistics_path, "r"))

    means = []
    stds = []

    # 1. Geomorphology channels come first
    for ch in config["geomorphology_channels"]:
        if ch in statistics:
            means.append(statistics[ch]["mean"])
            stds.append(statistics[ch]["std"])
        else:
            print(f"Missing stats for geomorph channel: {ch}")

    # 2. Atmospheric channels in primary-secondary pairs
    for ch in config["atmospheric_channels"]:
        if ch in statistics:
            # Add twice: once for primary, once for secondary
            means.extend([statistics[ch]["mean"], statistics[ch]["mean"]])
            stds.extend([statistics[ch]["std"], statistics[ch]["std"]])
        else:
            print(f"Missing stats for atmospheric variable: {ch}")

    # Apply normalization over each channel
    for i in range(len(means)):
        image_timeseries[:, i, :, :] = (image_timeseries[:, i, :, :] - means[i]) / stds[i]

    return image_timeseries


def augment(augmentations, insar_timeseries, mask_timeseries):    
    """Augment the image with the specified augmentations."""
    if not isinstance(insar_timeseries, np.ndarray):
        insar_timeseries = insar_timeseries.numpy()
    if not isinstance(mask_timeseries, np.ndarray):
        mask_timeseries = mask_timeseries.numpy()
    timeseries_length = insar_timeseries.shape[0]

    # Rearrange bands and masks to match the expected format
    # Timestep is given as different channels
    bands = einops.rearrange(insar_timeseries, "t c h w -> h w (c t)")
    masks = einops.rearrange(mask_timeseries, "t c h w -> h w (c t)")

    transform = augmentations(image=bands, mask=masks)
    augmented_bands = transform["image"]
    augmented_masks = transform["mask"]

    # Split time (T) back from channel (C*T)
    augmented_bands = einops.rearrange(augmented_bands, "h w (c t) -> t c h w", t=timeseries_length)
    augmented_masks = einops.rearrange(augmented_masks, "h w (c t) -> t c h w", t=timeseries_length)

    # If not tensor convert to tensor
    if not isinstance(augmented_bands, torch.Tensor):
        augmented_bands = torch.tensor(augmented_bands)
    if not isinstance(augmented_masks, torch.Tensor):
        augmented_masks = torch.tensor(augmented_masks)

    return augmented_bands, augmented_masks


def crop_around_deformation(
    insar,
    target_mask_method="union",
    crop_size=512,
):
    """
    Crop around deformation ensuring the target mask is fully contained,
    avoiding areas with no data (zeros) and performing a somewhat random crop.

    Args:
        insar (torch.Tensor): Input tensor of shape (T, C, H, W).
        crop_size (int): Desired crop size.
        target_mask_method (str): Method to select the target mask. Options: 'last', 'peak', 'union'.

    Returns:
        torch.Tensor: Cropped tensor of shape (T, C, crop_size, crop_size).
    """
    T, C, H, W = insar.shape

    # Ensure crop size is not larger than the image dimensions
    crop_size = min(crop_size, H, W)

    # Step 1: Select the target mask
    if target_mask_method == "last":
        target_mask = insar[-1, 2, :, :]
    elif target_mask_method == "peak":
        target_mask = insar[:, 2, :, :].sum(dim=0)  # Sum across time to get the peak mask
    elif target_mask_method == "union":
        target_mask = (insar[:, 2, :, :] > 0).any(dim=0).float()  # Union of all masks
    else:
        raise ValueError(f"Invalid target_mask_method: {target_mask_method}")

    # Ensure target_mask is binary (0 or 1)
    target_mask = (target_mask > 0).float()

    # Step 2: Find bounding box around the target mask
    if target_mask.max() > 0:  # If deformation exists
        coords = torch.nonzero(target_mask, as_tuple=False)  # Get coordinates of mask=1 regions
        min_y, min_x = coords.min(dim=0).values
        max_y, max_x = coords.max(dim=0).values
    else:  # No deformation, center the crop randomly
        min_y, min_x, max_y, max_x = 0, 0, H - 1, W - 1

    # Step 3: Compute crop center with randomness
    center_y = (min_y + max_y) // 2
    center_x = (min_x + max_x) // 2

    # Add small randomness to the center, ensuring we stay within bounds
    rand_y = torch.randint(-crop_size // 4, crop_size // 4 + 1, (1,)).item()
    rand_x = torch.randint(-crop_size // 4, crop_size // 4 + 1, (1,)).item()
    center_y = max(crop_size // 2, min(H - crop_size // 2, center_y + rand_y))
    center_x = max(crop_size // 2, min(W - crop_size // 2, center_x + rand_x))

    # Step 4: Perform cropping
    start_y = center_y - crop_size // 2
    start_x = center_x - crop_size // 2
    end_y = start_y + crop_size
    end_x = start_x + crop_size

    cropped_insar = insar[:, :, start_y:end_y, start_x:end_x]

    # Step 5: Validate crop contains valuable information
    assert cropped_insar.shape[-2:] == (crop_size, crop_size), "Crop dimensions are incorrect."
    assert cropped_insar[:, -1, :, :].max() == target_mask.max(), "Crop does not fully contain the target mask."

    return cropped_insar


def create_webdataset_loaders(configs, repeat=False, resample_shards=False):
    random.seed(configs['seed'])
    np.random.seed(configs['seed'])

    all_channels = [
        "insar_difference", "insar_coherence", "dem", 
        "primary_date_total_column_water_vapour", "secondary_date_total_column_water_vapour",
        "primary_date_surface_pressure", "secondary_date_surface_pressure",
        "primary_date_vertical_integral_of_temperature", "secondary_date_vertical_integral_of_temperature"
        ]

    def get_channel_indices(channel_list, all_channels, is_atmospheric=False):
        indices = []
        for channel in channel_list:
            if is_atmospheric:
                prim = f"primary_date_{channel}"
                sec = f"secondary_date_{channel}"
                if prim in all_channels:
                    indices.append(all_channels.index(prim))
                else:
                    print(f"Warning: {prim} not in all_channels")
                if sec in all_channels:
                    indices.append(all_channels.index(sec))
                else:
                    print(f"Warning: {sec} not in all_channels")
            else:
                if channel in all_channels:
                    indices.append(all_channels.index(channel))
                else:
                    print(f"Warning: {channel} not in all_channels")
        return indices
    
    geomorphology_indices = get_channel_indices(configs['geomorphology_channels'], all_channels)
    atmospheric_indices = get_channel_indices(configs['atmospheric_channels'], all_channels, is_atmospheric=True)

    def get_patches(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            sample = torch.load(io.BytesIO(sample["sample.pth"]))

            if label.ndim == 3:
                label = label[:, None, :, :]

            if isinstance(label, dict):
                label = label["label"]

            image = image.reshape(configs['timeseries_length'], len(all_channels), configs['image_size'], configs['image_size'])
            label = label.reshape(configs['timeseries_length'], 1, configs['image_size'], configs['image_size'])
            
            # Select only the relevant geomorphology channels and atmospheric channels
            selected_channels = geomorphology_indices + atmospheric_indices
            image = image[:, selected_channels, :, :]  # Keep only the selected channels

            if configs["augment"] == True:
                data_augmentations = augmentations.get_augmentations(configs, configs['image_size'], configs['seed'])
                image, label = augment(data_augmentations, image, label)

            image = normalize(image, configs)
            #produce_variable_figures_from_tensor(torch.cat([image, label], dim=1), all_channels, sample)

            if configs['task'] == 'segmentation':
                if configs['timeseries_length'] != 1:
                    if configs['mask_target'] == 'peak':
                        counts = torch.sum(label, dim=(2, 3))
                        label = label[torch.argmax(counts), :, :, :]
                    elif configs['mask_target'] == 'union':
                        label = torch.sum(label, dim=0)
                        label = torch.where(label > 0, 1, 0)
                    elif configs['mask_target'] == 'last':
                        label = label[-1, :, :, :]
                else:
                    label = label[-1, :, :, :] # Last channel
            else:
                if configs['mask_target'] == 'union':
                    label = torch.tensor(int(np.any(sample['label'])==1))
                elif configs['mask_target'] == 'last':
                    label = torch.tensor(int(sample['label'][-1]==1))

            image = image.reshape(configs['timeseries_length']*(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels'])), configs['image_size'], configs['image_size'])

            if configs['task'] == 'segmentation':
                label = label.reshape(configs['image_size'], configs['image_size'])

            yield (image, label, sample)


    def get_patches_eval(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"])).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]))
            sample = torch.load(io.BytesIO(sample["sample.pth"]))
            if isinstance(label, dict):
                label = label["label"]

            image = image.reshape(configs['timeseries_length'], len(all_channels), configs['image_size'], configs['image_size'])
            label = label.reshape(configs['timeseries_length'], 1, configs['image_size'], configs['image_size'])

            # Select only the relevant geomorphology channels and atmospheric channels
            selected_channels = geomorphology_indices + atmospheric_indices
            image = image[:, selected_channels, :, :]  # Keep only the selected channels

            image = normalize(image, configs)
            
            if configs['task'] == 'segmentation':
                if configs['timeseries_length'] != 1:
                    if configs['mask_target'] == 'peak':
                        counts = torch.sum(label, dim=(2, 3))
                        label = label[torch.argmax(counts)]
                    elif configs['mask_target'] == 'last':
                        label = label[-1, :, :, :]
                    elif configs['mask_target'] == 'union':
                        label = torch.sum(label, dim=0)
                        label = torch.where(label > 0, 1, 0)
                else:
                    label = label[-1, :, :, :] # Last channel
            else:
                label = torch.tensor(int(np.any(sample['label'])==1))

            image = image.reshape(configs['timeseries_length']*(len(configs['geomorphology_channels'])+2*len(configs['atmospheric_channels'])), configs['image_size'], configs['image_size'])

            if configs['task'] == 'segmentation':
                if configs['mask_target'] == 'all':
                    label = label.reshape(configs['timeseries_length'], configs['image_size'], configs['image_size'])
                else:
                    label = label.reshape(configs['image_size'], configs['image_size'])

            yield (image, label, sample)

    configs["webdataset_path"] = os.path.join(configs["webdataset_root"], str(configs['timeseries_length']))

    for mode in ["train", "val", "test"]:
        if mode == "train":
            if not os.path.isdir(os.path.join(configs["webdataset_path"], 'train_pos')) or not os.path.isdir(os.path.join(configs["webdataset_path"], 'train_neg')):
                wds_utils.wds_write_parallel(configs, mode)
                print("Created webdataset for: ", mode)
                exit(1)
        else:
            if not os.path.isdir(os.path.join(configs["webdataset_path"], mode)):
                wds_utils.wds_write_parallel(configs, mode)
                print("Created webdataset for: ", mode)
                exit(1)

    compress = configs.get("compress", False)
    ext = ".tar.gz" if compress else ".tar"

    max_train_pos_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train_pos", f"*{ext}")))[-1]
    max_train_pos_index = max_train_pos_shard.split("-train_pos-")[-1][:-4]
    max_train_neg_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train_neg", f"*{ext}")))[-1]
    max_train_neg_index = max_train_neg_shard.split("-train_neg-")[-1][:-4]
    max_train_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train_neg", f"*{ext}")))[-1]

    pos_train_shards = os.path.join(
        configs["webdataset_path"],
        "train_pos",
        "sample-train_pos-{000000.." + max_train_pos_index + "}"+ext,
    )
    neg_train_shards = os.path.join(
        configs["webdataset_path"],
        "train_neg",
        "sample-train_neg-{000000.." + max_train_neg_index + "}"+ext,
    )

    positives = wds.WebDataset(pos_train_shards, shardshuffle=True, resampled=False).shuffle(
        configs["webdataset_shuffle_size"]
    ).compose(get_patches)
    negatives = wds.WebDataset(neg_train_shards, shardshuffle=True, resampled=False).shuffle(
        configs["webdataset_shuffle_size"]
    ).compose(get_patches)

    #train_dataset = wds.RandomMix(datasets=[positives, negatives], probs=[1, 1])
    count_pos = len([iter(positives)])
    count_neg = len([iter(negatives)])
    train_dataset = RandomMix(datasets=[positives, negatives], probs=[1/count_pos, 1/count_neg])

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=False,
        prefetch_factor=configs["prefetch_factor"],
        persistent_workers=configs["persistent_workers"],
    ).shuffle(configs["webdataset_shuffle_size"]).batched(configs["batch_size"], partial=False)
    
    train_loader = (
        train_loader.unbatched()
        .shuffle(
            configs["webdataset_shuffle_size"],
            initial=configs["webdataset_initial_buffer"],
        )
        .batched(configs["batch_size"])
    )
    if repeat:
        train_loader = train_loader.repeat()

    max_val_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "val", f"*{ext}")))[-1]
    max_val_index = max_val_shard.split("-val-")[-1][:-4]
    val_shards = os.path.join(
        configs["webdataset_path"],
        "val",
        "sample-val-{000000.." + max_val_index + "}" + ext,
    )

    val_dataset = wds.WebDataset(val_shards, shardshuffle=False, resampled=False)
    val_dataset = val_dataset.compose(get_patches_eval)
    val_dataset = val_dataset.batched(configs["batch_size"], partial=True)

    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    max_test_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "test", f"*{ext}")))[-1]
    max_test_index = max_test_shard.split("-test-")[-1][:-4]
    test_shards = os.path.join(
        configs["webdataset_path"],
        "test",
        "sample-test-{000000.." + max_test_index + "}"+ext,
    )

    test_dataset = wds.WebDataset(test_shards, shardshuffle=False, resampled=False)
    test_dataset = test_dataset.compose(get_patches_eval)
    test_dataset = test_dataset.batched(configs["batch_size"], partial=True)

    test_loader = wds.WebLoader(
        test_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class RandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, probs=None, longest=False):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return random_samples(sources, self.probs, longest=self.longest)


def random_samples(sources, probs=None, longest=False):
    """Yield samples randomly from multiple sources based on given probabilities.

    Args:
        sources (list): List of iterable sources to draw samples from.
        probs (list, optional): List of probabilities for each source. Defaults to None.
        longest (bool): If True, continue until all sources are exhausted. Defaults to False.

    Yields:
        Sample randomly selected from one of the sources.
    """
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = random.random()
        i = np.searchsorted(cum, r)

        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break