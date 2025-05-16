import os
from pathlib import Path
import io

import albumentations as A
import torch
import tqdm
import webdataset as wds
from itertools import product

import utilities.utils as utils
import ray

@ray.remote
def original_wds_write_ith_shard(configs, dataset, mode, i, n):
    shard_path = Path(os.path.join(configs["webdataset_path"], mode))
    shard_path.mkdir(parents=True, exist_ok=True)
    
    ext = ".tar"
    pattern = os.path.join(str(shard_path), f"sample-{mode}-{i}-%06d{ext}")

    with wds.ShardWriter(pattern, maxcount=configs["max_samples_per_shard"]) as sink:
        pos_count, neg_count = 0, 0
        for index in tqdm.tqdm(range(i, len(dataset), n)):
            batch = dataset[index]

            if configs['patch_size'] != configs['image_size']:
                assert configs['image_size'] % configs['patch_size'] == 0, \
                    f'Image size ({configs["image_size"]}) must be evenly divided by patch size ({configs["patch_size"]})!'
                
            if isinstance(batch, dict):
                image = batch["image"].clone()
                labels = {key: batch[key] for key in batch if key != "image"}
                sample = batch["sample"]
            else:
                image, labels, sample = batch
                image = image.clone()

            x_idx = list(range(0, configs['image_size'], configs['patch_size']))
            y_idx = list(range(0, configs['image_size'], configs['patch_size']))
            crop_indices = [x_idx, y_idx]

            for i, (x, y) in enumerate(product(*crop_indices)):
                if 1 in labels[:, x:x+configs['patch_size'], y:y+configs['patch_size']]:
                    pos_count += 1
                else:
                    neg_count += 1
                
                # Save tensors using io.BytesIO
                def save_tensor(tensor):
                    buffer = io.BytesIO()
                    torch.save(tensor, buffer)
                    return buffer.getvalue()
                
                obj = {
                    "__key__": f"sample{index:06d}_{x}_{y}",
                    "image.pth": save_tensor(image[:, x:x+configs['patch_size'], y:y+configs['patch_size']].clone()),
                    "sample.pth": save_tensor(sample),
                }
                if isinstance(labels, dict):
                    obj["labels.pth"] = save_tensor(labels[:, x:x+configs['patch_size'], y:y+configs['patch_size']].clone())
                else:
                    obj["labels.pth"] = save_tensor(labels[:, x:x+configs['patch_size'], y:y+configs['patch_size']].clone())
                
                sink.write(obj)
        print(f"Shard {i}: Positives: {pos_count}, Negatives: {neg_count}")

@ray.remote
def wds_write_ith_shard(configs, dataset, mode, i, n):
    ext = ".tar"
    
    # One writer for non-train modes
    if mode != 'train':
        shard_path = Path(os.path.join(configs["webdataset_path"], mode))
        shard_path.mkdir(parents=True, exist_ok=True)
        pattern = os.path.join(str(shard_path), f"sample-{mode}-{i}-%06d{ext}")
        sink = wds.ShardWriter(pattern, maxcount=configs["max_samples_per_shard"])
    else:
        # Two writers for train mode (positive and negative samples)
        shard_path_pos = Path(os.path.join(configs["webdataset_path"], 'train_pos'))
        shard_path_neg = Path(os.path.join(configs["webdataset_path"], 'train_neg'))
        shard_path_pos.mkdir(parents=True, exist_ok=True)
        shard_path_neg.mkdir(parents=True, exist_ok=True)
        pos_pattern = os.path.join(str(shard_path_pos), f"sample-train_pos-{i}-%06d{ext}")
        neg_pattern = os.path.join(str(shard_path_neg), f"sample-train_neg-{i}-%06d{ext}")
        pos_sink = wds.ShardWriter(pos_pattern, maxcount=configs["max_samples_per_shard"])
        neg_sink = wds.ShardWriter(neg_pattern, maxcount=configs["max_samples_per_shard"])

    pos_count, neg_count = 0, 0

    for index in tqdm.tqdm(range(i, len(dataset), n)):
        batch = dataset[index]

        if isinstance(batch, dict):
            image = batch["image"].clone()
            labels = {key: batch[key] for key in batch if key != "image"}
            sample = batch["sample"]
        else:
            image, labels, sample = batch
            image = image.clone()

        # Save tensors using io.BytesIO
        def save_tensor(tensor):
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            return buffer.getvalue()

        obj = {
            "__key__": f"sample{index:06d}",
            "image.pth": save_tensor(image),
            "sample.pth": save_tensor(sample),
        }
        if isinstance(labels, dict):
            obj["labels.pth"] = save_tensor(labels)
        else:
            obj["labels.pth"] = save_tensor(labels.clone())
    
        if 1 in sample['label']:
            print(f"Positive {mode} sample, label: {sample['label']}")
            pos_count += 1
            if mode == 'train':
                pos_sink.write(obj)
            else:
                sink.write(obj)
        else:
            neg_count += 1
            if mode == 'train':
                neg_sink.write(obj)
            else:
                sink.write(obj)

        print(f"Shard {i}: Positives: {pos_count}, Negatives: {neg_count}")

    # Close the shard writers
    if mode != 'train':
        sink.close()
    else:
        pos_sink.close()
        neg_sink.close()


def wds_write_parallel(configs, mode):
    ray.init()
    n = configs["webdataset_write_processes"]
    dataset = utils.get_dataset(configs, mode=mode, webdataset_write=True)

    print("=" * 40)
    print("Creating shards for dataset: ")
    print("Mode: ", mode, "Timeseries length:", configs['timeseries_length'], "Size: ", len(dataset))
    print("=" * 40)

    ray.get([wds_write_ith_shard.remote(configs, dataset, mode, i, n) for i in range(n)])
