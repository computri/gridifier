import os

import pytorch_lightning as pl

# torch
import torch
from ml_collections import config_dict

# project
from src import datamodules


def construct_datamodule(cfg: config_dict.ConfigDict) -> pl.LightningDataModule:
    # Define num_workers
    if cfg.num_workers == -1:
        cfg.num_workers = int(os.cpu_count() / torch.cuda.device_count())

    # Define pin_memory
    if cfg.device == "cuda" and torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False

    # Gather module from datamodules & create instance
    dataset_name = f"{cfg.dataset.name}DataModule"
    dataset = getattr(datamodules, dataset_name)
    datamodule = dataset(
        cfg=cfg.dataset,
        batch_size=cfg.train.batch_size // cfg.train.accumulate_grad_steps,
        test_batch_size=cfg.test.batch_size_multiplier * cfg.train.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        **cfg.dataset.params,
    )

    # All datamodules must define these variables for model creation
    assert hasattr(datamodule, "input_channels")
    assert hasattr(datamodule, "output_channels")
    assert hasattr(datamodule, "data_dim")
    return datamodule