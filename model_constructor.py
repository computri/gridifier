import pytorch_lightning as pl
from ml_collections import config_dict

# torch
import torch

# project
import src
from experiments import lightning_wrappers


def construct_model(
        cfg: config_dict.ConfigDict, datamodule: pl.LightningDataModule
) -> pl.LightningModule:
    # Get parameters of model from task type
    in_channels = datamodule.input_channels
    out_channels = datamodule.output_channels

    # Get type of model from task type
    if cfg.net.type == "PointCloudResNet":
        if cfg.task == "classification":
            net_type = "PointCloudResNet"
        elif cfg.task == "segmentation":
            net_type = "PointCloudSegmentationResNet"
    elif cfg.net.type == "PointNet++":
        if cfg.task == "classification":
            net_type = "PointNet2"
        else:
            net_type = "PointNet2Segmentation"

    else:
        assert False, f"net_type '{cfg.net.type}' not recognized."

    # Overwrite data_dim in cfg.net
    cfg.net.in_channels = in_channels
    cfg.net.out_channels = out_channels

    # Print automatically derived model parameters.
    print(
        f"Automatic Parameters:\n dataset = {cfg.dataset.name}, "
        f" in_channels = {in_channels},"
        f" out_channels = {out_channels}."
    )

    # Create and return model
    net_type = getattr(src.models, net_type)
    network = net_type(in_channels=in_channels,
                       out_channels=out_channels,
                       cfg=cfg)

    if cfg.task == "classification":
        # Wrap the network in a LightningModule.
        model = lightning_wrappers.ClassificationWrapper(network=network, cfg=cfg)
    elif cfg.task == "segmentation":
        model = lightning_wrappers.SegmentationWrapper(network=network, cfg=cfg)
    else:
        assert False, f"task '{cfg.task}' not recognized."

    return model
