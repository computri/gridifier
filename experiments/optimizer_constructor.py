import ml_collections

# torch
import torch
from ml_collections import config_dict

# project
from src.nn import schedulers


def construct_optimizer(model, optim_cfg: config_dict.ConfigDict):
    """Constructs an optimizer for a given model."""

    # Unpack values from optim_cfg
    optimizer_type = optim_cfg.type
    lr = optim_cfg.lr
    weight_decay = optim_cfg.weight_decay

    parameters = model.parameters()

    # Construct optimizer
    if optimizer_type == "SGD":
        # Unpack values from optim_cfg.params
        momentum = optim_cfg.momentum
        nesterov = optim_cfg.nesterov
        optimizer = torch.optim.SGD(
            params=parameters,
            lr=float(lr),
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    else:
        optimizer_type = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_type(
            params=parameters,
            lr=float(lr),
            weight_decay=weight_decay,
        )
    return optimizer


def construct_scheduler(optimizer, scheduler_cfg: ml_collections.ConfigDict):
    """Creates a learning rate scheduler for a given model."""

    # Unpack values from cfg.train.scheduler_params
    scheduler_type = scheduler_cfg.type
    factor = scheduler_cfg.factor
    decay_steps = scheduler_cfg.decay_steps
    patience = scheduler_cfg.patience
    mode = scheduler_cfg.mode

    # Get iterations for warmup
    warmup_epochs = scheduler_cfg.warmup_epochs
    warmup_iterations = scheduler_cfg.warmup_epochs * scheduler_cfg.iters_per_train_epoch

    # Get total iterations (used for CosineScheduler)
    total_iterations = scheduler_cfg.total_train_iters
    iters_per_train_epoch = scheduler_cfg.iters_per_train_epoch

    # Create warm_up scheduler
    if warmup_epochs != -1:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iterations
        )
    else:
        warmup_scheduler = None

    # Check consistency
    if scheduler_type != "cosine" and factor == -1:
        raise ValueError(f"The factor cannot be {factor} for scheduler {scheduler_type}")

    # Create scheduler
    if scheduler_type == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=iters_per_train_epoch * decay_steps,
            gamma=factor,
            last_epoch=-warmup_iterations,
        )  # user to sync with warmup
    elif scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=True
        )
    elif scheduler_type == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=factor, last_epoch=-warmup_iterations
        )  # user to sync with warmup
    elif scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=total_iterations - warmup_iterations,
            last_epoch=-warmup_iterations,
        )
    else:
        lr_scheduler = None
        print(f"WARNING! No scheduler will be used. cfg.train.scheduler = {scheduler_type}")

    # Concatenate schedulers if required
    if warmup_scheduler is not None:
        # If both schedulers are defined, concatenate them
        if lr_scheduler is not None:
            lr_scheduler = schedulers.ChainedScheduler(
                [
                    warmup_scheduler,
                    lr_scheduler,
                ]
            )
        # Otherwise, return only the warmup scheduler
        else:
            lr_scheduler = lr_scheduler
    return lr_scheduler
