import ml_collections
import pytorch_lightning as pl

# torch
import torch
import wandb
from absl import app
from ml_collections import config_flags
from pytorch_lightning import loggers as pl_loggers

import cfg.utils as cfg_utils

import experiments

import warnings
warnings.filterwarnings("ignore")

_CONFIG_FILE = config_flags.DEFINE_config_file("cfg", default="cfg/default_config.py")


def main(_):
    cfg = _CONFIG_FILE.value

    cfg = cfg.unlock()  # Allow to append values to the config dict.

    # Before start training. Verify arguments in the cfg.
    verify_config(cfg)

    # Use Tensor cores properly
    torch.set_float32_matmul_precision("high")

    # Set seed
    # IMPORTANT! This does not make training entirely deterministic.
    # For that Trainer(deterministic=True) is also required!
    pl.seed_everything(cfg.seed, workers=True)

    # Check number of available gpus
    cfg.train.avail_gpus = torch.cuda.device_count()

    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Construct data_module
    datamodule = experiments.construct_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    # Append no of iteration to the cfg file for the definition of the schedulers
    distrib_batch_size = cfg.train.batch_size
    if cfg.train.distributed:
        distrib_batch_size *= cfg.train.avail_gpus
    cfg.scheduler.iters_per_train_epoch = len(datamodule.train_dataset) // distrib_batch_size
    cfg.scheduler.total_train_iters = cfg.scheduler.iters_per_train_epoch * cfg.train.epochs

    # Construct model
    model = experiments.construct_model(cfg, datamodule)

    print(model.network)
    # model = torch.compile(model, mode="reduce-overhead")  TODO(dwromero): Not working with torchmetrics

    # Initialize wandb logger
    if cfg.debug:
        log_model = False
        offline = True
    else:
        log_model = "all"
        offline = False
    wandb_logger = pl_loggers.WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=cfg_utils.flatten_configdict(cfg, parent_key="cfg"),
        log_model=log_model,  # used to save models to wandb during training
        offline=offline,
        save_code=True,
    )

    # Recreate & log the execution command used this run.
    if isinstance(wandb_logger.experiment.settings, wandb.Settings):
        args = wandb_logger.experiment.settings._args
        command = " ".join(args)
        wandb_logger.experiment.config.update({"command": command})

    # Print the cfg files prior to training
    print(f"Input arguments \n {cfg}")

    # Create trainer
    trainer, checkpoint_callback = experiments.construct_trainer(cfg, wandb_logger)
    # Load checkpoint
    if cfg.pretrained.load:
        # Construct artifact path.
        checkpoint_path = f"artifacts/{cfg.pretrained.filename}"
        # Load model from artifact
        print(
            f'IGNORE this validation run. Required due to problem with Lightning model loading \n {"#" * 200}'
        )
        trainer.validate(model, datamodule)
        print("#" * 200)
        checkpoint_path += "/model.ckpt"
        model = model.__class__.load_from_checkpoint(
            checkpoint_path, network=model.network, cfg=cfg
        )

    # GPU-WARM-UP
    if cfg.track_inference_time:
        dummy_input = next(iter(datamodule.train_dataloader())).to(cfg.device)

        dummy_input = datamodule.on_before_batch_transfer(dummy_input, 0)
        model.network.to(cfg.device)

        for it in range(10):
            _ = model.network(dummy_input)

    # Test before training
    if cfg.test.before_train:
        trainer.validate(model, datamodule)
        trainer.test(model, datamodule)

    # Train
    if cfg.train.do:
        if cfg.pretrained.load:
            # From preloaded point
            trainer.fit(model, datamodule, ckpt_path=checkpoint_path)
        else:
            # From scratch
            trainer.fit(model, datamodule)
        # Load state dict from best performing model
        model.load_state_dict(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
        )

    # Validate and test before finishing
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)


def verify_config(cfg: ml_collections.ConfigDict):
    pass


if __name__ == "__main__":
    app.run(main)
