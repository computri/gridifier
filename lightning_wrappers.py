import glob

import pytorch_lightning as pl

# torch
import torch
import torchmetrics
import wandb
from ml_collections import config_dict

# project
import experiments
import src
import src.nn as src_nn

class LightningWrapperBase(pl.LightningModule):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
    ):
        super().__init__()
        # Define network
        self.network = network
        # Save optimizer & scheduler parameters
        self.optim_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler
        # Placeholders for logging of best train & validation values
        self.num_params = -1
        # Explicitly define whether we are in distributed mode.
        self.distributed = cfg.train.distributed and cfg.train.avail_gpus != 1

        self.time_inference = cfg.track_inference_time
        if self.time_inference:
            self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def forward(self, x, time_inference=False):
        if time_inference:
            self.start_timer()

        out = self.network(x)

        if time_inference:
            self.end_timer()

        return out

    def start_timer(self):
        self.starter.record()

    def end_timer(self):
        self.ender.record()
        torch.cuda.synchronize()
        val_time = self.starter.elapsed_time(self.ender)
        self.log(
            "val/memory_allocated",
            torch.cuda.memory_allocated() /1024/1024/1024
        )
        self.log(
            "val/inference_time",
            val_time
        )

    def configure_optimizers(self):
        # Construct optimizer & scheduler
        optimizer = experiments.construct_optimizer(model=self, optim_cfg=self.optim_cfg)
        scheduler = experiments.construct_scheduler(
            optimizer=optimizer, scheduler_cfg=self.scheduler_cfg
        )
        # Construct output dictionary
        output_dict = {"optimizer": optimizer}
        if scheduler is not None:
            output_dict["lr_scheduler"] = {}
            output_dict["lr_scheduler"]["scheduler"] = scheduler
            output_dict["lr_scheduler"]["interval"] = "step"

            # If we use a ReduceLROnPlateu scheduler, we must monitor val/acc
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if self.scheduler_cfg.mode == "min":
                    output_dict["lr_scheduler"]["monitor"] = "val/loss"
                else:
                    output_dict["lr_scheduler"]["monitor"] = "val/acc"
                output_dict["lr_scheduler"]["reduce_on_plateau"] = True
                output_dict["lr_scheduler"]["interval"] = "epoch"
            # TODO(dwromero): ReduceLROnPlateau with warmup
            if isinstance(scheduler, src_nn.schedulers.ChainedScheduler) and isinstance(
                scheduler._schedulers[-1], torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                raise NotImplementedError("cannot use ReduceLROnPlateau with warmup")
        return output_dict

    def on_train_start(self):
        if self.global_rank == 0:
            # Calculate and log the size of the model
            if self.num_params == -1:
                with torch.no_grad():
                    # Log parameters
                    num_params = src.utils.num_params(self.network)
                    self.logger.experiment.summary["num_params"] = num_params
                    self.num_params = num_params
                    # Log source code files
                    code = wandb.Artifact(
                        f"source-code-{self.logger.experiment.name}", type="code"
                    )
                    # Get paths of all source code files
                    paths = glob.glob("**/*.py", recursive=True)
                    paths += glob.glob("**/*.yaml", recursive=True)
                    # Filter paths
                    paths = list(filter(lambda x: "wandb" not in x, paths))
                    # Get all source files
                    for path in paths:
                        code.add_file(path, name=path)
                    # Use the artifact
                    if not self.logger.experiment.offline:
                        wandb.run.use_artifact(code)


class ClassificationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
        **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)
        # Metric computers
        if cfg.net.out_channels == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_acc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.val_acc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.val_f1 = torchmetrics.F1Score(task=task, num_classes=cfg.net.out_channels, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.test_f1 = torchmetrics.F1Score(task=task, num_classes=cfg.net.out_channels, top_k=1)
        self.loss_metric = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

        # Caches for step responses
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.batch_size = cfg.train.batch_size

        # Placeholders for logging of best train & validation values
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0

        # Compute predictions
        self.get_predictions = lambda logits: torch.argmax(logits, 1)

    def _step(self, batch, accuracy_calculator, time_inference=False, f1_calculator=None):
        x = batch

        logits = self(x, time_inference)

        # Predictions
        predictions = self.get_predictions(logits)
        # Calculate accuracy and loss
        labels = batch.y

        accuracy_calculator(predictions, labels)

        if f1_calculator is not None:
            f1_calculator(predictions, labels)

        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.train_acc, time_inference=False)
        # Log and return loss (Required in training step)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "train/acc",
            self.train_acc,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.training_step_outputs.append(logits.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, self.val_acc, time_inference=self.time_inference, f1_calculator=self.val_f1)
        # Log and return loss (Required in training step)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "val/acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "val/F1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size
        )
        self.validation_step_outputs.append(logits)
        return logits  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(batch, self.test_acc, time_inference=False, f1_calculator=self.test_f1)
        # Log and return loss (Required in training step)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )
        self.log(
            "test/acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size,
        )

        self.log(
            "test/F1",
            self.test_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=self.batch_size
        )

    def on_train_epoch_end(self):
        flattened_logits = torch.cat(self.training_step_outputs)
        flattened_logits = torch.flatten(flattened_logits)
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )
        self.training_step_outputs.clear()
        # Log best accuracy
        train_acc = self.trainer.callback_metrics["train/acc_epoch"]
        if train_acc > self.best_train_acc:
            self.best_train_acc = train_acc.item()
            self.logger.experiment.log(
                {
                    "train/best_acc": self.best_train_acc,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Gather logits from validation set and construct a histogram of them.
        flattened_logits = torch.flatten(torch.cat(self.validation_step_outputs))

        self.logger.experiment.log(
            {
                "val/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "val/logit_max_abs_value": flattened_logits.abs().max().item(),
                "global_step": self.global_step,
            }
        )
        self.validation_step_outputs.clear()
        # Log best accuracy
        val_acc = self.trainer.callback_metrics["val/acc"]
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc.item()
            self.logger.experiment.log(
                {
                    "val/best_acc": self.best_val_acc,
                    "global_step": self.global_step,
                }
            )


class SegmentationWrapper(LightningWrapperBase):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: config_dict.ConfigDict,
        **kwargs,
    ):
        super().__init__(network=network, cfg=cfg)
        # Metric computers
        if cfg.net.out_channels == 2:
            task = "binary"
        else:
            task = "multiclass"

        self.train_mAcc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.val_mAcc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)
        self.test_mAcc = torchmetrics.Accuracy(task=task, num_classes=cfg.net.out_channels)

        self.time_inference = cfg.track_inference_time

        self.num_classes = cfg.net.out_channels

        self.loss_metric = torch.nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)

        # Caches for step responses
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.batch_size = cfg.train.batch_size

        self.best_train_mAcc = 0.0
        self.best_val_mAcc = 0.0
        self.best_val_ins_miou = 0.0
        self.best_val_cat_miou = 0.0

        assert cfg.dataset.name == "ShapeNet", "currently hardcoded for shapenet"

        self.global_train_iou = None
        self.global_val_iou = None
        self.category_to_seg_classes = {
            "Earphone": [16, 17, 18],
            "Motorbike": [30, 31, 32, 33, 34, 35],
            "Rocket": [41, 42, 43],
            "Car": [8, 9, 10, 11],
            "Laptop": [28, 29],
            "Cap": [6, 7],
            "Skateboard": [44, 45, 46],
            "Mug": [36, 37],
            "Guitar": [19, 20, 21],
            "Bag": [4, 5],
            "Lamp": [24, 25, 26, 27],
            "Table": [47, 48, 49],
            "Airplane": [0, 1, 2, 3],
            "Pistol": [38, 39, 40],
            "Chair": [12, 13, 14, 15],
            "Knife": [22, 23],
        }
        # inverse mapping
        self.seg_class_to_category = {}
        for cat in self.category_to_seg_classes.keys():
            for cls in self.category_to_seg_classes[cat]:
                self.seg_class_to_category[cls] = cat

        # Compute predictions
        self.get_predictions = lambda logits: torch.argmax(logits, 1)

    def _step(self, batch, time_inference=False, acc_calculator=None):
        x = batch

        logits = self(x, time_inference)

        # Predictions
        predictions = self.get_predictions(logits)

        # Calculate accuracy and loss

        labels = batch.y
        acc_calculator(predictions, labels)

        loss = self.loss_metric(logits, labels)
        # Return predictions and loss
        return predictions, logits, loss

    def training_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss= self._step(batch, time_inference=False, acc_calculator=self.train_mAcc)

        # Log and return loss (Required in training step)
        self.log("train/loss",
                 loss.item(),
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )

        self.log("train/mAcc",
                 self.train_mAcc,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )
        self.training_step_outputs.append(logits.detach().cpu())
        return loss

    def validation_step(self, batch, batch_idx):
        # Perform step
        predictions, logits, loss = self._step(batch, time_inference=self.time_inference, acc_calculator=self.val_mAcc)

        ious = self.compute_shape_ious(logits.reshape(self.batch_size, -1, self.num_classes), batch.y.reshape(self.batch_size, -1))
        self.validation_step_outputs.append(ious)

        # Log and return loss (Required in training step)
        self.log("val/loss",
                 loss.item(),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )


        self.log("val/mAcc",
                 self.val_mAcc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size
                )

        return ious  # used to log histograms in validation_epoch_step

    def test_step(self, batch, batch_idx):
        # Perform step
        predictions, _, loss = self._step(batch, time_inference=False, acc_calculator=self.test_mAcc)
        # Log and return loss (Required in training step)
        self.log("test/loss",
                 loss.item(),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size,
                )

        self.log("test/mAcc",
                 self.test_mAcc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=self.distributed,
                 batch_size=self.batch_size
                )


    def on_train_epoch_end(self):
        flattened_logits = torch.cat(self.training_step_outputs)
        flattened_logits = torch.flatten(flattened_logits)
        self.logger.experiment.log(
            {
                "train/logits": wandb.Histogram(flattened_logits.to("cpu")),
                "global_step": self.global_step,
            }
        )
        self.training_step_outputs.clear()

        train_mAcc = self.trainer.callback_metrics["train/mAcc_epoch"]
        if train_mAcc > self.best_train_mAcc:
            self.best_train_mAcc = train_mAcc.item()
            self.logger.experiment.log(
                {
                    "train/best_mAcc": self.best_train_mAcc,
                    "global_step": self.global_step,
                }
            )

    def on_validation_epoch_end(self):
        # Gather logits from validation set and construct a histogram of them.

        all_shape_mious, cat_mious = self.get_mious(self.validation_step_outputs)

        self.log("val/ins_miou", all_shape_mious.mean())
        self.log("val/cat_miou", torch.stack(list(cat_mious.values())).mean())
        for cat in sorted(cat_mious.keys()):
            self.log(f"val/cat_miou_{cat}", cat_mious[cat])

        self.validation_step_outputs.clear()

        val_mAcc = self.trainer.callback_metrics["val/mAcc"]
        if val_mAcc > self.best_val_mAcc:
            self.best_val_mAcc = val_mAcc.item()
            self.logger.experiment.log(
                {
                    "val/best_mAcc": self.best_val_mAcc,
                    "global_step": self.global_step,
                }
            )

        val_ins_miou = self.trainer.callback_metrics["val/ins_miou"]
        if val_ins_miou > self.best_val_ins_miou:
            self.best_val_ins_miou = val_ins_miou.item()
            self.logger.experiment.log(
                {
                    "val/best_ins_miou": self.best_val_ins_miou,
                    "global_step": self.global_step,
                }
            )

        val_cat_miou = self.trainer.callback_metrics["val/cat_miou"]
        if val_cat_miou > self.best_val_cat_miou:
            self.best_val_cat_miou = val_cat_miou.item()
            self.logger.experiment.log(
                {
                    "val/best_cat_miou": self.best_val_cat_miou,
                    "global_step": self.global_step,
                }
            )


    # Metric calculation taken and adapted from https://github.com/kabouzeid/point2vec
    def compute_shape_ious(self, log_probabilities, seg_labels):
        # log_probablities: (B, N, 50) \in -inf..<0
        # seg_labels:       (B, N) \in 0..<50
        # returns           { cat: (S, P) }


        shape_ious = {
            cat: [] for cat in self.category_to_seg_classes.keys()
        }

        for i in range(log_probabilities.shape[0]):
            cat = self.seg_class_to_category[seg_labels[i, 0].item()]
            seg_classes = self.category_to_seg_classes[cat]
            seg_preds = (
                torch.argmax(
                    log_probabilities[i, :, self.category_to_seg_classes[cat]], dim=1
                )
                + seg_classes[0]
            )  # (N,)

            seg_class_iou = torch.empty(len(seg_classes))
            for c in seg_classes:
                if ((seg_labels[i] == c).sum() == 0) and (
                    (seg_preds == c).sum() == 0
                ):  # part is not present, no prediction as well
                    seg_class_iou[c - seg_classes[0]] = 1.0
                else:
                    intersection = ((seg_labels[i] == c) & (seg_preds == c)).sum()
                    union = ((seg_labels[i] == c) | (seg_preds == c)).sum()
                    seg_class_iou[c - seg_classes[0]] = intersection / union
            shape_ious[cat].append(seg_class_iou.mean())

        return shape_ious

    def get_mious(self, outputs):
        shape_mious = {
            cat: [] for cat in self.category_to_seg_classes.keys()
        }
        for d in outputs:
            for k, v in d.items():
                shape_mious[k] = shape_mious[k] + v

        all_shape_mious = torch.stack(
            [miou for mious in shape_mious.values() for miou in mious]
        )
        cat_mious = {
            k: torch.stack(v).mean() for k, v in shape_mious.items() if len(v) > 0
        }
        return all_shape_mious, cat_mious

