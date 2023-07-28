# torch
import pytorch_lightning as pl
import torch
import torch_geometric.datasets as tg_datasets
from torch.utils.data import DataLoader, random_split
from torch_geometric import loader as tg_loader

import torch_geometric.transforms as tg_transforms

from torchvision import datasets, transforms


class ModelNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        **kwargs,
    ):
        super().__init__()

        num_classes = cfg.params.num_classes
        use_positions = cfg.params.use_positions
        use_normals = cfg.params.use_normals
        data_dir = cfg.data_dir
        num_samples = cfg.params.num_samples
        val_split_as_test_split = cfg.params.val_split_as_test_split

        if num_classes not in [10, 40]:
            raise ValueError(f"Number of classes {num_classes} not valid. num_classes = [10, 40]")

        # Save parameters to self
        self.data_dir = data_dir + f"/modelnet{num_classes}"
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.use_positions = use_positions
        self.use_normals = use_normals
        self.val_split_as_test_split = val_split_as_test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_dim = 3
        self.input_channels = self.data_dim * (int(self.use_positions) + int(self.use_normals))
        self.output_channels = self.num_classes

        # Create transforms
        transform = [
            tg_transforms.NormalizeScale(),
            tg_transforms.Center(),
            tg_transforms.SamplePoints(
                num=num_samples, remove_faces=True, include_normals=use_normals
            ),
        ]

        if cfg.augment:
            self.train_transform = tg_transforms.Compose(
                transform
                + [tg_transforms.RandomRotate(180, axis=-1), 
                   tg_transforms.RandomScale((0.8, 1.2)),
                   tg_transforms.RandomJitter(0.02)]
            )
            self.test_transform = tg_transforms.Compose(transform)
        else:
            self.train_transform = self.test_transform = tg_transforms.Compose(transform)

        self.pre_transform = None


    def prepare_data(self):
        # download
        tg_datasets.ModelNet(
            root=self.data_dir,
            name=str(self.num_classes),
            pre_transform=self.pre_transform,
            train=True,
        )
        tg_datasets.ModelNet(
            root=self.data_dir,
            name=str(self.num_classes),
            pre_transform=self.pre_transform,
            train=False,
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            modelnet_train = tg_datasets.ModelNet(
                root=self.data_dir,
                name=str(self.num_classes),
                train=True,
                transform=self.train_transform,
            )
            modelnet_test = tg_datasets.ModelNet(
                root=self.data_dir,
                name=str(self.num_classes),
                train=False,
                transform=self.test_transform,
            )

            if self.val_split_as_test_split:
                self.train_dataset = modelnet_train
                self.val_dataset = modelnet_test
            else:
                if self.num_classes == 40:
                    splits = [8858, 985]
                else:
                    splits = [3591, 400]
                self.train_dataset, self.val_dataset = random_split(
                    modelnet_train,
                    splits,
                    generator=torch.Generator().manual_seed(getattr(self, "seed", 42)),
                )

        if stage == "test" or stage is None:
            self.test_dataset = tg_datasets.ModelNet(
                root=self.data_dir,
                name=str(self.num_classes),
                train=False,
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return tg_loader.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return tg_loader.DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return tg_loader.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def on_before_batch_transfer(self, data, dataloader_idx):
        if self.use_positions:
            x = data.pos.clone()
        else:
            x = None
        if self.use_normals:
            if x is None:
                x = data.normal.clone()
            else:
                x = torch.cat([x, data.normal], dim=-1)
        data.x = x
        return data




