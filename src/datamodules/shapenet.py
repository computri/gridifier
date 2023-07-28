# torch
import pytorch_lightning as pl
import torch
import torch_geometric.datasets as tg_datasets
from torch_geometric import loader as tg_loader
import torch_geometric.transforms as tg_transforms



class ShapeNetDataModule(pl.LightningDataModule):
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

        # Save parameters to self
        self.data_dir = data_dir + f"/shapenet"
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
        ]

        if num_samples > -1:
            transform.append(tg_transforms.FixedPoints(num=num_samples))

        if cfg.augment:
            self.train_transform = tg_transforms.Compose(
                transform
                + [
                   tg_transforms.RandomScale((0.8, 1.2)),
                   tg_transforms.RandomJitter(0.02)]
            )
            self.test_transform = tg_transforms.Compose(transform)
        else:
            self.train_transform = self.test_transform = tg_transforms.Compose(transform)

        self.pre_transform = None


    def prepare_data(self):
        # download
        tg_datasets.ShapeNet(root=self.data_dir, include_normals=self.use_normals, split="train")
        tg_datasets.ShapeNet(root=self.data_dir, include_normals=self.use_normals, split="val")
        tg_datasets.ShapeNet(root=self.data_dir, include_normals=self.use_normals, split="test")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            shapenet_train = tg_datasets.ShapeNet(
                root=self.data_dir,
                split="train",
                include_normals=self.use_normals,
                transform=self.train_transform,
            )
            shapenet_val = tg_datasets.ShapeNet(
                root=self.data_dir,
                split="val",
                include_normals=self.use_normals,
                transform=self.test_transform,
            )

            self.train_dataset = shapenet_train
            self.val_dataset = shapenet_val

        if stage == "test" or stage is None:
            self.test_dataset = tg_datasets.ShapeNet(
                root=self.data_dir,
                split="test",
                include_normals=self.use_normals,
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
                x = data.x.clone()
            else:
                x = torch.cat([x, data.x.clone()], dim=-1)
        data.x = x

        return data
