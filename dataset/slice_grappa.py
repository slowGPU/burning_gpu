import torch
import lightning as L

from pathlib import Path
from collections import defaultdict
import h5py
import os
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List, Union, Final, Sequence

from .slice import SliceDataset, SliceDataType, SliceDataTransform
from .grappa import GrappaDataset, GrappaDataType

from common.utils import PUBLIC_ACCS

SliceGrappaDataType = Tuple[
    torch.Tensor,  # mask
    torch.Tensor,  # masked kspace
    torch.Tensor,  # grappa image
    Optional[torch.Tensor],  # target
    Optional[float],  # maximum
    str,  # filename
    int,  # slice number
]


def collate_with_none(batch: Sequence[SliceGrappaDataType]):
    batch_without_none = [
        (mask, kspace, grappa, fname, slice) for mask, kspace, grappa, _, _, fname, slice in batch
    ]

    collated_mask, collated_kspace, collated_grappa, collated_fname, collated_slice = (
        torch.utils.data._utils.collate.default_collate(batch_without_none)
    )

    return collated_mask, collated_kspace, collated_grappa, None, None, collated_fname, collated_slice


class SliceGrappaDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform,
        input_key: str,
        grappa_key: str,
        target_key: Optional[str] = None,
    ):
        self.ds_slice = SliceDataset(root, transform, input_key, target_key)
        self.ds_grappa = GrappaDataset(root, grappa_key, None)

        assert len(self.ds_slice) == len(
            self.ds_grappa
        ), "Length of Slice and Grappa datasets must be equal"

    def __len__(self) -> int:
        return len(self.ds_slice)

    def __getitem__(self, idx: int) -> SliceGrappaDataType:
        mask, kspace, target, maximum, fname, slice = self.ds_slice[idx]
        grappa, _ = self.ds_grappa[idx]

        return mask, kspace, grappa, target, maximum, fname, slice


class SliceGrappaDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 1,
        input_key: str = "kspace",
        grappa_key: str = "image_grappa",
        target_key: str = "image_label",
        max_key: str = "max",
        train_suffix: str = "train",
        val_suffix: str = "val",
        leaderboard_suffix: str = "leaderboard",
    ) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.input_key = input_key
        self.grappa_key = grappa_key
        self.target_key = target_key
        self.max_key = max_key

        root = Path(root)
        public_acc, private_acc = sorted(
            os.listdir(root / leaderboard_suffix),
            key=lambda x: (x not in PUBLIC_ACCS),
        )

        self.path_train = root / train_suffix
        self.path_val = root / val_suffix
        self.path_test = root / leaderboard_suffix / public_acc
        self.path_predict = root / leaderboard_suffix / private_acc

        self.dataset_train: Optional[Dataset] = None
        self.dataset_val: Optional[Dataset] = None
        self.dataset_test: Optional[Dataset] = None
        self.dataset_predict: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", "train") and self.dataset_train is None:
            self.dataset_train = SliceGrappaDataset(
                self.path_train,
                transform=SliceDataTransform(max_key=self.max_key),
                input_key=self.input_key,
                grappa_key=self.grappa_key,
                target_key=self.target_key,
            )

        if stage in ("fit", "validate") and self.dataset_val is None:
            self.dataset_val = SliceGrappaDataset(
                self.path_val,
                transform=SliceDataTransform(max_key=self.max_key),
                input_key=self.input_key,
                grappa_key=self.grappa_key,
                target_key=self.target_key,
            )

        if stage in ("test",) and self.dataset_test is None:
            self.dataset_test = SliceGrappaDataset(
                self.path_test,
                transform=SliceDataTransform(),
                input_key=self.input_key,
                grappa_key=self.grappa_key,
            )

        if stage in ("predict",) and self.dataset_predict is None:
            self.dataset_predict = SliceGrappaDataset(
                self.path_predict,
                transform=SliceDataTransform(),
                input_key=self.input_key,
                grappa_key=self.grappa_key,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_with_none,
            num_workers=8,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_with_none,
            num_workers=8,
        )
