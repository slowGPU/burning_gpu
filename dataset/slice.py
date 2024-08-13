import os
from pathlib import Path
from typing import Final, List, Optional, Sequence, Tuple, Union

import h5py
import lightning as L
import numpy as np
import torch
import torch.utils
from torch.utils.data import DataLoader, Dataset
from common.utils import PUBLIC_ACCS


SliceDataType = Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[float],
    str,
    int,
]


def collate_with_none(batch: Sequence[SliceDataType]):
    batch_without_none = [
        (mask, kspace, fname, slice) for mask, kspace, _, _, fname, slice in batch
    ]

    collated_mask, collated_kspace, collated_fname, collated_slice = (
        torch.utils.data._utils.collate.default_collate(batch_without_none)
    )

    return collated_mask, collated_kspace, None, None, collated_fname, collated_slice


class SliceDataTransform:
    def __init__(self, max_key: Optional[str] = None):
        self.max_key = max_key

    def __call__(self, mask, input, target, attrs, fname, slice) -> SliceDataType:
        if self.max_key is not None:
            target = torch.from_numpy(target)
            maximum = attrs[self.max_key]
        else:
            target = None
            maximum = None

        kspace = torch.from_numpy(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(
            mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)
        ).byte()
        return mask, kspace, target, maximum, fname, slice


class SliceDataset(Dataset):
    def __init__(
        self, root: Path, transform, input_key: str, target_key: Optional[str] = None
    ):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.image_examples = []
        self.kspace_examples = []

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]

        if self.target_key is not None:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i: int) -> SliceDataType:
        kspace_fname, dataslice = self.kspace_examples[i]
        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])

        if self.target_key is not None:
            image_fname, _ = self.image_examples[i]
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
        else:
            target = None
            attrs = None

        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


class SliceDataModule(L.LightningDataModule):

    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 1,
        input_key: str = "kspace",
        target_key: str = "image_label",
        max_key: str = "max",
        train_suffix: str = "train",
        val_suffix: str = "val",
        leaderboard_suffix: str = "leaderboard",
    ):
        super().__init__()

        self.batch_size = batch_size

        self.input_key = input_key
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
            self.dataset_train = SliceDataset(
                self.path_train,
                transform=SliceDataTransform(max_key=self.max_key),
                input_key=self.input_key,
                target_key=self.target_key,
            )

        if stage in ("fit", "validate") and self.dataset_val is None:
            self.dataset_val = SliceDataset(
                self.path_val,
                transform=SliceDataTransform(max_key=self.max_key),
                input_key=self.input_key,
                target_key=self.target_key,
            )

        if stage in ("test",) and self.dataset_test is None:
            self.dataset_test = SliceDataset(
                self.path_test,
                transform=SliceDataTransform(),
                input_key=self.input_key,
            )

        if stage in ("predict",) and self.dataset_predict is None:
            self.dataset_predict = SliceDataset(
                self.path_predict,
                transform=SliceDataTransform(),
                input_key=self.input_key,
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
    
# class SliceGrappaDataset(SliceDataset):
    