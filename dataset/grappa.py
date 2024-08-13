import torch
import lightning as L

from pathlib import Path
from collections import defaultdict
import h5py
import os
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, List, Union, Final

from common.utils import PUBLIC_ACCS

GrappaDataType = Tuple[
    torch.Tensor,  # Grappa Image
    Optional[torch.Tensor],  # Ground Truth Image (target)
]


class GrappaDataset(Dataset):
    def __init__(
        self, root: Path, input_key: str, target_key: Optional[str] = None
    ) -> None:
        self.input_key = input_key
        self.target_key = target_key
        self.image_examples = []
        self.grappa_examples = []

        grappa_files = list(Path(root / "image").iterdir())
        for fname in sorted(grappa_files):
            num_slices = self._get_metadata(fname)

            for slice_ind in range(num_slices):
                self.grappa_examples.append((fname, slice_ind))
            
        if target_key is not None:
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


    def __len__(self) -> int:
        return len(self.grappa_examples)

    def __getitem__(self, idx: int) -> GrappaDataType:
        grappa_fname, dataslice = self.grappa_examples[idx]
        with h5py.File(grappa_fname, "r") as hf:
            grappa_image = torch.tensor(hf[self.input_key][dataslice])

        if self.target_key is not None:
            image_fname, _ = self.image_examples[idx]
            with h5py.File(image_fname, "r") as hf:
                target = torch.tensor(hf[self.target_key][dataslice])
        else:
            target = None
        
        return grappa_image, target


class GrappaDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int = 1,
        input_key: str = "image_grappa",
        target_key: str = "image_label",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.input_key = input_key
        self.target_key = target_key

        public_acc, private_acc = sorted(
            os.listdir(root / "leaderboard"),
            key=lambda x: (x not in PUBLIC_ACCS),
        )

        self.path_train = self.root / "train"
        self.path_val = self.root / "val"
        self.path_test = self.root / "leaderboard" / public_acc
        self.path_predict = self.root / "leaderboard" / private_acc

        self.data_train: Optional[GrappaDataset] = None
        self.data_val: Optional[GrappaDataset] = None
        self.data_test: Optional[GrappaDataset] = None
        self.data_predict: Optional[GrappaDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "train") and self.dataset_train is None:
            self.dataset_train = GrappaDataset(
                self.path_train, self.input_key, self.target_key
            )

        if stage in ("fit", "val") and self.dataset_val is None:
            self.dataset_val = GrappaDataset(
                self.path_val, self.input_key, self.target_key
            )

        if stage in ("test",) and self.dataset_test is None:
            self.dataset_test = GrappaDataset(self.path_test, self.input_key)

        if stage in ("predict",) and self.dataset_predict is None:
            self.dataset_predict = GrappaDataset(self.path_predict, self.input_key)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
        )
