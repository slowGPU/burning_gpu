import torch
import numpy as np

from dataset import SliceDataModule
from common.utils import save_reconstructions
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from models.freezed_varnet_nafnet import FreezedVarNetNAFNet as Model

MODEL_PATH = "~/result/best_model.pt"
SAVE_DIR = "reconstructions"

model = torch.load(MODEL_PATH)

dm = SliceDataModule(root="/home/Data")
dm.setup("test")
dm.setup("predict")


work = {
    "public": dm.test_dataloader(),
    "private": dm.predict_dataloader()
}

model.eval()
with torch.no_grad():
    for phase, dataloader in work.items():
        reconstructions = defaultdict(dict)
        print(f"Reconstructing {phase} leaderboard...")
        for mask, kspace, target, maximum, fnames, slices in tqdm(dataloader):
            output = model(kspace.cuda(non_blocking=True), mask.cuda(non_blocking=True))
            output = model.image_space_crop(output)
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][slices[i]] = output[i].cpu().numpy()

        for fname in reconstructions:
            reconstructions[fname] = np.stack(
                [reconstructions[fname][slice] for slice in sorted(reconstructions[fname])]
            )
        print(f"Saving reconstructions of {phase} leaderboard...")
        save_reconstructions(reconstructions, Path(f"{SAVE_DIR}/{phase}"))