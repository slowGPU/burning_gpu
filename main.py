import mlflow
import torch
from lightning.pytorch.cli import LightningCLI

from dataset import SliceDataModule
from models import VarNet, VarNetFreezedSens, VarNetHybridFreeze


class VarNetOL(VarNet):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetFreezedSensOL(VarNetFreezedSens):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetHybridFreezeOL(VarNetHybridFreeze):
    def configure_optimizers(self, lr1: float = 1e-3, lr2: float = 1e-3):
        sme_opt = torch.optim.Adam(self.sme_parameters(), lr=lr1)
        recon_opt = torch.optim.Adam(self.recon_parameters(), lr=lr2)

        return sme_opt, recon_opt


mlflow.autolog()
cli = LightningCLI(datamodule_class=SliceDataModule)
