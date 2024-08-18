import torch

from .varnet import VarNet
from .varnet_freezed_sens import VarNetFreezedSens
from .varnet_hybrid_freeze import VarNetHybridFreeze
from .varnet_rough_sens import VarNetRoughSens
from .varnet_rough_sens_faster import VarNetRoughSensFaster
from .varnet_logistic_sens import VarNetLogisticSens
from .varnet_logistic_sens_residual import VarNetLogisticSensResidual
from .varnet_logistic_unet_sens import VarNetLogisticUnetSens
from .varnet_logistic_unet_sens_fix import VarNetLogisticUnetSensFix
from .varnet_freezed_sens_nafnet import VarNetFreezedSensNAFNet
from .freezed_varnet_nafnet import FreezedVarNetNAFNet


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


class VarNetRoughSensOL(VarNetRoughSens):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetRoughSensFasterOL(VarNetRoughSensFaster):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetLogisticSensOL(VarNetLogisticSens):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetLogisticSensResidualOL(VarNetLogisticSensResidual):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetLogisticUnetSensOL(VarNetLogisticUnetSens):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetLogisticUnetSensFixOL(VarNetLogisticUnetSensFix):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class VarNetFreezedSensNAFNetOL(VarNetFreezedSensNAFNet):
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class FreezedVarNetNAFNetOL(FreezedVarNetNAFNet):
    def configure_optimizers(self, lr: float = 1e-3):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=[0.9, 0.9], weight_decay=0
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200000, eta_min=1e-7
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
