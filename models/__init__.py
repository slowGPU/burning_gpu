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
from .varnet_logistic_mamba import VarNetLogisticMamba
from .varnet_logistic_bound import VarNetLogisticBound
from .varnet_logistic_bound_full import VarNetLogisticBoundFull


class LitAdam:
    def configure_optimizers(self, lr: float = 1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)


class LitAdamW:
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


class VarNetOL(LitAdam, VarNet):
    pass


class VarNetFreezedSensOL(LitAdam, VarNetFreezedSens):
    pass


class VarNetHybridFreezeOL(LitAdam, VarNetHybridFreeze):
    pass


class VarNetRoughSensOL(LitAdam, VarNetRoughSens):
    pass


class VarNetRoughSensFasterOL(LitAdam, VarNetRoughSensFaster):
    pass


class VarNetLogisticSensOL(LitAdam, VarNetLogisticSens):
    pass


class VarNetLogisticSensResidualOL(LitAdam, VarNetLogisticSensResidual):
    pass


class VarNetLogisticUnetSensOL(LitAdam, VarNetLogisticUnetSens):
    pass


class VarNetLogisticUnetSensFixOL(LitAdam, VarNetLogisticUnetSensFix):
    pass


class VarNetFreezedSensNAFNetOL(LitAdam, VarNetFreezedSensNAFNet):
    pass


class FreezedVarNetNAFNetOL(LitAdamW, FreezedVarNetNAFNet):
    pass


class VarNetLogisticMambaOL(LitAdamW, VarNetLogisticMamba):
    pass


class VarNetLogisticBoundOL(LitAdamW, VarNetLogisticBound):
    pass


class VarNetLogisticBoundFullOL(LitAdamW, VarNetLogisticBoundFull):
    pass
