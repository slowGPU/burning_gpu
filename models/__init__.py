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

# from .varnet_logistic_mamba import VarNetLogisticMamba
from .varnet_logistic_bound import VarNetLogisticBound
from .varnet_logistic_bound_full import VarNetLogisticBoundFull
from .varnet_freezed_sens_nafnet_l1 import VarNetFreezedSensNAFNetL1
from .varnet_logistic_bound_l1 import VarNetLogisticBoundL1
from .freezed_varnet_nbnet import FreezedVarNetNBNet
from .freezed_varnet_restormer import FreezedVarNetRestormer
from .varnet_bound_moe_l1 import VarNetBoundMOEL1
from .varnet_full import VarNetFull
from .freezed_varnet_cascadedgaze import FreezedVarNetCascadedGaze
from .varnet_toy_restormer import VarNetToyRestormer
from .varnet_shared_mooshed import VarNetSharedMooshed
from .varnet_toy import VarNetToy

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


# class VarNetLogisticMambaOL(LitAdamW, VarNetLogisticMamba):
#     pass


class VarNetLogisticBoundOL(LitAdamW, VarNetLogisticBound):
    pass


class VarNetLogisticBoundFullOL(LitAdamW, VarNetLogisticBoundFull):
    pass


class VarNetFreezedSensNAFNetL1OL(LitAdamW, VarNetFreezedSensNAFNetL1):
    pass


class VarNetLogisticBoundL1OL(LitAdamW, VarNetLogisticBoundL1):
    pass


class FreezedVarNetNBNetOL(LitAdam, FreezedVarNetNBNet):
    pass


class FreezedVarNetRestormerOL(LitAdamW, FreezedVarNetRestormer):
    pass


class VarNetBoundMOEL1OL(LitAdamW, VarNetBoundMOEL1):
    pass


class VarNetFullOL(LitAdamW, VarNetFull):
    pass


class FreezedVarNetCascadedGazeOL(LitAdamW, FreezedVarNetCascadedGaze):
    pass

class VarNetToyRestormerOL(LitAdamW, VarNetToyRestormer):
    pass

class VarNetSharedMooshedOL(LitAdamW, VarNetSharedMooshed):
    pass

class VarNetToyOL(LitAdamW, VarNetToy):
    pass
