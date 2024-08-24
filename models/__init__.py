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
from .grappa_nafnet import GrappaNAFNet
from .grappa_conv_nafnet import GrappaConvNAFNet
from .grappa_conv3_nafnet import GrappaConv3NAFNet
from .grappa_nafnet_mse import GrappaNAFNetMSE
from .grappa_conv_nafnet_mse import GrappaConvNAFNetMSE
from .freezed_varnet_nafnet import FreezedVarNetNAFNet
from .varnet_logistic_mamba import VarNetLogisticMamba
from .varnet_logistic_bound import VarNetLogisticBound
from .varnet_logistic_bound_full import VarNetLogisticBoundFull
from .varnet_mamba import VarNetMamba
from .varnet_full import VarNetFull
from .varnet_full_single_grappa import VarNetFullSingleGrappa

from .freezed_varnet_single_nafnet import FreezedVarNetSingleNAFNet

from .varnet_freezed_sens_toy import VarNetFreezedSensToy
from .varnet_freezed_sens_toy_nafnet import VarNetFreezedSensToyNAFNet

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

class GrappaConvNAFNetOL(LitAdamW, GrappaConvNAFNet):
    pass

class GrappaConv3NAFNetOL(LitAdamW, GrappaConv3NAFNet):
    pass

class GrappaNAFNetMSEOL(LitAdamW, GrappaNAFNetMSE):
    pass

class VarNetMambaOL(LitAdamW, VarNetMamba):
    pass

class VarNetFullOL(LitAdamW, VarNetFull):
    pass

class VarNetFullSingleGrappaOL(LitAdamW, VarNetFullSingleGrappa):
    pass


class FreezedVarNetSingleNAFNetOL(LitAdamW, FreezedVarNetSingleNAFNet):
    pass

class VarNetFreezedSensToyOL(LitAdamW, VarNetFreezedSensToy):
    pass

class VarNetFreezedSensToyNAFNetOL(LitAdamW, VarNetFreezedSensToyNAFNet):
    pass