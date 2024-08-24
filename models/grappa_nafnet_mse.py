from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn

import fastmri
from models.base.unet import NormUnet
from models.base.varnet_primitive import VarNetBlock
from models.lit.abstraction import LitBaseGrappaMSEE2E
from models.external.NAFNet.NAFNet_arch import NAFNet

class GrappaNAFNetMSE(LitBaseGrappaMSEE2E):
    def __init__(
        self,
        nafnet_width: int = 16,
        nafnet_enc_blk_nums: List[int] = [1, 1, 28],
        nafnet_middle_blk_num: int = 1,
        nafnet_dec_blk_nums: List[int] = [1, 1, 1],
    ):
        super().__init__()
        self.nafnet = NAFNet(
            img_channel=1,
            width=nafnet_width,
            enc_blk_nums=nafnet_enc_blk_nums,
            middle_blk_num=nafnet_middle_blk_num,
            dec_blk_nums=nafnet_dec_blk_nums,
        )
    
    def forward(
        self, _masked_kspace, _mask, grappa: torch.Tensor
    ) -> torch.Tensor:
        return self.nafnet(grappa.unsqueeze(1)).squeeze(1)