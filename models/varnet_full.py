"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from models.base.unet import NormUnet
from models.base.varnet_primitive import SMEBlock, VarNetBlock
from models.external.NAFNet.NAFNet_arch import NAFNet
from models.lit.abstraction import LitBaseGrappaE2E


class VarNetFull(LitBaseGrappaE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 8,
        sens_chans: int = 4,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        nafnet_width: int = 64,
        nafnet_enc_blk_nums: List[int] = [2, 2, 4, 8],
        nafnet_middle_blk_num: int = 12,
        nafnet_dec_blk_nums: List[int] = [2, 2, 2, 2],
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_hidden_size: Number of hidden units for the rough processor.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """

        def make_sme_image_processor():
            return NormUnet(sens_chans, sens_pools)

        def make_cascade_image_processor():
            return NormUnet(chans, pools)

        super().__init__()

        self.sens_net = SMEBlock(image_processor=make_sme_image_processor())
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(image_processor=make_cascade_image_processor())
                for _ in range(num_cascades)
            ]
        )
        self.nafnet = NAFNet(
            img_channel=2,
            width=nafnet_width,
            enc_blk_nums=nafnet_enc_blk_nums,
            middle_blk_num=nafnet_middle_blk_num,
            dec_blk_nums=nafnet_dec_blk_nums,
        )
        self.postprocess = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa: torch.Tensor
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        varnet_result = fastmri.rss(
            fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1
        )
        varnet_result = self.image_space_crop(varnet_result)

        scaling_factor = torch.maximum(varnet_result.abs().max(), grappa.abs().max())
        scaling_factor = scaling_factor / 255.0

        varnet_result = varnet_result / scaling_factor
        grappa = grappa / scaling_factor

        nafnet_result = self.nafnet(torch.stack([varnet_result, grappa], dim=1))
        nafnet_result = self.postprocess(nafnet_result).squeeze(1) + varnet_result
        nafnet_result = nafnet_result * scaling_factor

        return nafnet_result
