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
from models.lit.abstraction import LitBaseE2E


class VarNetSharedMooshed(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_proceesors: int = 3,
        num_cascades: int = 12,
        sens_chans: int = 4,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        nafnet_width: int = 64,
        nafnet_enc_blk_nums: List[int] = [2, 2, 4, 8],
        nafnet_middle_blk_num: int = 12,
        nafnet_dec_blk_nums: List[int] = [2, 2, 2, 2],
        nafnet_fin_width: int = 64,
        nafnet_fin_enc_blk_nums: List[int] = [2, 2, 4, 8],
        nafnet_fin_middle_blk_num: int = 12,
        nafnet_fin_dec_blk_nums: List[int] = [2, 2, 2, 2],
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

        self.casacde_processor_list = nn.ModuleList(
            [make_cascade_image_processor() for _ in range(num_proceesors)]
        )
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    image_processor=self.casacde_processor_list[idx % num_proceesors]
                )
                for idx in range(num_cascades)
            ]
        )
        self.nafnet = NAFNet(
            img_channel=1,
            width=nafnet_width,
            enc_blk_nums=nafnet_enc_blk_nums,
            middle_blk_num=nafnet_middle_blk_num,
            dec_blk_nums=nafnet_dec_blk_nums,
        )

        self.nafnet_final = NAFNet(
            img_channel=1,
            width=nafnet_fin_width,
            enc_blk_nums=nafnet_fin_enc_blk_nums,
            middle_blk_num=nafnet_fin_middle_blk_num,
            dec_blk_nums=nafnet_fin_dec_blk_nums,
        )

        self.mooosh = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=False,
        )

        self.mooosh.weight.data.fill_(1.0 / 9.0)
        self.mooosh.eval()
        self.mooosh.requires_grad_(False)

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        varnet_result = fastmri.rss(
            fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1
        )
        varnet_result = self.image_space_crop(varnet_result)

        scaling_factor = varnet_result.abs().max()
        scaling_factor = scaling_factor / 255.0

        varnet_result = varnet_result / scaling_factor

        nafnet_result = self.nafnet(varnet_result.unsqueeze(1))
        nafnet_result = self.mooosh(nafnet_result)
        nafnet_result = self.nafnet_final(nafnet_result)

        nafnet_result = nafnet_result * scaling_factor
        return nafnet_result
