"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn

import fastmri
from models.base.unet import NormUnet
from models.base.varnet_primitive import VarNetBlock
from models.lit.abstraction import LitBaseGrappaE2E
from models.external.NAFNet.NAFNet_arch import NAFNet
from common.loss_function import SSIMLossWithL1



class VarNetFreezedSensNAFNetL1(LitBaseGrappaE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        sens_net_path: Union[str, Path],
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        nafnet_width: int = 32,
        nafnet_enc_blk_nums: List[int] = [2, 2, 4, 8],
        nafnet_middle_blk_num: int = 12,
        nafnet_dec_blk_nums: List[int] = [2, 2, 2, 2],
        l1_lambda: float = 0.1,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
        """

        def make_cascade_image_processor():
            return NormUnet(chans, pools)

        super().__init__()

        self.sens_net = torch.load(sens_net_path)
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

        self.sens_net.eval()
        for params in self.sens_net.parameters():
            params.requires_grad = False

        self.criterion = SSIMLossWithL1(lamb=l1_lambda)


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

        return self.nafnet(torch.stack([varnet_result, grappa], dim=1))[:, 1]
