"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

import fastmri
from models.base.unet import NormUnet
from models.base.varnet_primitive import VarNetBlock
from models.lit.abstraction import LitBaseE2E


class VarNetFreezedSensToy(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        sens_net_path: Union[str, Path],
        num_processors: int = 2,
        num_cascades: int = 8,
        chans: int = 18,
        pools: int = 4,
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
        self.image_processor_list = [
            make_cascade_image_processor() for _ in range(num_processors)
        ]
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(
                    image_processor=self.image_processor_list[idx % num_processors]
                )
                for idx in range(num_cascades)
            ]
        )

        self.sens_net.eval()
        for params in self.sens_net.parameters():
            params.requires_grad = False

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
