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
from models.external.LightMUNet import LightMUNet
from models.base.varnet_primitive import SMEBlock, VarNetBlock
from models.external.ResNet import ResNet
from models.lit.abstraction import LitBaseE2E


class LightMNormUNetProcessor(nn.Module):
    def __init__(self, init_filters: int):
        super().__init__()

        self.unet = LightMUNet(
            spatial_dims=2, in_channels=2, out_channels=2, init_filters=init_filters
        )

    @staticmethod
    def pad(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    @staticmethod
    def unpad(
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]
    
    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, comp = x.shape

        assert c == 1 and comp == 2

        # b, 1, h, w, 2 -> b, 1, 2, h, w -> b, 2, h, w
        x = x.permute(0, 1, 4, 2, 3).reshape(b, c * comp, h, w)

        # b, 2, h, w -> b, 2, h', w' -> b, 2, h, w
        x, mean, std = self.norm(x)
        x, pad_params = self.pad(x)
        x = self.unet(x)
        x = self.unpad(x, *pad_params)
        x = self.unnorm(x, mean, std)

        # b, 2, h, w -> b, 2, h, w, 1
        x = x.unsqueeze(-1)

        # b, 2, h, w, 1 -> b, 1, h, w, 2
        x = x.permute(0, 4, 2, 3, 1)

        return x


class VarNetMamba(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 1,
        sens_init_filters: int = 8,
        init_filters: int = 8,
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
            return LightMNormUNetProcessor(sens_init_filters)    

        def make_cascade_image_processor():
            return LightMNormUNetProcessor(init_filters)

        super().__init__()

        self.sens_net = SMEBlock(image_processor=make_sme_image_processor())
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(image_processor=make_cascade_image_processor())
                for _ in range(num_cascades)
            ]
        )

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
