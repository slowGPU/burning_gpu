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
from models.base.unet import NormUnet, Unet
from models.base.varnet_primitive import SMEBlock, VarNetBlock
from models.external.ResNet import ResNet
from models.lit.abstraction import LitBaseE2E


class LogisticResidualAbsProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        background_upper_bound: float,
        positive_lower_bound: float,
        decay_upper_bound: float,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.background_upper_bound = background_upper_bound
        self.positive_lower_bound = positive_lower_bound
        self.decay_upper_bound = decay_upper_bound

        self.resnet = ResNet(
            layers=[2, 2, 1, 1],
            planes=[16, 32, 64, 128],
            num_channels=2,
            num_classes=hidden_size,
        )

        self.fc_source_x = nn.Linear(hidden_size, 1)
        self.fc_source_y = nn.Linear(hidden_size, 1)

        self.fc_source_amplitude = nn.Linear(hidden_size, 1)
        self.fc_background_ratio = nn.Linear(hidden_size, 1)
        self.fc_decay = nn.Linear(hidden_size, 1)
        self.fc_cutoff = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        b, h, w, comp = x.shape

        assert comp == 2

        # b, h, w, comp -> b, comp, h, w
        x = x.permute(0, 3, 1, 2)

        # b, comp, h, w -> b, hid
        x = self.resnet(x)

        source_x = self.fc_source_x(x)
        source_y = self.fc_source_y(x)

        source_amplitude = (
            F.elu(self.fc_source_amplitude(x)) + 1 + self.positive_lower_bound
        )
        background_ratio = (
            F.sigmoid(self.fc_background_ratio(x)) * self.background_upper_bound
        )
        decay = F.sigmoid(self.fc_decay(x)) * self.decay_upper_bound
        cutoff = F.elu(self.fc_cutoff(x)) + 1 + self.positive_lower_bound

        pos_x = torch.linspace(-1, 1, w, device=x.device).view(1, 1, w)
        pos_y = torch.linspace(-h / w, h / w, h, device=x.device).view(1, h, 1)
        source_x = source_x.view(b, 1, 1)
        source_y = source_y.view(b, 1, 1)
        distance = torch.sqrt((source_x - pos_x) ** 2 + (source_y - pos_y) ** 2)

        decay = decay.view(b, 1, 1)
        source_amplitude = source_amplitude.view(b, 1, 1)
        background_ratio = background_ratio.view(b, 1, 1)
        cutoff = cutoff.view(b, 1, 1)

        out = (1 - F.tanh(decay * (distance - cutoff))) / (1 - F.tanh(-decay * cutoff))
        out = (1 - background_ratio) * out + background_ratio
        out = source_amplitude * out + 0.1
        out = out.unsqueeze(-1).unsqueeze(1)

        return out


class LogisticUnetProcessor(nn.Module):
    def __init__(
        self,
        chans: int,
        pools: int,
        hidden_size: int,
        background_upper_bound: float,
        positive_lower_bound: float,
        decay_upper_bound: float,
    ):
        super().__init__()

        self.unet = Unet(in_chans=2, out_chans=2, chans=chans, num_pool_layers=pools)
        self.logistic = LogisticResidualAbsProcessor(
            hidden_size, background_upper_bound, positive_lower_bound, decay_upper_bound
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, comp = x.shape

        assert c == 1 and comp == 2

        amp = self.logistic(x)

        # b, 1, h, w, 1 -> b, 1, h, w, 2
        amp = torch.cat((amp, torch.zeros_like(amp)), dim=-1)

        # b, 1, h, w, 2 -> b, 1, 2, h, w -> b, 2, h, w
        x = x.permute(0, 1, 4, 2, 3).reshape(b, c * comp, h, w)

        # b, 2, h, w -> b, 2, h', w' -> b, 2, h, w
        x, pad_params = self.pad(x)
        rot = self.unet(x)
        rot = self.unpad(rot, *pad_params)

        # b, 2, h, w -> b, 2, h, w, 1
        rot = rot.unsqueeze(-1)

        # b, 2, h, w, 1 -> b, 1, h, w, 2
        rot = rot.permute(0, 4, 2, 3, 1)
        rot = rot / fastmri.complex_abs(rot).unsqueeze(-1)

        return fastmri.complex_mul(amp, rot)


class VarNetLogisticBound(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 4,
        sens_hidden_size: int = 32,
        sens_background_upper_bound: float = 0.5,
        sens_positive_lower_bound: float = 0.05,
        sens_decay_upper_bound: float = 8,
        sens_chans: int = 4,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
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
            return LogisticUnetProcessor(
                sens_chans,
                sens_pools,
                sens_hidden_size,
                sens_background_upper_bound,
                sens_positive_lower_bound,
                sens_decay_upper_bound,
            )

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

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)
