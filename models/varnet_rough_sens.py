"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import fastmri
from models.base.unet import NormUnet
from models.base.varnet_primitive import SMEBlock, VarNetBlock
from models.lit.abstraction import LitBaseE2E

from torchvision.models.resnet import resnet18


class RoughProcessor(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()

        self.hidden_size = hidden_size

        self.resnet = resnet18(num_classes=hidden_size)
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.fc_cos = nn.Linear(hidden_size, 1)
        self.fc_sin = nn.Linear(hidden_size, 1)
        self.fc_decay = nn.Linear(hidden_size, 2)
        self.fc_amplitude = nn.Linear(hidden_size, 2)
        self.fc_radius = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        b, h, w, comp = x.shape

        # b, h, w, comp -> b, comp, h, w
        x = x.permute(0, 3, 1, 2)

        # b, comp, h, w -> b, hid
        x = self.resnet(x)

        cos = self.fc_cos(x)
        sin = self.fc_sin(x)
        theta = torch.atan2(sin, cos)
        theta = torch.cat((theta, theta), dim=-1)

        decay = F.sigmoid(self.fc_decay(x)) * 5
        radius = F.elu(self.fc_radius(x)) + 1
        amplitude = self.fc_amplitude(x)

        # print(theta, decay, radius, amplitude)

        pos_x = (
            torch.linspace(-1, 1, w, device=x.device)
            .view(1, 1, w, 1)
            .expand(b, h, w, 2)
        )
        pos_y = (
            torch.linspace(-1, 1, h, device=x.device)
            .view(1, h, 1, 1)
            .expand(b, h, w, 2)
        )
        theta = theta.view(b, 1, 1, 2).expand(b, h, w, 2)
        decay = decay.view(b, 1, 1, 2).expand(b, h, w, 2)
        radius = radius.view(b, 1, 1, 2).expand(b, h, w, 2)
        amplitude = amplitude.view(b, 1, 1, 2).expand(b, h, w, 2)

        distance = torch.sqrt(
            (radius * torch.cos(theta) - pos_x) ** 2
            + (radius * torch.sin(theta) - pos_y) ** 2
        )

        return (amplitude * torch.exp(-decay * distance)).unsqueeze(1)


class VarNetRoughSens(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_hidden_size: int = 32,
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
            return RoughProcessor(sens_hidden_size)

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
