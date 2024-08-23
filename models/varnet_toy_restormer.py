"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn

import fastmri
from typing import List
from models.base.unet import NormUnet
from models.base.varnet_primitive import SMEBlock, VarNetBlock
from models.lit.abstraction import LitBaseE2E

from models.external.Restormer.restormer_arch import Restormer


class VarNetToyRestormer(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        restormer_dim: int = 48,
        restormer_num_blocks: List[int] = [4, 6, 6, 8],
        restormer_num_refinement_blocks: int = 4,
        restormer_heads: List[int] = [1, 2, 4, 8],
        restormer_ffn_expansion_factor: float = 2.66,
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

        def make_sme_image_processor():
            return NormUnet(sens_chans, sens_pools)

        def make_cascade_image_processor():
            return NormUnet(chans, pools)

        super().__init__()

        self.sens_net = SMEBlock(image_processor=make_sme_image_processor())
        self.restormer = Restormer(
            inp_channels=1,
            out_channels=1,
            dim=restormer_dim,
            num_blocks=restormer_num_blocks,
            num_refinement_blocks=restormer_num_refinement_blocks,
            heads=restormer_heads,
            ffn_expansion_factor=restormer_ffn_expansion_factor,
        )

        self.cascades_image_processor = make_cascade_image_processor()
        self.cascades = nn.ModuleList(
            [
                VarNetBlock(image_processor=self.cascades_image_processor)
                for _ in range(num_cascades)
            ]
        )

    def forward(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
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

        restormer_result = self.restormer(varnet_result.unsqueeze(1))
        restormer_result = restormer_result * scaling_factor

        return restormer_result
