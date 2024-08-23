"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn as nn

from models.external.Restormer.restormer_arch import Restormer
from models.lit.abstraction import LitBaseGrappaE2E
# from fastmri import rss


class FreezedVarNetRestormer(LitBaseGrappaE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        varnet_path: Union[str, Path],
        varnet_with_grappa: bool = False,
        with_grappa: bool = True,
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

        super().__init__()

        self.varnet = torch.load(varnet_path)
        self.varnet_with_grappa = varnet_with_grappa

        self.restormer = Restormer(
            inp_channels=2 if with_grappa else 1,
            out_channels=1,
            dim=restormer_dim,
            num_blocks=restormer_num_blocks,
            num_refinement_blocks=restormer_num_refinement_blocks,
            heads=restormer_heads,
            ffn_expansion_factor=restormer_ffn_expansion_factor,
        )

        self.with_grappa = with_grappa

        self.varnet.eval()
        for params in self.varnet.parameters():
            params.requires_grad = False

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        grappa: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.varnet_with_grappa:
            varnet_result = self.varnet(masked_kspace, mask, grappa)
            varnet_result = self.image_space_crop(varnet_result)
        else:
            varnet_result = self.varnet(masked_kspace, mask)
            varnet_result = self.image_space_crop(varnet_result)

        if self.with_grappa:
            scaling_factor = torch.maximum(
                varnet_result.abs().max(), grappa.abs().max()
            )
            scaling_factor = scaling_factor / 255.0

            varnet_result = varnet_result / scaling_factor
            grappa = grappa / scaling_factor

            restormer_result = self.restormer(torch.stack([varnet_result, grappa], dim=1))
            restormer_result = restormer_result * scaling_factor
        else:
            scaling_factor = varnet_result.abs().max()
            scaling_factor = scaling_factor / 255.0

            varnet_result = varnet_result / scaling_factor

            restormer_result = self.restormer(varnet_result.unsqueeze(1))
            restormer_result = restormer_result * scaling_factor


        return restormer_result
