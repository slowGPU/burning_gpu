"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import List, Union, Optional

import torch
import torch.nn as nn

from models.external.CascadedGaze.CGNetMultiHead_arch import CascadedGaze
from models.lit.abstraction import LitBaseGrappaE2E
# from fastmri import rss


class FreezedVarNetCascadedGaze(LitBaseGrappaE2E):
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
        cg_width: int = 60,
        cg_enc_blk_nums: List[int] = [2, 2, 4, 6],
        cg_middle_blk_num: int = 10,
        cg_dec_blk_nums: List[int] = [2, 2, 2, 2],
        cg_GCE_CONVS_nums: List[int] = [3, 3, 2, 2],
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

        self.cascadedgaze = CascadedGaze(
            img_channel=2 if with_grappa else 1,
            width=cg_width,
            enc_blk_nums=cg_enc_blk_nums,
            middle_blk_num=cg_middle_blk_num,
            dec_blk_nums=cg_dec_blk_nums,
            GCE_CONVS_nums=cg_GCE_CONVS_nums
        )

        self.postprocess = (
            nn.Conv2d(
                in_channels=2,
                out_channels=1,
                kernel_size=3,
                padding=1,
                stride=1,
                groups=1,
                bias=True,
            )
            if with_grappa
            else nn.Identity()
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

            cg_result = self.cascadedgaze(torch.stack([varnet_result, grappa], dim=1))
            cg_result = self.postprocess(cg_result).squeeze(1) + varnet_result
            cg_result = cg_result * scaling_factor
        else:
            scaling_factor = varnet_result.abs().max()
            scaling_factor = scaling_factor / 255.0

            varnet_result = varnet_result / scaling_factor

            cg_result = self.cascadedgaze(varnet_result.unsqueeze(1))
            cg_result = cg_result * scaling_factor

        return cg_result
