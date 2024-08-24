"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn

from models.external.NAFNet.NAFNet_arch import NAFNet
from models.lit.abstraction import LitBaseE2E
# from fastmri import rss


class L2LossModule(nn.Module):
    def forward(
        self, recon: torch.Tensor, target: torch.Tensor, maximum: torch.Tensor
    ) -> torch.Tensor:
        recon = recon / maximum
        target = target / maximum

        return torch.mean((recon - target) ** 2)


class FreezedVarNetSingleNAFNet(LitBaseE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        varnet_path: Union[str, Path],
        nafnet_width: int = 64,
        nafnet_enc_blk_nums: List[int] = [2, 2, 8, 28],
        nafnet_middle_blk_num: int = 16,
        nafnet_dec_blk_nums: List[int] = [4, 8, 8, 2],
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
        self.nafnet = NAFNet(
            img_channel=1,
            width=nafnet_width,
            enc_blk_nums=nafnet_enc_blk_nums,
            middle_blk_num=nafnet_middle_blk_num,
            dec_blk_nums=nafnet_dec_blk_nums,
        )

        self.varnet.eval()
        for params in self.varnet.parameters():
            params.requires_grad = False

        # self.criterion = L2LossModule()

    def forward(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        varnet_result = self.varnet(masked_kspace, mask)
        varnet_result = self.image_space_crop(varnet_result)

        scaling_factor = varnet_result.abs().max()
        scaling_factor = scaling_factor / 255.0
        # scaling_factor = 1.0

        varnet_result = varnet_result / scaling_factor

        nafnet_result = self.nafnet(
                varnet_result.unsqueeze(1)
                )
        nafnet_result = nafnet_result * scaling_factor

        return nafnet_result
        # return nafnet_result.mean(dim=1, keepdim=False)
        # return rss(self.nafnet(varnet_result.unsqueeze(1)), dim=1)
