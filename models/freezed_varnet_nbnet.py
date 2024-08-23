"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn

from models.external.NAFNet.NAFNet_arch import NAFNet
from models.lit.abstraction import LitBaseGrappaE2E
from models.external.NBNet import UNetD


class FreezedVarNetNBNet(LitBaseGrappaE2E):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        varnet_path: Union[str, Path],
        nbnet_wf: int = 32,
        nbnet_depth: int = 5,
        nbnet_subspace_dim: int = 16,
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

        self.varnet = torch.load(varnet_path).cuda()
        
        self.nbnet = UNetD(in_chn=1, wf=nbnet_wf, depth=nbnet_depth, subspace_dim=nbnet_subspace_dim).cuda()

        self.varnet.eval()
        for params in self.varnet.parameters():
            params.requires_grad = False


    def forward(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa: torch.Tensor
    ) -> torch.Tensor:
        varnet_result = self.varnet(masked_kspace, mask, grappa)
        varnet_result = self.image_space_crop(varnet_result)
        
        # b, h, w -> b, 1, h, w
        varnet_result = varnet_result.unsqueeze(1)
        out = self.nbnet(varnet_result)
        out = out.squeeze(1)

        return out
