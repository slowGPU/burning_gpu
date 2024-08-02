"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import fastmri
import torch
import torch.nn as nn

from models.base.unet import NormUnet
from models.base.varnet_primitive import SMEBlock, VarNetBlock
from models.lit.abstraction import LitBaseHybrid


class VarNetHybridFreeze(LitBaseHybrid):
    def __init__(
        self,
        num_shared_cascades: int = 0,
        num_sme_enc_cascades: int = 0,
        num_sme_dec_cascades: int = 4,
        num_recon_enc_cascades: int = 0,
        num_recon_dec_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_shared_cascades: Number of cascades (i.e., layers) for the shared translator.
            num_sme_enc_cascades: Number of cascades (i.e., layers) for the encoder of the sens net.
            num_sme_dec_cascades: Number of cascades (i.e., layers) for the decoder of the sens net.
            num_recon_enc_cascades: Number of cascades (i.e., layers) for the encoder of the recon net.
            num_recon_dec_cascades: Number of cascades (i.e., layers) for the decoder of the recon net.
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

        def make_cascades(num_cascades: int):
            return nn.ModuleList(
                [
                    VarNetBlock(image_processor=make_cascade_image_processor())
                    for _ in range(num_cascades)
                ]
            )

        super().__init__()
        self.sens_net = SMEBlock(image_processor=make_sme_image_processor())

        self.sme_enc_cascades = make_cascades(num_sme_enc_cascades)
        self.sme_dec_cascades = make_cascades(num_sme_dec_cascades)
        self.recon_enc_cascades = make_cascades(num_recon_enc_cascades)
        self.recon_dec_cascades = make_cascades(num_recon_dec_cascades)
        self.shared_trans_cascades = make_cascades(num_shared_cascades)

    def forward_primitive(
        self, cascades: nn.ModuleList, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in cascades:
            kspace_pred = cascade(kspace_pred, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

    def sme_parameters(self):
        return [
            *self.sens_net.parameters(),
            *self.sme_enc_cascades.parameters(),
            *self.shared_trans_cascades.parameters(),
            *self.sme_dec_cascades.parameters(),
        ]

    def recon_parameters(self):
        return [
            *self.recon_enc_cascades.parameters(),
            *self.shared_trans_cascades.parameters(),
            *self.recon_dec_cascades.parameters(),
        ]

    def forward_sme(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        for params in self.sens_net.parameters():
            params.requires_grad = True

        self.forward_primitive(
            [*self.sme_enc_cascades, *self.shared_cascades, *self.sme_dec_cascades],
            masked_kspace,
            mask,
        )

    def forward_recon(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        for params in self.sens_net.parameters():
            params.requires_grad = False

        self.forward_primitive(
            [*self.recon_enc_cascades, *self.shared_cascades, *self.recon_dec_cascades],
            masked_kspace,
            mask,
        )
