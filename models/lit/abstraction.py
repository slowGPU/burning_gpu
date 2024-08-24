from abc import ABCMeta, abstractmethod
from typing import Final, Sequence, Union

import lightning as L
import torch
import torch.nn as nn

import fastmri


class LitBase(L.LightningModule, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters(ignore=self.ignore_hparams())

    @staticmethod
    def image_space_crop(image: torch.Tensor) -> torch.Tensor:
        CROP_WIDTH: Final[int] = 384
        CROP_HEIGHT: Final[int] = 384

        return fastmri.data.transforms.center_crop(image, (CROP_HEIGHT, CROP_WIDTH))

    @staticmethod
    def ignore_hparams() -> Union[str, Sequence[str], None]:
        return None

    @abstractmethod
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass


class LitBaseE2E(LitBase, metaclass=ABCMeta):
    model: nn.Module = None

    def __init__(self):
        super().__init__()

        self.criterion = fastmri.losses.SSIMLoss()

    @abstractmethod
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx):
        mask, masked_kspace, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))

        loss = self.criterion(recon, target, maximum)

        self.log("loss", loss, batch_size=masked_kspace.size(0), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mask, masked_kspace, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))

        loss = self.criterion(recon, target, maximum)

        self.log("val_loss", loss, batch_size=masked_kspace.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        mask, masked_kspace, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))

        loss = self.criterion(recon, target, maximum)

        self.log("test_loss", loss, batch_size=masked_kspace.size(0))

        return loss


class LitBaseHybrid(LitBase, metaclass=ABCMeta):
    model: nn.Module = None

    def __init__(self):
        super().__init__()

        self.criterion = fastmri.losses.SSIMLoss()

        self.automatic_optimization = False

    @abstractmethod
    def forward_sme(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_recon(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_recon(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        sme_opt, recon_opt = self.optimizers()

        mask, masked_kspace, target, maximum, _, _ = batch

        # sensitivity map estimation
        recon_sme = self.image_space_crop(self.forward_sme(masked_kspace, mask))
        loss_sme = self.criterion(recon_sme, target, maximum)

        sme_opt.zero_grad()
        self.manual_backward(loss_sme)
        sme_opt.step()

        # reconstruction
        recon = self.image_space_crop(self.forward_recon(masked_kspace, mask))
        loss_recon = self.criterion(recon, target, maximum)

        recon_opt.zero_grad()
        self.manual_backward(loss_recon)
        recon_opt.step()

        self.log_dict(
            {"loss_sme": loss_sme, "loss_recon": loss_recon},
            batch_size=masked_kspace.size(0),
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))
        loss = self.criterion(recon, target, maximum)

        self.log("val_loss", loss, batch_size=masked_kspace.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        masked_kspace, mask, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))
        loss = self.criterion(recon, target, maximum)

        self.log("test_loss", loss, batch_size=masked_kspace.size(0))

        return loss


class LitBaseAdversarial(L.LightningModule, metaclass=ABCMeta):
    model: nn.Module = None

    def __init__(self):
        super().__init__()

        self.criterion_image = fastmri.losses.SSIMLoss()
        self.criterion_adv = nn.BCEWithLogitsLoss()

        self.save_hyperparameters(ignore=self.ignore_hparams())

        self.automatic_optimization = False

    @staticmethod
    def ignore_hparams():
        return None

    @abstractmethod
    def forward_sme(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_disc(self, target: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_gen(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_gen(masked_kspace, mask)

    def training_step(self, batch, batch_idx):
        sme_opt, disc_opt, gen_opt = self.optimizers()

        mask, masked_kspace, target, maximum, _, _ = batch

        # sensitivity map estimation
        recon_sme = self.image_space_crop(self.forward_sme(masked_kspace, mask))
        loss_sme = self.criterion_image(recon_sme, target, maximum)

        sme_opt.zero_grad()
        self.manual_backward(loss_sme)
        sme_opt.step()

        recon = self.image_space_crop(self.forward_gen(masked_kspace, mask))

        # discriminator
        disc_real = self.forward_disc(target)
        disc_fake = self.forward_disc(recon.detach())
        loss_disc_real = self.criterion_adv(disc_real, torch.ones_like(disc_real))
        loss_disc_fake = self.criterion_adv(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2

        disc_opt.zero_grad()
        self.manual_backward(loss_disc)
        disc_opt.step()

        # generator
        disc_fake = self.forward_disc(recon)
        loss_gen = self.criterion_adv(disc_fake, torch.ones_like(disc_fake))

        gen_opt.zero_grad()
        self.manual_backward(loss_gen)
        gen_opt.step()

        self.log_dict(
            {
                "loss_sme": loss_sme,
                "loss_disc_real": loss_disc_real,
                "loss_disc_fake": loss_disc_fake,
                "loss_disc": loss_disc,
                "loss_gen": loss_gen,
            },
            batch_size=masked_kspace.size(0),
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        mask, masked_kspace, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))
        loss = self.criterion_image(recon, target, maximum)

        self.log("val_loss", loss, batch_size=masked_kspace.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        mask, masked_kspace, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask))
        loss = self.criterion_image(recon, target, maximum)

        self.log("test_loss", loss, batch_size=masked_kspace.size(0))

        return loss


class LitBaseGrappa(L.LightningModule, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

        self.save_hyperparameters(ignore=self.ignore_hparams())

    @staticmethod
    def image_space_crop(image: torch.Tensor) -> torch.Tensor:
        CROP_WIDTH: Final[int] = 384
        CROP_HEIGHT: Final[int] = 384

        return fastmri.data.transforms.center_crop(image, (CROP_HEIGHT, CROP_WIDTH))

    @staticmethod
    def ignore_hparams() -> Union[str, Sequence[str], None]:
        return None

    @abstractmethod
    def forward(
        self, masked_kspace: torch.Tensor, mask: torch.Tensor, grappa: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):
        pass


class LitBaseGrappaE2E(LitBaseGrappa, metaclass=ABCMeta):
    model: nn.Module = None

    def __init__(self):
        super().__init__()

        self.criterion = fastmri.losses.SSIMLoss()

    def training_step(self, batch, batch_idx):
        mask, masked_kspace, grappa, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask, grappa))

        loss = self.criterion(recon, target, maximum)

        self.log("loss", loss, batch_size=masked_kspace.size(0), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mask, masked_kspace, grappa, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask, grappa))

        loss = self.criterion(recon, target, maximum)

        self.log("val_loss", loss, batch_size=masked_kspace.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        mask, masked_kspace, grappa, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask, grappa))

        loss = self.criterion(recon, target, maximum)

        self.log("test_loss", loss, batch_size=masked_kspace.size(0))

        return loss

class LitBaseGrappaMSEE2E(LitBaseGrappa, metaclass=ABCMeta):
    model: nn.Module = None

    def __init__(self):
        super().__init__()

        # self.criterion = fastmri.losses.SSIMLoss()
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        mask, masked_kspace, grappa, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask, grappa))

        loss = self.criterion(recon/maximum, target/maximum)

        self.log("loss", loss, batch_size=masked_kspace.size(0), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mask, masked_kspace, grappa, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask, grappa))

        loss = self.criterion(recon/maximum, target/maximum)

        self.log("val_loss", loss, batch_size=masked_kspace.size(0))

        return loss

    def test_step(self, batch, batch_idx):
        mask, masked_kspace, grappa, target, maximum, _, _ = batch

        recon = self.image_space_crop(self.forward(masked_kspace, mask, grappa))

        loss = self.criterion(recon/maximum, target/maximum)

        self.log("test_loss", loss, batch_size=masked_kspace.size(0))

        return loss