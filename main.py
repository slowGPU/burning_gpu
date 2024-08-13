import mlflow
from lightning.pytorch.cli import LightningCLI

from dataset import SliceDataModule, GrappaDataModule, SliceGrappaDataModule
from models import *  # noqa: F403

mlflow.autolog()

cli = LightningCLI()
