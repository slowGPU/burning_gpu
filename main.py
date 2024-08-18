import mlflow
from lightning.pytorch.cli import LightningCLI

from dataset import SliceDataModule, GrappaDataModule, SliceGrappaDataModule  # noqa: F401
from models import *  # noqa: F403

mlflow.autolog()

cli = LightningCLI()
