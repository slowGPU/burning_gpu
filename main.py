import mlflow
from lightning.pytorch.cli import LightningCLI

from dataset import SliceDataModule
from models import *  # noqa: F403

mlflow.autolog()

cli = LightningCLI(datamodule_class=SliceDataModule)
