import mlflow
from lightning.pytorch.cli import LightningCLI

from dataset import SliceDataModule, GrappaDataModule, SliceGrappaDataModule  # noqa: F401
from models import *  # noqa: F403

mlflow.pytorch.autolog(checkpoint_save_best_only=False)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

cli = LightningCLI()
