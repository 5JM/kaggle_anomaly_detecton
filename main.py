import mlflow
from pytorch_lightning.utilities.cli import LightningCLI
from model import Network
from dataset import DataModule
import os

def cli_main():
    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = LightningCLI(
        Network, DataModule, seed_everything_default=42, save_config_overwrite=True, run=False
    )
    with mlflow.start_run():
        mlflow.log_artifact(os.path.abspath(__file__), 'source code')
        cli.trainer.fit(cli.model, datamodule=cli.datamodule)

        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    os.chdir('E:\\jaemu\\daconAnomalyDetection')
    mlflow.set_tracking_uri('http://117.16.123.14:9650')  # set up connection
    mlflow.set_experiment('jaemu_baseline')  # set the experiment
    mlflow.pytorch.autolog()
    cli_main()