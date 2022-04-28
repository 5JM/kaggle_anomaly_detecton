import os
import mlflow
from pytorch_lightning.utilities.cli import LightningCLI
from model import Network
from dataset import DataModule
import itertools
import torch
import pandas as pd
import numpy as np

if __name__ == '__main__':
    os.chdir('E:\\jaemu\\daconAnomalyDetection')
    cli = LightningCLI(
        Network, DataModule, seed_everything_default=42, save_config_overwrite=True, run=False
    )

    preds = cli.trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path='checkpoints/epoch=49-step=25100.ckpt')

    img = list(itertools.chain(*[list(pred[0]) for pred in preds]))
    preds = torch.cat([pred[1] for pred in preds], dim = 0)
    pred_score = torch.argmax(preds, dim=1).detach().cpu().numpy()

    train_y = pd.read_csv("open/train_df.csv")
    train_labels = train_y["label"]
    label_unique = sorted(np.unique(train_labels))
    label_unique = {key: value for key, value in zip(label_unique, range(len(label_unique)))}

    label_decoder = {val: key for key, val in label_unique.items()}

    with open('open/sample_submission.csv', 'w') as f:
        f.write('index,label\n')
        for idx, pred_class in zip(img, pred_score):
            f.write(f'{idx},{label_decoder[pred_class]}\n')
