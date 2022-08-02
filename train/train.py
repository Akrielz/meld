from typing import Literal

import torch
import pytorch_lightning as pl
import yaml

from data_loader.meld_data_module import MeldDataModule
from models.multi_input_perceiver import MultiInputPerceiverParallel
from models.multi_input_perceiver_pl import MultiInputPerceiverPL


def model_train(
        mip_config: MultiInputPerceiverParallel.Config,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 4,
        category: Literal['sentiment', 'emotion'] = 'emotion',
        max_len: int = 50
):
    gpus = -1 if torch.cuda.device_count() else 0

    if gpus == 0:
        print("Running on CPU")
    else:
        print("Running on GPU")

    trainer = pl.Trainer(
        precision=32,
        reload_dataloaders_every_n_epochs=1,
        gpus=gpus
    )

    pl_module = MultiInputPerceiverPL(
        model_config=mip_config,
        lr=lr
    )

    datamodule = MeldDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        category=category,
        max_len=max_len
    )

    trainer.fit(pl_module, datamodule)


if __name__ == '__main__':
    mip_config = MultiInputPerceiverParallel.Config(
        input_channels=[64, 64],
        input_axis=[1, 1],

        num_classes=7
    )

    model_train(
        mip_config=mip_config,
        lr=1e-3,
        batch_size=64,
        num_workers=0,
        category='emotion',
        max_len=50
    )

