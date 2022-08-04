from typing import Literal

import torch
import pytorch_lightning as pl

from data_loader.meld_data_module import MeldDataModule
from models.multi_input_perceiver import MultiInputPerceiverParallel
from models.multi_input_perceiver_pl import MultiInputPerceiverPL

NUM_WORDS = 6336
MAX_LEN = 88


def model_train(
        mip_config: MultiInputPerceiverParallel.Config,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 4,
        category: Literal['sentiment', 'emotion'] = 'emotion',
        max_len: int = 50,
        text_dim: int = 2048,
        audio_dim: int = 256,
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
        lr=lr,
        num_words=NUM_WORDS,
        text_dim=text_dim,
        audio_dim=audio_dim,
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
        input_channels=[1024, 64],
        input_axis=[1, 1],

        latent_dim=128,
        num_latents=128,

        depth=2,

        cross_heads=8,
        cross_dim_head=64,

        self_per_cross_attn=8,
        latent_heads=8,
        latent_dim_head=64,

        attn_dropout=0.2,
        ff_dropout=0.2,
        merged_dropout=0.2,

        max_freq=10,
        num_freq_bands=6,
        fourier_encode_data=True,

        num_classes=7,
        final_classifier_head=True,
    )

    model_train(
        mip_config=mip_config,
        lr=1e-3,
        batch_size=4,
        num_workers=4,
        category='emotion',
        max_len=MAX_LEN,
        text_dim=1024,
        audio_dim=64,
    )
