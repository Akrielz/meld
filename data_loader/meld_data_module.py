from typing import Literal

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data_loader.meld_tensor_dataset import MeldTensorDataset


class MeldDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 64,
            num_workers: int = 4,
            categories: Literal['sentiment', 'emotion'] = 'sentiment',
            modalities: Literal['text', 'audio', 'bimodal', 'bimodal_fused'] = 'bimodal',
            max_len: int = 50
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tensor_dataset = MeldTensorDataset(categories, modalities, max_len)

    def train_dataloader(self):
        return DataLoader(
            self.tensor_dataset.get_tensor_dataset('train'),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.tensor_dataset.get_tensor_dataset('val'),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.tensor_dataset.get_tensor_dataset('test'),
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )