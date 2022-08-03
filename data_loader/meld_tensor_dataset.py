from typing import Literal

import torch
from torch.utils.data import TensorDataset

from data_loader.meld_data_loader import MeldDataloader


class MeldTensorDataset:
    def __init__(
            self,
            category: Literal['sentiment', 'emotion'] = 'sentiment',
            modalities: Literal['text', 'audio', 'bimodal', 'bimodal_fused'] = 'bimodal',
            max_len: int = 50
    ):
        self.categories = category
        self.modalities = modalities

        self.data_loaders = []
        if modalities in ['text', 'bimodal']:
            text_dl = MeldDataloader(category, max_len)
            text_dl.load_text_data()
            self.data_loaders.append(text_dl)

        if modalities in ['audio', 'bimodal']:
            audio_dl = MeldDataloader(category, max_len)
            audio_dl.load_audio_data()
            self.data_loaders.append(audio_dl)

        if modalities in ['bimodal_fused']:
            fused_dl = MeldDataloader(category, max_len)
            fused_dl.load_bimodal_data()
            self.data_loaders.append(fused_dl)

        self._tensor_datasets = {"train": None, "val": None, "test": None}

    def get_tensor_dataset(self, stage: Literal['train', 'val', 'test']):
        if self._tensor_datasets[stage] is not None:
            return self._tensor_datasets[stage]

        features = []
        labels = []
        mask = []

        for dl in self.data_loaders:
            if stage == 'train':
                features.append(dl.train_dialogue_features)
                labels.append(dl.train_dialogue_label)
                mask.append(dl.train_mask)

            elif stage == 'val':
                features.append(dl.val_dialogue_features)
                labels.append(dl.val_dialogue_label)
                mask.append(dl.val_mask)

            elif stage == 'test':
                features.append(dl.test_dialogue_features)
                labels.append(dl.test_dialogue_label)
                mask.append(dl.test_mask)

        features = list(map(lambda x: torch.from_numpy(x), features))

        labels = list(map(lambda x: torch.from_numpy(x), labels[:1]))
        labels[0] = labels[0].argmax(dim=-1)

        mask = list(map(lambda x: torch.from_numpy(x), mask[:1]))
        mask[0] = mask[0].bool()

        self._tensor_datasets[stage] = TensorDataset(*features, *mask, *labels)
        return self._tensor_datasets[stage]


if __name__ == "__main__":
    td = MeldTensorDataset(category="emotion", modalities="bimodal", max_len=100)
    td.get_tensor_dataset(stage="val")