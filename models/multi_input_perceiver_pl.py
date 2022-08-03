import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import functional as F
from torch import nn

from models.multi_input_perceiver import MultiInputPerceiverParallel


class MultiInputPerceiverPL(pl.LightningModule):

    def __init__(
            self,
            model_config: MultiInputPerceiverParallel.Config,
            lr: float = 1e-3,
            num_words: int = 6336,
            text_dim: int = 2048,
            audio_dim: int = 256,
            audio_bias: bool = True,
    ):
        super().__init__()

        self.model = MultiInputPerceiverParallel(model_config)

        self.text_embedder = nn.Embedding(num_words, text_dim, padding_idx=0)
        self.audio_embedder = nn.Sequential(
            Rearrange("... t -> ... t 1"),
            nn.Linear(1, audio_dim, bias=audio_bias)
        )

        self.lr = lr

    def training_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        return self.__step(*batch, stage="train")

    def validation_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        return self.__step(*batch, stage="val")

    def test_step(self, batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        return self.__step(*batch, stage="test")

    def __step(self, text, audio, mask, labels, *, stage) -> STEP_OUTPUT:
        predicted = self.forward(text, audio, mask)

        labels = labels[mask]

        loss = F.cross_entropy(predicted, labels)
        self.log_dict({f'{stage}_loss': loss})
        return loss

    def forward(self, text, audio, mask):
        text = text[mask]
        text_pad_mask = text != 0
        text_embedded = self.text_embedder(text)

        audio = audio[mask]
        audio = audio.type(self.audio_embedder[1].weight.type())
        audio_silence_mask = audio != 0
        audio_embedded = self.audio_embedder(audio)

        data_list = [text_embedded, audio_embedded]
        mask_list = [text_pad_mask, audio_silence_mask]

        predicted = self.model(data_list, mask_list)

        return predicted

    def configure_optimizers(self):
        return torch.optim.Adam(lr=self.lr, params=self.model.parameters())
