import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn import functional as F

from models.multi_input_perceiver import MultiInputPerceiverParallel


class MultiInputPerceiverPL(pl.LightningModule):

    def __init__(
            self,
            model_config: MultiInputPerceiverParallel.Config,
            lr: float = 1e-3,
    ):
        super().__init__()
        self.model = MultiInputPerceiverParallel(model_config)
        self.lr = lr

    def training_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        return self.__step(*batch, stage="train")

    def validation_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        return self.__step(*batch, stage="val")

    def test_step(self, batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        return self.__step(*batch, stage="test")

    def __step(self, text, audio, mask, labels, *, stage) -> STEP_OUTPUT:
        predicted = self.forward(text, audio, mask)

        loss = F.cross_entropy(predicted, labels)
        self.log_dict({f'{stage}_loss': loss})
        return loss

    def forward(self, text, audio, mask):
        data_list = [text, audio]
        mask_list = [mask, mask]

        predicted = self.model(data_list, mask_list)

        return predicted

    def configure_optimizers(self):
        return torch.optim.Adam(lr=self.lr, params=self.model.parameters())
