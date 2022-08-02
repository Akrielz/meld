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
        return self.__step__(*batch, mode="train")

    def validation_step(self, batch, batch_idx, *args) -> STEP_OUTPUT:
        return self.__step__(*batch, mode="val")

    def test_step(self, batch, batch_idx, **kwargs) -> STEP_OUTPUT:
        return self.__step__(*batch, mode="test")

    def __step(self, text, audio, mask, labels, *, mode) -> STEP_OUTPUT:
        mask_list = [mask, mask]
        data_list = [text, audio]
        predicted = self.model(data_list, mask_list)

        loss = F.cross_entropy(predicted, labels)
        self.log_dict({f'{mode}_loss': loss})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(lr=self.lr, params=self.model.parameters())
