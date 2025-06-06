import torch
from braindecode.models.deep4 import Deep4Net
from lightning import LightningModule
from torch.nn import NLLLoss
from torch.optim import Adam


class Deep4LightningModule(LightningModule):
    def __init__(
        self,
        input_window_samples,
        n_channels,
        n_classes,
        lr=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = Deep4Net(
            in_chans=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            final_conv_length="auto",
        )
        self.loss_fn = NLLLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, *_ = batch
        y = y.long()
        logits = self(x)
        # logits = torch.log_softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, *_ = batch
        y = y.long()
        logits = self(x)
        # logits = torch.log_softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
