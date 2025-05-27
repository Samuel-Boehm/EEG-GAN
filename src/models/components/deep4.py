import lightning.pytorch as pl
import torch
import torchmetrics
from braindecode.models.deep4 import Deep4Net


class Deep4LightningModule(pl.LightningModule):
    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples,
        ds_metadata,
        final_conv_length="auto",
        learning_rate=1e-3,
        n_datasets=4,
        use_all_heads: bool = False,
        use_individual_heads: bool = True,
        reduce_metrics: bool = True,
        mask_logits_during_validation: bool = False,
        *args,
        **kwargs,
    ):
        super(Deep4LightningModule, self).__init__()

        self.model = Deep4Net(
            in_chans=in_chans,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            final_conv_length=final_conv_length,
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=n_classes
        )

        def train_step(batch, batch_idx):
            x, y = batch
            logits = self.model(x)
            loss = self.loss(logits, y)
            acc = self.train_accuracy(logits, y)
            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", acc, prog_bar=True)
            return loss

        def validation_step(batch, batch_idx):
            x, y = batch
            logits = self.model(x)
            loss = self.loss(logits, y)
            acc = self.val_accuracy(logits, y)
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)
            return {"val_loss": loss, "val_acc": acc}

        def test_step(batch, batch_idx):
            x, y = batch
            logits = self.model(x)
            loss = self.loss(logits, y)
            acc = self.train_accuracy(logits, y)
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", acc, prog_bar=True)
            return {"test_loss": loss, "test_acc": acc}

        def configure_optimizers():
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return ([optimizer],)
