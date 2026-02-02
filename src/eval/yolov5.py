import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class YOLOv5Classifier(pl.LightningModule):
    def __init__(self, num_classes=102, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr


        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path='yolov5s-cls.pt',
            force_reload=True
        )

        original_head = self.model.model.model[-1]
        in_features = original_head.linear.in_features

        self.model.model.model[-1].linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Calculate loss and accuracy for the test set
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log with 'test_' prefix
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)