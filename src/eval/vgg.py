import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class VGG19Classifier(pl.LightningModule):
    def __init__(self, num_classes=102, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate


        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        #add the last classification head on top of the current one 
        out_features = self.model.classifier[6].out_features
        self.model.classifier.append(nn.Linear(out_features, num_classes))

        #freeze the feartures
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

 