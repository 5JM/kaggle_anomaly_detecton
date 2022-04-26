import timm
from pytorch_lightning import LightningModule
import torch
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss

class Network(LightningModule):
    # def forward(self, x):
    #     x = self.model(x)
    #     return x

    def __init__(self, model, lr, train_loss, valid_loss): # yml 사용할 파라미터를 선언
        super(Network, self).__init__()
        self.save_hyperparameters(ignore="model")
        self.model = timm.create_model(model, pretrained=True, num_classes=88)
        self.lr = lr
        if train_loss == 'crossE':
            self.train_loss = CrossEntropyLoss()
        if valid_loss == 'crossE':
            self.valid_loss = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adadelta(self.model.parameters(), lr=self.hparams.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.7)

        return (
            {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "monitor": "valid/acc",
                },
            }
        )

    def training_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.train_loss(pred, label)

        acc = sum(torch.argmax(pred, dim=1) == label) / len(img)

        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        self.log('train/acc', acc, on_epoch=True)
        self.log("train_lr", self.optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        pred = self.forward(img)
        loss = self.valid_loss(pred, label)

        acc = sum(torch.argmax(pred, dim=1) == label) / len(img)

        self.log('valid/loss', loss, prog_bar=True, on_epoch=True)
        self.log('valid/acc', acc, on_epoch=True)
        return torch.argmax(pred, dim=1), label, loss

    def validation_step_end(self, output_list):
        valid_pred = output_list[0]
        valid_label = output_list[1]
        valid_f1_score = f1_score(valid_label.detach().cpu().numpy(),
                                  valid_pred.detach().cpu().numpy(),
                                  average='macro')
        self.log('valid/f1_score', valid_f1_score)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img_idx, img = batch
        with torch.no_grad():
            pred = self.forward(img)
        pred = torch.softmax(pred, dim = -1)
        return img_idx, pred