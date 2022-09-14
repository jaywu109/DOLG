import torch
import pytorch_lightning as pl
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.losses import ArcFaceLoss
from .lr import LinearWarmupCosineAnnealingLR
from .models.dolg import Dolg, DolgInfer


class BaselineModel(pl.LightningModule):

    def __init__(self,
                 nclasses,
                 finetune=False,
                 iter_per_epoch=8000,
                 init_lr=0.01,
                 num_epochs=12,
                 embedding_dim=512,
                 freeze_bn=False,
                 loss_config=None,
                 cs=1024,
                 global_only=False,
                 map_at_k=100,
                 optim='adamw'):
        super(BaselineModel, self).__init__()
        self.nclasses = nclasses
        self.finetune = finetune
        self.iter_per_epoch = iter_per_epoch
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.embedding_dim = embedding_dim
        self.freeze_bn = freeze_bn
        self.loss_config = loss_config
        self.cs = cs
        self.global_only = global_only
        self.criterion = ArcFaceLoss(scale=loss_config['scale'],
                                     margin=loss_config['margin'],
                                     num_classes=self.nclasses,
                                     embedding_size=self.embedding_dim)
        self.optim = optim
        self.map_at_k = map_at_k

        self.calculater = AccuracyCalculator(include=['precision_at_1', 'r_precision', 'mean_average_precision_at_r'], k=self.map_at_k)
        pretrained = False if self.finetune else True
        self.model = Dolg(self.embedding_dim, pretrained=pretrained, cs=self.cs, global_only=self.global_only)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch

        feature = self(x)

        loss = self.criterion(feature, y)

        sync_dist = self.loss_config['is_ddp']
        self.log('train/loss', loss, sync_dist=sync_dist)

        sch = self.lr_schedulers()
        sch.step()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        feature = self(x)
        return y, feature

    def validation_epoch_end(self, outputs):
        label, embedding = list(zip(*outputs))
        label = torch.cat(label)
        embedding = torch.cat(embedding)
        result = self.calculater.get_accuracy(embedding, embedding, label, label, True)

        # use sync_dist when ddp
        sync_dist = self.loss_config['is_ddp']
        self.log('valid/P@1', result['precision_at_1'], prog_bar=True, sync_dist=sync_dist)
        self.log('valid/RP', result['r_precision'], prog_bar=False, sync_dist=sync_dist)
        self.log('valid/MAP@R', result['mean_average_precision_at_r'], prog_bar=True, sync_dist=sync_dist)

    def configure_optimizers(self):
        param_groups = [{'params': self.model.parameters()}, {'params': self.criterion.parameters()}]
        if self.optim == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=self.init_lr)
        elif self.optim == 'sgd':
            optimizer = torch.optim.SGD(param_groups, lr=self.init_lr, momentum=0.9, weight_decay=0.0001)
        else:
            raise ValueError('Does not support `{}`'.format(self.optim))
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                     int(0.1 * self.num_epochs) * self.iter_per_epoch,
                                                     self.iter_per_epoch * self.num_epochs)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class BaselineModelInfer(pl.LightningModule):

    def __init__(self, embedding_dim=512, cs=1024, global_only=False):
        super(BaselineModelInfer, self).__init__()
        self.embedding_dim = embedding_dim
        self.cs = cs
        self.global_only = global_only
        self.model = DolgInfer(self.embedding_dim, pretrained=False, cs=self.cs, global_only=self.global_only)

    def forward(self, x):
        return self.model(x)
