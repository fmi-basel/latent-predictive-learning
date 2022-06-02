import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from argparse import ArgumentParser

from models.networks import LPLNet
from metrics.losses import cosine_push_loss, cosine_pull_loss


def _stack_spatial_features(a):
    b, c, w, h = a.shape
    a_stacked = a.view(b, c, -1).transpose(1, 2).reshape(-1, c)
    return a_stacked


class LPL(pl.LightningModule):
    """

    Example::
        model = LPL(num_classes=10, hyperparams_dict)
        dm = CIFAR10DataModule()
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)
        trainer = pl.Trainer()
        trainer.fit(model, dm)

    """

    def __init__(self,
                 num_classes,
                 train_end_to_end: bool = False,
                 use_projector_mlp: bool = False,
                 projection_size: int = 256,
                 mlp_hidden_size: int = 2048,
                 optimizer: str = 'adam',
                 no_pooling: bool = False,
                 stale_estimates: bool = False,
                 no_biases: bool = False,
                 batch_size: int = 1024,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1.5e-6,
                 max_epochs: int = 1000,
                 pull_coeff: float = 1.0,
                 push_coeff: float = 1.0,
                 decorr_coeff: float = 10.0,
                 warmup_epochs: int = 10,
                 start_lr: float = 0.0,
                 final_lr: float = 1e-6,
                 base_image_size: int = 32,
                 **kwargs):

        super(LPL, self).__init__()
        self.save_hyperparameters()

        self.network = LPLNet(train_end_to_end, use_projector_mlp, projection_size, mlp_hidden_size, base_image_size=base_image_size, no_biases=no_biases)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.optimizer = optimizer

        self.num_classes = num_classes
        self.z_dim = self.network.feature_size

        self.predictive_loss = F.mse_loss
        self.push_loss_fn = self.hebbian_loss

        if no_pooling:
            assert not train_end_to_end, 'End-to-end training with unpooled feature maps makes no sense, pretty sure you didn\'t want to do this'
            assert not use_projector_mlp, 'Projector networks for unpooled feature maps would be too large to fit into memory; currently throws an error'

        if stale_estimates:
            assert not no_pooling, 'Stale estimates currently not supported for unpooled training'
            self.first_epoch_flag = True
            self.mean_estimates = []
            self.var_estimates = []

    def forward(self, x):
        # Returns pooled features from final conv layer of the network
        y, _, _ = self.network(x)
        return y

    def pull_loss_fn(self, a, b):
        return self.predictive_loss(a, b)

    def hebbian_loss(self, a, stale_var=None, stale_mean=None):
        epsilon = 1e-4
        if stale_var is None:
            a_center = a.mean(dim=0).detach()
            variance = ((a - a_center) ** 2).sum(dim=0) / (a.shape[0] - 1)
            loss = -torch.log(variance + epsilon).mean()
        else:
            loss = -(1.0/(stale_var + epsilon) * (a - stale_mean)**2).sum(dim=0).mean() / (a.shape[0] - 1)
        return loss

    def decorr_loss(self, a):
        a_center = a.mean(dim=0).detach()
        a_centered = a - a_center
        cov = torch.einsum('ij,ik->jk', a_centered, a_centered).fill_diagonal_(0) / (a.shape[0] - 1)
        loss = torch.sum(cov ** 2) / (cov.shape[0] ** 2 - cov.shape[0])
        return loss
    
    def loss_step(self, batch):
        # This is the core logic of the lightning module 

        (img_1, img_2, _), _ = batch

        _, fm1, z1 = self.network(img_1)
        _, fm2, z2 = self.network(img_2)

        pull_loss = torch.zeros(len(z1))
        push_loss = torch.zeros_like(pull_loss)
        decorr_loss = torch.zeros_like(pull_loss)

        idxes = torch.randperm(len(z2[0]))

        for i in range(len(z1)):

            # Pull loss
            if self.hparams.no_pooling:
                pull_loss[i] = self.predictive_loss(fm1[i], fm2[i])
            else:
                if self.hparams.shuffle_positives:
                    pull_loss[i] = self.predictive_loss(z1[i], z2[i][idxes])
                else:
                    pull_loss[i] = self.predictive_loss(z1[i], z2[i])

            # Stale estimates of mean and variance
            stale_mean = None
            stale_var = None
            if self.hparams.stale_estimates and not self.first_epoch_flag:
                stale_mean = self.mean_estimates[i]
                stale_var = self.var_estimates[i]

            # Push loss
            if self.hparams.no_pooling:
                push_loss[i] = 0.5 * (self.push_loss_fn(fm1[i]) + self.push_loss_fn(fm2[i]))
            else:
                push_loss[i] = 0.5 * (self.push_loss_fn(z1[i], stale_mean=stale_mean, stale_var=stale_var) +
                                      self.push_loss_fn(z2[i], stale_mean=stale_mean, stale_var=stale_var))

            # Update stale estimates
            if self.hparams.stale_estimates:
                mean = 0.5 * (z1[i].mean(dim=0) + z2[i].mean(dim=0)).detach().clone()
                variance = 0.5 * (z1[i].var(dim=0) + z2[i].var(dim=0)).detach().clone()
                if self.first_epoch_flag:
                    self.mean_estimates.append(mean)
                    self.var_estimates.append(variance)
                else:
                    self.mean_estimates[i] = mean
                    self.var_estimates[i] = variance

            # Feature decorrelation loss
            if self.hparams.no_pooling:
                decorr_loss[i] = 0.5 * (self.decorr_loss(_stack_spatial_features(fm1[i])) +
                                        self.decorr_loss(_stack_spatial_features(fm2[i])))
            else:
                decorr_loss[i] = 0.5 * (self.decorr_loss(z1[i]) + self.decorr_loss(z2[i]))

        if self.hparams.stale_estimates:
            self.first_epoch_flag = False
        
        return pull_loss, push_loss, decorr_loss

    def training_step(self, batch, batch_idx):
        pull_loss, push_loss, decorr_loss = self.loss_step(batch)

        for i in range(len(pull_loss) - 1):
            self.log('Layerwise train losses/Layer {} pull loss'.format(i + 1), pull_loss[i], on_epoch=True,
                     on_step=False, logger=True)
            self.log('Layerwise train losses/Layer {} push loss'.format(i + 1), push_loss[i], on_epoch=True,
                     on_step=False, logger=True)
            self.log('Layerwise train losses/Layer {} decorr loss'.format(i + 1), decorr_loss[i], on_epoch=True,
                     on_step=False, logger=True)

        self.log('Layerwise train losses/Final layer pull loss', pull_loss[-1], on_epoch=True, on_step=False,
                 logger=True)
        self.log('Layerwise train losses/Final layer push loss', push_loss[-1], on_epoch=True, on_step=False,
                 logger=True)
        self.log('Layerwise train losses/Final layer decorr loss', decorr_loss[-1], on_epoch=True, on_step=False,
                 logger=True)

        total_loss = self.hparams.pull_coeff * pull_loss.sum() + \
                     self.hparams.push_coeff * push_loss.sum() + \
                     self.hparams.decorr_coeff * decorr_loss.sum()

        self.log('Loss/train_loss', total_loss, on_epoch=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pull_loss, push_loss, decorr_loss = self.loss_step(batch)

        for i in range(len(pull_loss) - 1):
            self.log('Layerwise validation losses/Layer {} pull loss'.format(i + 1), pull_loss[i], on_epoch=True,
                     on_step=False, logger=True)
            self.log('Layerwise validation losses/Layer {} push loss'.format(i + 1), push_loss[i], on_epoch=True,
                     on_step=False, logger=True)
            self.log('Layerwise validation losses/Layer {} decorr loss'.format(i + 1), decorr_loss[i], on_epoch=True,
                     on_step=False, logger=True)

        self.log('Layerwise validation losses/Final layer pull loss', pull_loss[-1], on_epoch=True, on_step=False,
                 logger=True)
        self.log('Layerwise validation losses/Final layer push loss', push_loss[-1], on_epoch=True, on_step=False,
                 logger=True)
        self.log('Layerwise validation losses/Final layer decorr loss', decorr_loss[-1], on_epoch=True, on_step=False,
                 logger=True)

        total_loss = self.hparams.pull_coeff * pull_loss.sum() + \
                     self.hparams.push_coeff * push_loss.sum() + \
                     self.hparams.decorr_coeff * decorr_loss.sum()

        self.log('Loss/val_loss', total_loss, on_epoch=True, logger=True)
        return total_loss

    def set_param_specific_optim_params(self, named_params, skip_list=None):
        if skip_list is None:
            skip_list = ['bias', 'bn', 'norm']
        params = []
        params_no_weight_decay = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                params_no_weight_decay.append(param)
            else:
                params.append(param)
        return [{'params': params},
                {'params': params_no_weight_decay, 'weight_decay': 0.}]

    def configure_optimizers(self):
        parameters = self.set_param_specific_optim_params(self.named_parameters())
        if self.optimizer == 'adam':
            optimizer = Adam(parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = SGD(parameters, lr=self.hparams.learning_rate, momentum=self.hparams.momentum,
                            weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_start_lr=self.hparams.start_lr,
                                                  eta_min=self.hparams.final_lr,
                                                  warmup_epochs=self.hparams.warmup_epochs,
                                                  max_epochs=self.hparams.max_epochs)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Architecture params
        parser.add_argument('--use_projector_mlp', action='store_true')
        parser.add_argument('--projection_size', type=int, default=256)
        parser.add_argument('--mlp_hidden_size', type=int, default=2048)

        # Loss coefficients
        parser.add_argument('--pull_coeff', type=float, default=1.0)
        parser.add_argument('--push_coeff', type=float, default=1.0)
        parser.add_argument('--decorr_coeff', type=float, default=10.0)

        # Experimental params
        parser.add_argument('--no_pooling', action='store_true')
        parser.add_argument('--stale_estimates', action='store_true')
        parser.add_argument('--no_biases', action='store_true')
        parser.add_argument('--shuffle_positives', action='store_true')

        return parser


class SupervisedBaseline(LPL):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        assert not self.hparams.stale_estimates, 'Supervised training with stale estimates makes no difference, pretty sure you didn\'t want to do this'

        self.classifier_heads = nn.ModuleList([])
        for i in range(self.network.encoder.num_trainable_hooks):
            self.classifier_heads.append(torch.nn.Linear(self.network.encoder.projection_sizes[i], self.num_classes))

    def loss_step(self, batch):
        (img_1, img_2, _), label = batch

        _, _, z1 = self.network(img_1)
        _, _, z2 = self.network(img_2)

        loss = 0.
        for i in range(len(z1)):
            pred_1 = self.classifier_heads[i](z1[i])
            pred_2 = self.classifier_heads[i](z2[i])
            loss += 0.5 * (F.cross_entropy(pred_1, label) + F.cross_entropy(pred_2, label))

        return loss

    def training_step(self, batch, batch_idx):
        total_loss = self.loss_step(batch)
        self.log('Loss/train_loss', total_loss, on_epoch=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self.loss_step(batch)
        self.log('Loss/val_loss', total_loss, on_epoch=True, logger=True)
        return total_loss


class NegSampleBaseline(LPL):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        assert not self.hparams.stale_estimates, 'Negative-sample training with stale estimates makes no difference, pretty sure you didn\'t want to do this'
        assert self.hparams.decorr_coeff==0, 'Negative-sample based training should have the decorrelation coefficient at zero'
       
        self.predictive_loss = cosine_pull_loss
        self.push_loss_fn = self.contrastive_push_loss

    def contrastive_push_loss(self, a, stale_var=None, stale_mean=None):
        return cosine_push_loss(a)
