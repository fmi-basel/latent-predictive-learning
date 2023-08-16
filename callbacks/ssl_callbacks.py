from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pl_bolts.models.regression import LogisticRegression
from torchmetrics.functional import pairwise_cosine_similarity

import torch
from torch.utils.data import TensorDataset, DataLoader

from metrics.metrics import compute_cosine_measures, compute_manifold_measures, compute_push_metrics


class SSLEvalCallback(Callback):

    def __init__(self, num_epochs, test_dataloader, log_interval_hists=100, z_dim: Optional[int] = None,
                 num_classes: Optional[int] = None):

        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.representations = {'default': [],
                                'first_step': [],
                                'final_step': []}

        self.test_representations = {'default': [],
                                     'first_step': [],
                                     'final_step': []}
        self.labels = []
        self.images = []

        self.log_interval_hists = log_interval_hists
        self.log_hists_at_epochs = [1, 5, 10, 50, 100, 500, 800]

        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs

    @staticmethod
    def get_representations(pl_module, x):
        representations = pl_module(x)
        return representations

    @staticmethod
    def to_device(batch, device):
        x, y = batch
        x = x[-1].to(device)
        y = y.to(device)
        return x, y

    def reinitialize_linear_evaluator(self, pl_module):
        # attach the evaluator to the module
        self.linear_evaluator = LogisticRegression(self.z_dim, self.num_classes, learning_rate=1e-3, checkpoint_callback=False)
        self.linear_evaluator.to(pl_module.device)

    def on_validation_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module, 'z_dim'):
            self.z_dim = pl_module.z_dim
        if hasattr(pl_module, 'num_classes'):
            self.num_classes = pl_module.num_classes

        self.representations = {'default': [],
                                'first_step': [],
                                'final_step': []}
        self.labels = []
        self.reinitialize_linear_evaluator(pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x).detach()
            if len(representations.shape) == 3:
                self.representations['default'].append(representations.mean(dim=1))
                self.representations['first_step'].append(representations[:, 0])
                self.representations['final_step'].append(representations[:, -1])
            else:
                self.representations['default'].append(representations)
            self.labels.append(y)

        if (trainer.current_epoch + 1) in self.log_hists_at_epochs:
            if len(x.shape) == 5:
                x = x[:, 0]
            self.images.append(x)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.labels = torch.cat(self.labels)
        for repr_type in self.representations:
            if len(self.representations[repr_type]) == 0:
                continue
            representations = torch.cat(self.representations[repr_type])

            self.do_linear_classification_eval(trainer, pl_module, representations, repr_type)
            self.do_representation_eval(trainer, pl_module, representations, repr_type)
            if (trainer.current_epoch + 1) in self.log_hists_at_epochs and repr_type == 'default':
                epoch_num = trainer.current_epoch + 1
                if trainer.global_step == 0:
                    epoch_num = 0
                self.log_histograms(trainer, epoch_num, representations)
                # self.log_projections(trainer, epoch_num, representations)

        self.representations = {'default': [],
                                'first_step': [],
                                'final_step': []}
        self.labels = []
        self.images = []

    def do_linear_classification_eval(self, trainer, pl_module, representations, repr_type):
        dataset = TensorDataset(representations, self.labels)
        train_dataloader = DataLoader(dataset, batch_size=512)
        self.reinitialize_linear_evaluator(pl_module)
        readout_trainer = pl.Trainer(gpus=trainer.gpus, max_epochs=self.num_epochs, enable_checkpointing=False, logger=False)
        readout_trainer.fit(self.linear_evaluator, train_dataloader)
        self.linear_evaluator.to(pl_module.device)

        total = 0.
        correct = 0.
        # TODO: test on gpu (need to deal with annoying bug where cuda memory access becomes restricted after
        #  first call to to_device)
        for batch_idx, batch in enumerate(self.test_dataloader, 0):
            x, y = self.to_device(batch, pl_module.device)
            with torch.no_grad():
                test_representation = self.get_representations(pl_module, x).detach()
                if repr_type == 'default':
                    if len(test_representation.shape) == 3:
                        test_representation = test_representation.mean(dim=1)
                if repr_type == 'first_step':
                    test_representation = test_representation[:, 0]
                if repr_type == 'final_step':
                    test_representation = test_representation[:, -1]
                preds = torch.argmax(self.linear_evaluator(test_representation), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_acc = correct / total
        print("Test accuracy {}: {:.2f}%".format(repr_type, test_acc * 100))
        if repr_type == 'default':
            repr_type = ''
        else:
            repr_type = '_' + repr_type
        metrics = {'Performance/linear_readout_acc' + repr_type: test_acc}
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)

    def do_representation_eval(self, trainer, pl_module, representations, repr_type):
        cross_class_mean_cosine, within_class_mean_cosine = compute_cosine_measures(representations, self.labels)
        mean_class_correlation, mean_class_radius, mean_class_dim, global_radius = \
            compute_manifold_measures(representations, self.labels)
        cosine_push, uniformity_push, kl_push, swd_gaussian_push, swd_hypersphere_push, swd_hypercube_push = \
            compute_push_metrics(representations)
        print('Feature alignments: Cross-class : {:.2f}   Within-class: {:.2f}   Ratio: {:.2f}'
              .format(cross_class_mean_cosine, within_class_mean_cosine,
                      torch.abs(within_class_mean_cosine / cross_class_mean_cosine)))
        if repr_type == 'default':
            repr_type = ''
        else:
            repr_type = ' ' + repr_type
        metrics = {'Cosine Metrics{}/within_class_mean_cosine'.format(repr_type): within_class_mean_cosine,
                   'Cosine Metrics{}/cross_class_mean_cosine'.format(repr_type): cross_class_mean_cosine,
                   'Cosine Metrics{}/ratio'.format(repr_type): torch.abs(within_class_mean_cosine /
                                                                         cross_class_mean_cosine),
                   'Manifold Metrics{}/mean_class_correlation'.format(repr_type): mean_class_correlation,
                   'Manifold Metrics{}/mean_class_radius'.format(repr_type): mean_class_radius,
                   'Manifold Metrics{}/global_radius'.format(repr_type): global_radius,
                   'Manifold Metrics{}/mean_class_dim'.format(repr_type): mean_class_dim,
                   'Push Metrics{}/cosine_push'.format(repr_type): cosine_push,
                   'Push Metrics{}/uniformity_push'.format(repr_type): uniformity_push,
                   'Push Metrics{}/kl_push'.format(repr_type): kl_push,
                   'Push Metrics{}/swd_gaussian_push'.format(repr_type): swd_gaussian_push,
                   'Push Metrics{}/swd_hypersphere_push'.format(repr_type): swd_hypersphere_push,
                   'Push Metrics{}/swd_hypercube_push'.format(repr_type): swd_hypercube_push}

        pl_module.logger.log_metrics(metrics, step=trainer.global_step)

    def log_histograms(self, trainer, epoch_num, representations):
        tensorboard = trainer.logger.experiment

        X = representations
        num_units = X.shape[-1]
        num_samples = X.shape[0]
        X_std = (X - X.mean(dim=0, keepdim=True)) / (X.std(dim=0, keepdim=True) + 1e-8)

        mean_sq_activity = (X ** 2).mean(dim=0)

        corr = torch.matmul(X_std.T, X_std) / (num_samples - 1)
        off_diag_idxs = torch.triu_indices(num_units, num_units, offset=1)
        cross_unit_corrs = corr[off_diag_idxs[0], off_diag_idxs[1]]

        eig_vals_corr, _ = torch.linalg.eig(corr)
        eig_vals_corr = eig_vals_corr.real
        eig_vals_corr_norm = eig_vals_corr.clamp(1e-8) / eig_vals_corr.max()

        gram = torch.matmul(X.T, X)
        eig_vals_gram, _ = torch.linalg.eig(gram)
        eig_vals_gram = eig_vals_gram.real
        eig_vals_gram_norm = eig_vals_gram.clamp(1e-8) / eig_vals_gram.max()

        cos_dists = pairwise_cosine_similarity(X - X.mean(dim=0))
        off_diag_idxs = torch.triu_indices(num_samples, num_samples, offset=1)
        cos_dists = cos_dists[off_diag_idxs[0], off_diag_idxs[1]]

        tensorboard.add_histogram('Activity Statistics/unit_activities', mean_sq_activity,
                                  global_step=epoch_num)
        tensorboard.add_histogram('Activity Statistics/unit_correlations', cross_unit_corrs,
                                  global_step=epoch_num)

        tensorboard.add_histogram('Correlation Matrix/eig_vals', eig_vals_corr, global_step=epoch_num)
        tensorboard.add_histogram('Correlation Matrix/normalized_eig_vals', torch.log(eig_vals_corr_norm),
                                  global_step=epoch_num)

        tensorboard.add_histogram('Gramian Matrix/eig_vals', eig_vals_gram, global_step=epoch_num)
        tensorboard.add_histogram('Gramian Matrix/normalized_eig_vals', torch.log(eig_vals_gram_norm),
                                  global_step=epoch_num)

        tensorboard.add_histogram('Embedding Similarity/cos_dists', cos_dists, global_step=epoch_num)
