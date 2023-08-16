# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

# Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# Lightning bolts imports
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.models.self_supervised.simclr import SimCLRTrainDataTransform, SimCLREvalDataTransform
from pl_bolts.callbacks.printing import PrintTableMetricsCallback
from pl_bolts.models.regression import LogisticRegression
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)

# Other imports
import numpy as np
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
import time
import copy
import yaml
from sklearn.model_selection import KFold

# Custom imports
from models import *
from utils import *
from models.module import LPL, SupervisedBaseline, NegSampleBaseline
from metrics.metrics import compute_manifold_measures
from datasets.shapes3d_datamodule import Shapes3DDataModule
_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}

# Set random seed
torch.manual_seed(0)

def compute_participation_ratio(x):
    """
    Calculates participation ratio of the correlation matrix of the input
    pr = (sum of eigenvalues)^2 / sum of (eigenvalues^2)
    :param x: (torch.tensor) input feature tensor (batch x features)
    :return: (torch.tensor)
             participation ratio (singleton tensor)
    """
    x_c = x - x.mean(dim=0)
    corr_matrix = torch.einsum('ij,kj->ik', x_c, x_c)
    eig_vals, _ = torch.linalg.eig(corr_matrix)
    eig_vals = eig_vals.real
    pr = eig_vals.sum() ** 2 / (eig_vals ** 2).sum()
    return pr

def get_pixels(dataloader):
    
    pixels = []
    labels = []
    pooler = nn.AdaptiveAvgPool2d((1, 1))
    for inputs, label in dataloader:
        x = inputs[-1].to('cuda')
        labels.append(label.to('cuda'))
        pixels.append(x.view(x.shape[0],-1))
        
    labels = torch.cat(labels)
    pixels = torch.cat(pixels)
    return pixels, labels

def get_representations(dataloader, encoder_blocks, last_layer_only=False, compute_pr=False):
    if last_layer_only:
        representations = [[]]
        participation_ratios = [[]]
        mean_acts = [[]]
    else:
        representations = [[] for i in range(len(encoder_blocks))]
        participation_ratios = [[] for i in range(len(encoder_blocks))]
        mean_acts = [[] for i in range(len(encoder_blocks))]

    labels = []
    pooler = nn.AdaptiveAvgPool2d((1, 1))
    for inputs, label in dataloader:
        x = inputs[-1].to('cuda')
        labels.append(label.to('cuda'))
        with torch.no_grad():
            z = x
            for i, block in enumerate(encoder_blocks):
                z = block(z)
                if isinstance(z, tuple):
                    z = z[0]
                z_pooled = pooler(z)
                # z_pooled = z.view(z.shape[0], -1)
                if not last_layer_only: 
                    representations[i].append(z_pooled.squeeze())
                    if compute_pr:
                        pr = compute_participation_ratio(z_pooled.squeeze())
                    else:
                        pr = torch.tensor([0.])
                    participation_ratios[i].append(pr)
        if last_layer_only:
            representations[0].append(z_pooled.squeeze())
            if compute_pr:
                pr = compute_participation_ratio(z_pooled.squeeze())
            else:
                pr = torch.tensor([0.])
            participation_ratios[0].append(pr)
    labels = torch.cat(labels)
    for i, layer_repr in enumerate(representations):
        representations[i] = torch.cat(layer_repr)
        mean_acts[i] = torch.mean(representations[i]).item()
        participation_ratios[i] = torch.mean(torch.tensor(participation_ratios[i])).item()

        # # zero-mean unit-variance normalization for each layer representation
        # representations[i] = (representations[i] - representations[i].mean(dim=0)) / representations[i].std(dim=0)

    return representations, labels, participation_ratios, mean_acts


def run_eval(data_module, model_type, path, last_layer_only=False, encoder='vgg', stl_dataset=False, cross_val=False):
    
    if model_type == 'lpl':
        model = LPL.load_from_checkpoint(path, strict=False).to('cuda')
    elif model_type == 'supervised':
        model = SupervisedBaseline.load_from_checkpoint(path, strict=False).to('cuda')
    elif model_type == 'neg-samples':
        model = NegSampleBaseline.load_from_checkpoint(path, strict=False).to('cuda')
    
    resnet = encoder == 'resnet'
    if resnet:
        encoder = model.network.encoder
        encoder_blocks = [nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool), encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4]
    else:
        encoder_blocks = model.network.encoder.blocks

    data_loader = data_module.val_dataloader()
    if stl_dataset:
        data_loader = data_module.train_dataloader_labeled()
    val_reprs, val_labels, _, _ = get_representations(data_loader, encoder_blocks, last_layer_only=last_layer_only)
    test_reprs, test_labels, participation_ratios, mean_acts = get_representations(data_module.test_dataloader(), encoder_blocks, last_layer_only=last_layer_only, compute_pr=True)
    
    test_readout_acc = []
    top_5_acc = []
    for i, layer_repr in enumerate(val_reprs):
        
        train_dataset = TensorDataset(layer_repr, val_labels)
        test_dataset = TensorDataset(test_reprs[i], test_labels)

        repr_dim = layer_repr.shape[1]

        if not cross_val:
            train_dataloader = DataLoader(train_dataset, batch_size=512)
            test_dataloader = DataLoader(test_dataset, batch_size=512)
            acc, top_5 = return_lep_acc(train_dataloader, test_dataloader, repr_dim, num_classes=data_module.num_classes)
            test_readout_acc.append([acc])
            top_5_acc.append([top_5])
        else:
            num_folds = 10
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
            accs = []
            top_5_accs = []
            for train_index, test_index in kf.split(layer_repr):
                train_dataloader = DataLoader(TensorDataset(layer_repr[train_index], val_labels[train_index]), batch_size=512)
                test_dataloader = DataLoader(TensorDataset(layer_repr[test_index], val_labels[test_index]), batch_size=512)
                acc, top_5 = return_lep_acc(train_dataloader, test_dataloader, repr_dim, num_classes=data_module.num_classes)
                accs.append(acc)
                top_5_accs.append(top_5)
            test_readout_acc.append(accs)
            top_5_acc.append(top_5_accs)

        # print average acc
        if not cross_val:
            print('Layer {} test acc: {:.3f}'.format(i, test_readout_acc[-1][0]))
            print('Layer {} top 5 acc: {:.3f}'.format(i, top_5_acc[-1][0]))
        else:
            print('Layer {} test acc: {:.3f} +- {:.3f}'.format(i, np.mean(test_readout_acc[-1]), np.std(test_readout_acc[-1])))
            print('Layer {} top 5 acc: {:.3f} +- {:.3f}'.format(i, np.mean(top_5_acc[-1]), np.std(top_5_acc[-1])))

        print(f"Layer {i} mean activations: {mean_acts[i]}")
        print(f"Layer {i} participation ratio: {participation_ratios[i]}")

    return test_readout_acc, top_5_acc, participation_ratios, mean_acts

def return_lep_acc(train_dataloader, test_dataloader, repr_dim, num_classes=10):
    linear_evaluator = LogisticRegression(repr_dim, num_classes, learning_rate=1e-3, checkpoint_callback=False).to('cuda')
    readout_trainer = pl.Trainer(gpus=1, max_epochs=20, checkpoint_callback=False)
    readout_trainer.fit(linear_evaluator, train_dataloader)
    accuracy = 0.
    top_5 = 0.
    count = 0.
    linear_evaluator.to('cuda')
    for x in test_dataloader:
        pred = linear_evaluator(x[0].to('cuda'))
        accuracy += (pred.argmax(dim=-1)==x[1]).sum().item()
        if pred.shape[1] > 5:
            # evaluate top-5 accuracy as well
            top_5 += (pred.topk(5, dim=-1)[1]==x[1].unsqueeze(1)).sum().item()
        else:
            top_5 += (pred.argmax(dim=-1)==x[1]).sum().item()
        count += len(x[1])
    return accuracy/count, top_5/count


def evaluate_model(data_module, model_type, prefix, suffix, encoder='vgg', last_layer_only=True, stl_dataset=False, cross_val=False):

    path = os.path.join(prefix, suffix, 'checkpoints')
    for f in os.listdir(path):
        if '.ckpt' in f:
            model_filename = f
            break
    path = os.path.join(path, model_filename)
    return run_eval(data_module, model_type, path, last_layer_only=last_layer_only, encoder=encoder, stl_dataset=stl_dataset, cross_val=cross_val)

def evaluate_pixel_acc(data_module, stl_dataset=False, cross_val=False):
    
    data_loader = data_module.val_dataloader()
    if stl_dataset:
        data_loader = data_module.train_dataloader_labeled()
    test_data_loader = data_module.test_dataloader()

    val_pixels, val_labels = get_pixels(data_loader)
    test_pixels, test_labels = get_pixels(test_data_loader)

    num_pixels = val_pixels.shape[-1]
    
    accs = []
    top_5_accs = []
    if not cross_val:
        train_dataloader = DataLoader(TensorDataset(val_pixels, val_labels), batch_size=512)
        test_dataloader = DataLoader(TensorDataset(test_pixels, test_labels), batch_size=512)
        acc, top_5 = return_lep_acc(train_dataloader, test_dataloader, num_pixels, num_classes=data_module.num_classes)
        accs.append(acc)
        top_5_accs.append(top_5)
    else:
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        accs = []
        top_5_accs = []
        for train_index, test_index in kf.split(val_pixels):
            train_dataloader = DataLoader(TensorDataset(val_pixels[train_index], val_labels[train_index]), batch_size=512)
            test_dataloader = DataLoader(TensorDataset(val_pixels[test_index], val_labels[test_index]), batch_size=512)
            acc, top_5 = return_lep_acc(train_dataloader, test_dataloader, num_pixels, num_classes=data_module.num_classes)
            accs.append(acc)
            top_5_accs.append(top_5)
    
    return accs, top_5_accs


def cli_main():

    # Eval options
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet2012', 'stl10', 'shapes3d'], default='cifar10')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--downsample_images', action='store_true')
    parser.add_argument('--experiment_dir', type=str)
    parser.add_argument('--model_type', type=str, default='lpl')
    parser.add_argument('--encoder', type=str, choices=['vgg', 'resnet', 'alexnet'], default='vgg')
    parser.add_argument('--last_layer_only', action='store_true')
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=16, type=int, help="num of workers per GPU")
    parser.add_argument("--do_cross_val", action='store_true')
    parser.add_argument("--evaluate_pixel_acc", action='store_true')
    args = parser.parse_args()

    # Set up log directories (for experimental sanity) and tensorboard logger
    dataset = args.dataset
    # Create datamodule
    data_dir = os.path.join(os.path.expanduser("~/data/datasets"), args.dataset)
    data_module = None
    h = 0  # image size
    if args.dataset == 'cifar10':
        data_module = CIFAR10DataModule.from_argparse_args(args, data_dir=data_dir)
        normalization = cifar10_normalization()
        (c, h, w) = data_module.size()
    elif args.dataset == 'stl10':
        data_module = STL10DataModule.from_argparse_args(args, data_dir=data_dir)
        normalization = stl10_normalization()
        data_module.val_dataloader = data_module.train_dataloader_labeled
        (c, h, w) = data_module.size()
        if args.downsample_images:
            h = 32
    elif args.dataset == 'imagenet2012':
        data_module = ImagenetDataModule.from_argparse_args(args, data_dir=data_dir, image_size=196)
        normalization = imagenet_normalization()
        (c, h, w) = data_module.size()
    elif args.dataset == 'shapes3d':
        data_module = Shapes3DDataModule.from_argparse_args(args, data_dir=data_dir)
        (c, h, w) = (3, 64, 64)

    if args.dataset != 'shapes3d':
        data_module.train_transforms = SimCLREvalDataTransform(h, normalize=normalization)
        data_module.val_transforms = SimCLREvalDataTransform(h, normalize=normalization)
        data_module.test_transforms = SimCLREvalDataTransform(h, normalize=normalization)
    if args.downsample_images:
        data_module.train_transforms = transforms.Compose([transforms.Resize(32),
                                                           data_module.train_transforms])
        data_module.val_transforms = transforms.Compose([transforms.Resize(32),
                                                         data_module.val_transforms])
        data_module.test_transforms = transforms.Compose([transforms.Resize(32),
                                                          data_module.test_transforms])

    args.num_classes = data_module.num_classes
    data_module.prepare_data()
    data_module.setup()

    # read results_dict from yaml file if it exists
    results_file_name = ['results.yaml'] if args.dataset != 'shapes3d' else ['results_{}.yaml'.format(factor) for factor in _FACTORS_IN_ORDER]
    for i, results_file in enumerate(results_file_name):
        print('Evaluating results to {}'.format(results_file))
        results_path = os.path.join(args.experiment_dir, results_file)
        results_dict = {}
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        for f in os.listdir(args.experiment_dir):
            # check if this is a directory
            if os.path.isdir(os.path.join(args.experiment_dir, f)) and f not in results_dict:
                results_dict[f] = {}
                if os.path.isdir(os.path.join(args.experiment_dir, f)):
                    if args.dataset == 'shapes3d':
                        data_module._create_labels(_FACTORS_IN_ORDER[i])
                        args.num_classes = data_module.num_classes
                    acc, top_5, pr, mean_acts = evaluate_model(data_module, args.model_type, args.experiment_dir, f, encoder=args.encoder, last_layer_only=args.last_layer_only, stl_dataset=args.dataset=='stl10', cross_val=args.do_cross_val)
                    results_dict[f] = {'acc': acc, 'top_5': top_5, 'pr': pr, 'mean_acts': mean_acts}
                    with open(results_path, 'w') as f:
                        yaml.dump(results_dict, f)

        if args.evaluate_pixel_acc and 'pixels' not in results_dict:
            if args.dataset == 'shapes3d':
                data_module._create_labels(_FACTORS_IN_ORDER[i])
                args.num_classes = data_module.num_classes
            pixel_acc, pixel_top_5 = evaluate_pixel_acc(data_module, stl_dataset=args.dataset=='stl10', cross_val=args.do_cross_val)
            results_dict['pixels'] = {'acc': pixel_acc, 'top_5': pixel_top_5}
            with open(results_path, 'w') as f:
                yaml.dump(results_dict, f)

if __name__ == '__main__':
    cli_main()
