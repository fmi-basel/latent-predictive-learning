import os
import h5py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from pytorch_lightning import LightningDataModule

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}

class Shapes3DSequences(Dataset):
    
    def __init__(self, data_dir, images, labels, batch_size=1024, set='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.images = images
        self.labels = labels

        self.image_shape = self.images.shape[1:]
        self.n_samples = self.labels.shape[0]

        shuffled_colors = True
        if shuffled_colors:
            indexed_trajectories = np.load(os.path.join(data_dir, 'indexed_trajectories.npy'))
        else:
            indexed_trajectories = np.load(os.path.join(data_dir, 'indexed_trajectories_no_shuffle.npy'))
        self.batch_size = batch_size

        # split indexed_trajectories into blocks of batch_size + 1 dropping any remainder
        n_blocks = len(indexed_trajectories) // (self.batch_size + 1)
        self.trajectories = indexed_trajectories[:n_blocks * (self.batch_size + 1)].reshape([-1, self.batch_size + 1])
        self.trajectories = torch.from_numpy(self.trajectories)

        if set == 'train':
            self.trajectories = self.trajectories[:-30]
        if set == 'val':
            self.trajectories = self.trajectories[-30:-10]
        if set == 'test':
            self.trajectories = self.trajectories[-10:]
        
        self.num_iter = 0
        self.train = set == 'train'

    def __len__(self):
        return (1000 - 1) if self.train else (len(self.trajectories) - 1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.train:
            # sample a random trajectory
            idx = np.random.randint(len(self.trajectories) - 1)
        trajectory = self.trajectories[idx]
        ims = torch.zeros([self.batch_size + 1] + list(self.image_shape))
        labels = torch.zeros([self.batch_size + 1], dtype=torch.long)
        for i, index in enumerate(trajectory):
            ims[i] = self.images[index]
            labels[i] = self.labels[index]

        if self.transform:
            ims = self.transform(ims)

        return (ims[:-1], ims[1:], ims[:-1]), (labels[:-1])

class Shapes3DDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size=1024, factor_to_use_as_label='shape'):
        super().__init__()
        self.data_path = data_dir
        self.batch_size = batch_size
        self.dims = (3, 64, 64)
        self.output_dims = (6,)

        self.size = (3, 64, 64)

        dataset = h5py.File(os.path.join(data_dir, '3dshapes.h5'), 'r')
        images = np.array(dataset['images'])  # (480000, 64, 64, 3)
        factors = np.array(dataset['labels'])  # (480000, 6)

        # convert from hdf5 to torch tensors
        self.images = torch.from_numpy(images) / 255.0
        self.factors = torch.from_numpy(factors)

        # per-channel zero-mean unit-variance normalization of the images
        self.images = (self.images - self.images.mean(dim=(0, 1, 2))) / self.images.std(dim=(0, 1, 2))

        # change to channel first
        self.images = self.images.permute(0, 3, 1, 2) # shape (batch_size, 3, 64, 64)
        self._create_labels(factor_to_use_as_label)

    def _create_labels(self, factor_to_use_as_label):
        self.factor_to_use_as_label = factor_to_use_as_label
        self.num_classes = _NUM_VALUES_PER_FACTOR[self.factor_to_use_as_label]
        
        factors = self.factors[:, _FACTORS_IN_ORDER.index(self.factor_to_use_as_label)]
        self.labels = torch.zeros([factors.shape[0]], dtype=torch.long)
        if self.factor_to_use_as_label == 'floor_hue' or self.factor_to_use_as_label == 'wall_hue' or self.factor_to_use_as_label == 'object_hue':
            self.labels = (factors * 10).long()
        elif self.factor_to_use_as_label == 'scale':
            # data values do not match the description given for this factor, hence the alternative discretization
            self.labels = np.digitize(factors, np.linspace(0.75, 1.25, 8)) - 1
        elif self.factor_to_use_as_label == 'shape':
            self.labels = (factors).long()
        elif self.factor_to_use_as_label == 'orientation':
            self.labels = np.digitize(factors, np.linspace(-30, 30, 15)) - 1
        
        return
            
    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU
        pass

    def setup(self, stage=None):
        # transforms
        # transform = transforms.Compose([ToTensor()])

        # data
        self.shapes3d_full = Shapes3DSequences(self.data_path, self.images, self.labels, batch_size=self.batch_size, transform=None, set='train')
        self.shapes3d_val = Shapes3DSequences(self.data_path, self.images, self.labels, batch_size=self.batch_size, transform=None, set='val')
        self.shapes3d_test = Shapes3DSequences(self.data_path, self.images, self.labels, batch_size=self.batch_size, transform=None, set='test')

    def train_dataloader(self):
        return DataLoader(self.shapes3d_full, batch_size=None, batch_sampler=None, num_workers=8, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.shapes3d_val, batch_size=None, batch_sampler=None, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.shapes3d_test, batch_size=None, batch_sampler=None, num_workers=8)
