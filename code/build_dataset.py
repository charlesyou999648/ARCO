from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import pandas as pd
import numpy as np
import torch
import os
import random
import itertools
import glob

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from torch.utils.data.sampler import Sampler
import h5py

class BaseDataSetsWithIndex(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, index=16, label_type=0):
        self._base_dir = base_dir
        self.index = index
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train' and 'ACDC' in base_dir:
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
            if(label_type==1):
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]
        elif self.split == 'train' and 'MM' in base_dir:
            with open(self._base_dir + '/train_slices.txt', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('.h5\n', '')
                                for item in self.sample_list]
            if(label_type==1):
                self.sample_list = self.sample_list[:index]
            else:
                self.sample_list = self.sample_list[index:]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num-index]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train" and self.transform!=None:
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample

# class TwoStreamBatchSampler(Sampler):
#     """Iterate two sets of indices
#     An 'epoch' is one iteration through the primary indices.
#     During the epoch, the secondary indices are iterated through
#     as many times as needed.
#     """

#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size

#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = self.iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                    grouper(secondary_iter, self.secondary_batch_size))
#         )

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size


#     def iterate_eternally(self,indices):
#         n = len(self.data_source)
#         # def infinite_shuffles():
#         #     while True:
#         #         yield np.random.permutation(indices)
#         # return itertools.chain.from_iterable(infinite_shuffles())
#         for _ in range(self.num_samples // 32):
#             yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=torch.Generator()).tolist()
#         yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=torch.Generator()).tolist()


# def iterate_once(iterable):
#     # return np.random.permutation(iterable) # changes here
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(iterable)
#     return itertools.chain.from_iterable(infinite_shuffles())



# def grouper(iterable, n):
#     "Collect data into fixed-length chunks or blocks"
#     # grouper('ABCDEFG', 3) --> ABC DEF"
#     args = [iter(iterable)] * n
#     return zip(*args)

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        if (split == 'test' or split == 'val'):
            self.sample_list = open(os.path.join(list_dir, self.split+'_vol_40.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split+'_40.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

class Synapse_datasetWithIndex(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, index=221, label_type=1):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        if 'Lits' in list_dir:
            self.sample_list = open(os.path.join(list_dir, self.split+'_40.txt')).readlines()
        
        elif (split == "test" or split == 'val'):
            self.sample_list = open(os.path.join(list_dir, self.split+'_vol.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
            
        self.data_dir = base_dir
        self.index = index
        self.label_type = label_type
        if(label_type==1):
            self.sample_list = self.sample_list[:index]
        else:
            self.sample_list = self.sample_list[index:]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            # print(data_path)
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
