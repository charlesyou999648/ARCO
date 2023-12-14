import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

def random_crop(image, label):
    output_size = [256, 256]
    if label.shape[0] <= output_size[0] or label.shape[1] <= output_size[1]:
        pw = max((output_size[0] - label.shape[0]) // 2 + 3, 0)
        ph = max((output_size[1] - label.shape[1]) // 2 + 3, 0)
        image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
        label = np.pad(label, [(pw, pw), (ph, ph),], mode='constant', constant_values=0)

    (w, h) = image.shape

    w1 = int(round((w - output_size[0]) / 2.))
    h1 = int(round((h - output_size[1]) / 2.))

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    return image, label


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        elif random.random() > 0.5:
            image, label = random_crop(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, num=None, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        if 'Lits' in list_dir:
            self.sample_list = open(os.path.join(list_dir, self.split+'_40.txt')).readlines()
        elif (split == "test" or split == 'val'):
            self.sample_list = open(os.path.join(list_dir, self.split+'_vol.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

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
            filepath = self.data_dir + "_40/{}.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

if __name__ == '__main__':
    dataset = Synapse_dataset(base_dir='/data/data/Synapse/data/Synapse/train_npz', \
        list_dir='/data/data/Synapse/data/lists_Synapse', split='train')
    print(len(dataset)) # 2211