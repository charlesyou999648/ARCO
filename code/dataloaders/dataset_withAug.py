import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from torchvision.transforms import *
from PIL.ImageEnhance import *
from PIL import Image
from torch.utils.data import DataLoader
from scipy.ndimage.filters import gaussian_filter
from PIL import ImageFilter
# from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform


class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}.h5".format(image_name), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
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


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        elif random.random() > 0.5:
            image, label = random_crop(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size
        # self.index=index
    def __call__(self, sample):
        # if self.index==0:
        image, label = sample['image'], sample['label']
        # else:
        #     image, label = sample[1]['image'], sample[1]['label']
        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph),], mode='constant', constant_values=0)

        (w, h) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        # if self.index==0:
            # return [{'image': image, 'label': label}, sample[1]]
        # else:
            # return [sample[0], {'image': image, 'label': label}]
        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
        # self.index = index
    def __call__(self, sample):
        # if self.index==0:
        #     image, label = sample[0]['image'], sample[0]['label']
        # else:
        #     image, label = sample[1]['image'], sample[1]['label']

        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)

        (w, h) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

        # if self.index==0:
        #     return [{'image': image, 'label': label}, sample[1]]
        # else:
        #     return [sample[0], {'image': image, 'label': label}]
        return {'image': image, 'label': label}


class RandomCropBatch(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size
        # self.index = index
    def __call__(self, sample):

        image, label = sample['image'], sample['label']
        new_image = []
        new_label = []
        # print(image.shape)
        for i in range(image.shape[0]):
            cur_image = image[i]
            cur_label = label[i]
        # pad the sample if necessary
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                cur_image = np.pad(cur_image, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
                cur_label = np.pad(cur_label, [(pw, pw), (ph, ph)], mode='constant', constant_values=0)
            # print(image[0].shape) # (180, 150, 88)
            # print(self.output_size[0]) # 112
            # exit()
            (w, h) = image[0].shape
            # print(w)
            # if np.random.uniform() > 0.33:
            #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
            #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
            # else:
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])

            cur_label = cur_label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]
            cur_image = cur_image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1]]

            new_image.append(cur_image)
            new_label.append(cur_label)

        # if self.index==0:
        #     return [{'image': image, 'label': label}, sample[1]]
        # else:
        #     return [sample[0], {'image': image, 'label': label}]
        new_image = torch.FloatTensor(np.array(new_image))
        new_label = torch.FloatTensor(np.array(new_label))
        return {'image': new_image, 'label': new_label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    # def __init__(self, index=0) -> None:
    #     self.index=index

    def __call__(self, sample):
        # if self.index==0:
        #     # print(np.shape(sample))
        #     image, label = sample[0]['image'], sample[0]['label']
        # else:
        #     image, label = sample[1]['image'], sample[1]['label']
        image, label = sample['image'], sample['label']
        # print(image.shape) # torch.Size([4, 180, 150, 88])
        # print(image.shape)
        for i in range(image.shape[0]):
            cur_img = image[i]
            cur_label = label[i]
            k = np.random.randint(0, 4)
            cur_img = np.rot90(cur_img, k)
            cur_label = np.rot90(cur_label, k)
            axis = np.random.randint(0, 2)
            cur_img = np.flip(cur_img, axis=axis).copy()
            cur_label = np.flip(cur_label, axis=axis).copy()

            image[i] = torch.FloatTensor(cur_img)
            label[i] = torch.FloatTensor(cur_label)
        # if self.index==0:
        #     return [{'image': image, 'label': label}, sample[1]]
        # else:
        #     return [sample[0], {'image': image, 'label': label}]
        # print('after rotation',image.shape)
        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else: 
            image, label = sample['image'], sample['label']
            new_image = []
            # noise = np.clip(self.sigma * np.random.randn(*image.shape), -2*self.sigma, 2*self.sigma)
            sigma = random.uniform(0.15, 1.15)
            for i in range(image.shape[0]):
                image_i = ToPILImage()(image[i, 0, :, :]).filter(ImageFilter.GaussianBlur(radius=sigma))
                new_image.append(np.array(image_i)/255)
            # image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

            image = torch.tensor(np.array(new_image), dtype=torch.float64)
            return {'image': image, 'label': label}

# class RandomApply(object):
#     def __init__(self, p=0.1, trans=None):
#         self.p = p
#         self.trans=trans
#     def __call__(self, sample):
#         if self.trans==None or np.random.uniform(low=0, high=1, size=1) > self.p:
#             return sample
#         else: 
#             return self.trans(sample)

class RandomColorJitter(object):
    def __init__(self, color = (0.4, 0.4, 0.4, 0.1), p=0.1) -> None:
        self.color = color
        self.p = p
    
    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            for j in range(image.shape[0]):
                image[j, :, :, :] = ColorJitter(
                    brightness=self.color[0], 
                    contrast=self.color[1], 
                    saturation=self.color[2], 
                    hue=self.color[3])((image[j, :, :, :]))
                    

            return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        # image = sample
        # img_another = sample[1]['image']
        image = image.reshape(1, *image.shape).astype(np.float64)
        # img_another = img_another.reshape(1, img_another.shape[0], img_another.shape[1], img_another.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample[0]['onehot_label']).long()}
        # return torch.from_numpy(image)
               
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class Resize(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        pose = transforms.Compose([
            transforms.Resize((1, 256, 256))])
        image = pose(image)
        return {'image': image, 'label': label}


class BrightnessTransform(object):
    def __init__(self, p=0.5, mu=0.8, sigma=0.1) -> None:
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(low=0, high=1, size=1) > self.p:
            return sample
        else:
            image, label = sample['image'], sample['label']
            for j in range(image.shape[0]):
                image[j, :, :, :] = torch.clamp(self.mu*image[j, :, :, :] + self.sigma, min=0.0, max=1.0)
            return {'image': image, 'label': label}


# class GammaTransform(object):
#     def __init__(self, p=0.5, gamma = 0.5) -> None:
#         self.gamma = gamma
#         self.p = p
#     def __call__(self, sample):
#         if np.random.uniform(low=0, high=1, size=1) > self.p:
#             return sample
#         else:
#             image, label = sample['image'], sample['label']
#             for j in range(image.shape[0]):
#                 image[j, :, :, :] = torch.clamp(torch.pow(image[j, :, :, :], self.gamma), min=0.0, max=1.0)
#             return {'image': image, 'label': label}


# class GammaTransform(object):
#     def __init__(self, p=0.5, gamma = 0.5) -> None:
#         self.gamma = gamma
#         self.p = p
#     def __call__(self, sample):
#         if np.random.uniform(low=0, high=1, size=1) > self.p:
#             return sample
#         else:
#             image, label = sample['image'], sample['label']
#             for j in range(image.shape[0]):
#                 image[j, :, :, :] = torch.clamp(torch.pow(image[j, :, :, :], self.gamma), min=0.0, max=1.0)
#             return {'image': image, 'label': label}

# class GaussianNoise(object):
#     def __init__(self, p=0.5, sigma = 0.5) -> None:
#         self.sigma = sigma
#         self.p = p
#     def __call__(self, sample):
#         if np.random.uniform(low=0, high=1, size=1) > self.p:
#             return sample
#         else:
#             image, label = sample['image'], sample['label']
#             for j in range(image.shape[0]):
#                 image[j, :, :, :] = image[j, :, :, :] + (self.sigma**0.5) * torch.randn(image[j, :, :, :].shape)
#             return {'image': image, 'label': label}
# 
# 

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable) # changes here
    # def infinite_shuffles():
    #     while True:
    #         yield np.random.permutation(iterable)
    # return itertools.chain.from_iterable(infinite_shuffles())


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def worker_init_fn(worker_id):
    random.seed(100+worker_id)


if __name__ == '__main__':
    
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    root_path = '../data/ACDC'
    patch_size_larger = (256, 256)
    patch_size = (224, 224)
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_size = 4
    labeled_bs = 2
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)


    transform_student = transforms.Compose([
                            RandomCropBatch(patch_size),
                            RandomRotFlip(),
                            RandomColorJitter(p=0.8, color = (0.2, 0.2, 0.2, 0.1)),
                            RandomNoise(mu=0.1, sigma=2.0),
                        ])

    transform_teacher = transforms.Compose([
                            RandomCropBatch(patch_size),
                            RandomRotFlip(),
                        ])
    db_train = BaseDataSets(base_dir=root_path, split="train", num=80, transform=transforms.Compose([
                        RandomCrop(patch_size_larger),
                        ]))
    # print(db_train)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    for i_batch, batch in enumerate(trainloader):
        # print(batch['image'].shape)
        teacher_batch = transform_teacher(batch)
        student_batch = transform_student(batch)
        teacher_batch, teacher_label = teacher_batch['image'], teacher_batch['label']
        student_batch, student_label = student_batch['image'], student_batch['label']

        student_batch = student_batch.unsqueeze(1)
        teacher_batch = teacher_batch.unsqueeze(1)
        print(torch.max(student_label[1]))
        # print(student_batch.shape)
        # print(student_label.shape)

        # print(torch.mean(teacher_batch))
        # print(torch.mean(student_batch))
        # isEqual = teacher_label.eq(student_label)
        # if not (isEqual.all()):
        #     print('problem at index ', i_batch)
        #     exit()

# print('no problem')