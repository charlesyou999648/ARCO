from torch import equal
from torch.utils.data import DataLoader
# from dataloaders.la_heart import *
import random
from PIL import ImageFilter

from operator import index
import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import torchvision.transforms.functional as transforms_f
from torchvision.transforms import *
from PIL.ImageEnhance import *
from PIL import Image
import copy
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb
from adv_morph import *


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y 
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def image_in_painting(x):
    _, img_rows, img_cols = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y) * 1.0
        cnt -= 1
    return x


def image_out_painting(x):
    _, img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y]


def transform(image, label, logits=None, crop_size=(321, 321), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, InterpolationMode.BILINEAR)
    label = transforms_f.resize(label, resized_size, InterpolationMode.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, InterpolationMode.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    # print(np.shape(image)) # (394, 525, 3)
    image = transforms_f.crop(image, i, j, h, w)
    # print(type(image)) # <class 'PIL.Image.Image'>
    # print(np.shape(image)) # (321, 321, 3)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # pass
        # #Random color jitter
        if torch.rand(1) > 0.5:
            #  color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  For PyTorch 1.9/TorchVision 0.10 users
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            # print(type(color_transform)) # <class 'tuple'>
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        # if torch.rand(1) > 0.5:
        #     image = transforms_f.hflip(image)
        #     label = transforms_f.hflip(label)
        #     if logits is not None:
        #         logits = transforms_f.hflip(logits)
        # elif torch.rand(1) > 0.5:
        #     angle = np.random.randint(-20, 20)/180
        #     image = transforms_f.rotate(image, angle, interpolation =InterpolationMode.BILINEAR)
        #     label = transforms_f.rotate(label, angle, interpolation =InterpolationMode.NEAREST)
        #     if logits is not None:
        #         logits = transforms_f.rotate(logits, angle, interpolation =InterpolationMode.BILINEAR)


    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    # print(torch.unique(label))
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    # image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label


# def denormalise(x, imagenet=True):
#     # if imagenet:
#     #     x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
#     #     x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
#     #     return x
#     # else:
#         return (x + 1) / 2


def tensor_to_pil(im, label, logits):
    # im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits


def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()
    

def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    data_size = data.shape

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).cuda(), torch.cat(label_list).cuda(), torch.cat(logits_list).cuda()
    if torch.rand(1) > 0.5 and apply_augmentation:
        augmentor = AdvMorph(config_dict={'epsilon': 1.5,
                                        'xi': 0.5,
                                        'data_size': data_size,
                                        'vector_size': [data_size[-1]//8, data_size[-1]//8],
                                        'interpolator_mode': 'bilinear'
                                        }, debug=False, use_gpu=True)
        augmentor.init_parameters()
        data_trans = augmentor.forward(data_trans.cuda())
    
    return data_trans, label_trans, logits_trans


def generate_unsup_data(data, target, logits, mode='cutout'):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutout':
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            target[i][(1 - mix_mask).bool()] = -1

            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(target[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        elif mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        elif mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)
        else:
            mix_mask = torch.ones_like(target[i]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long(), new_logits



def random_rot_flip(image, label, logit):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    logit = np.rot90(logit, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    logit = np.flip(logit, axis=axis).copy()
    return image, label, logit


def random_rotate(image, label, logit):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    logit = ndimage.rotate(logit, angle, order=0, reshape=False)
    return image, label, logit



def randomGeneratorWithLogits(image, label, logit, output_size=[256, 256]):
    new_data = []
    new_target = []
    new_logits = []
    _, _, x, y = image.shape
    for i in range(image.shape[0]):
        image_i = image[i, 0, :, :].data.cpu().numpy()
        label_i = label[i, :, :].data.cpu().numpy()
        logit_i = logit[i, :, :].data.cpu().numpy()
        # if random.random() > 0.5:
        #     image_i, label_i, logit_i = random_rot_flip(image_i, label_i, logit_i)
        # elif random.random() > 0.5:
        #     image_i, label_i, logit_i = random_rotate(image_i, label_i, logit_i)
        # print(image_i.shape)
        image_i = zoom(
            image_i, (output_size[0] / x, output_size[1] / y), order=0)
        label_i = zoom(
            label_i, (output_size[0] / x, output_size[1] / y), order=0)
        logit_i = zoom(
            logit_i, (output_size[0] / x, output_size[1] / y), order=0)

        new_data.append(image_i)
        new_target.append(label_i)
        new_logits.append(logit_i)

    image = torch.from_numpy(np.array(new_data).astype(np.float32))
    if len(image.shape) == 3:
        image = image.unsqueeze(1)
    label = torch.from_numpy(np.array(new_target).astype(np.uint8)).long()
    logit = torch.from_numpy(np.array(new_logits))
    return image, label, logit


# if __name__ == '__main__':
#     trainloader = DataLoader(db_train_teacher, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

#     for i_batch, batch in enumerate(trainloader):
#             # print(batch.shape)
#             teacher_batch = transform_teacher(batch)
#             student_batch = transform_student(batch)
#             # print(teacher)
#             teacher_batch, teacher_label = teacher_batch['image'], teacher_batch['label']
#             student_batch, student_label = student_batch['image'], student_batch['label']
#             # print(torch.max(teacher_batch))
#             # teacher_batch = transform_teacher(teacher_batch)
#             # student_batch = transform_student(student_batch)
#             # print(teacher_batch.shape)
#             # print(teacher_label.shape)
#             print(torch.max(teacher_batch))
#             print(torch.max(student_batch))
#             # print(type(teacher_label))
#             exit()
#             # isEqual = teacher_batch.eq(student_batch)
#             # if not (isEqual.all()):
#             #     print('problem at index ', i_batch)
#             #     exit()
# print('no pb.')