import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from networks.net_factory_args import net_factory
from tqdm import tqdm
# from dataloaders.dataset_synapse import Synapse_dataset
# from utils import test_single_volume
import SimpleITK as sitk
from scipy.ndimage import zoom
# from model_ISD_modified_withRECO import *
from model_ISD_modified_old_abla import *
from medpy import metric
import h5py
from torch.utils.data import Dataset

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        if 'Lits' in list_dir:
            self.sample_list = open(os.path.join(list_dir, self.split+'_vol_40.txt')).readlines()
        else:
            self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
            
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if "Syn" in self.data_dir or "syn" in self.data_dir:
            if self.split == "train":
                slice_name = self.sample_list[idx].strip('\n')
                data_path = os.path.join(self.data_dir, slice_name+'.npz')
                data = np.load(data_path)
                image, label = data['image'], data['label']
            else:
                vol_name = self.sample_list[idx].strip('\n')
                filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
                data = h5py.File(filepath)
                image, label = data['image'][:], data['label'][:]
        
        else:
            if self.split == "train":
                slice_name = self.sample_list[idx].strip('\n')
                data_path = os.path.join(self.data_dir, slice_name+'.npz')
                data = np.load(data_path)
                image, label = data['image'], data['label']
            else:
                vol_name = self.sample_list[idx].strip('\n')
                filepath = self.data_dir + "/{}.h5".format(vol_name)
                data = h5py.File(filepath)
                image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/data/Lits/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--exp', type=str,
                    default='Lits/example_training_final_RECO_2phase', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--list_dir', type=str,
                    default='/data/data/Lits', help='list dir')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--epoch', type=int,
                    default=6000, help='testing epoch')
parser.add_argument('--K', type=int, default=36, help='the size of cache')
parser.add_argument('--latent_pooling_size', type=int, default=1, help='the pooling size of latent vector')
parser.add_argument('--latent_feature_size', type=int, default=512, help='the feature size of latent vectors')
parser.add_argument('--output_pooling_size', type=int, default=8, help='the pooling size of output head')


args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = Synapse_dataset(base_dir=args.root_path, split="test", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    # model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[256, 256],
                                      test_save_path=test_save_path, case=case_name, z_spacing=10)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f jc %f mean_hd95 %f asd %f ' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2], np.mean(metric_i, axis=0)[3]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f jc %f mean_hd95 %f asd %f ' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2], metric_list[i-1][3]))
    performance = np.mean(metric_list, axis=0)[0]
    jc = np.mean(metric_list, axis=0)[1]
    mean_hd95 =np.mean(metric_list, axis=0)[2]
    asd = np.mean(metric_list, axis=0)[3]
    logging.info('Testing performance in best val model: mean_dice : %f jc %f mean_hd95 : %f asd %f' % (performance, jc, mean_hd95, asd))
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # net.eval()
            with torch.no_grad():
                outputs = net(input)[0]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        # net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:

        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, jc, hd95, asd
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 1, 0, 0
    else: 
        return 0, 0, 0, 0

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    if 'RECO' in FLAGS.exp:
        test_save_path = "../model/{}_{}_labeled_train_RECO/{}_predictions/".format(#
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
        snapshot_path = "../model/{}_{}_labeled_train_RECO/{}".format(#
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    # elif 'final' in FLAGS.exp:
    else:
        test_save_path = "../model/{}_{}_labeledfinal/{}_predictions/".format(#
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
        snapshot_path = "../model/{}_{}_labeledfinal/{}".format(#
            FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    # else:
    #     test_save_path = "../model/{}_{}_labeled_train_decoder/{}_predictions/".format(#
    #         FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    #     snapshot_path = "../model/{}_{}_labeled_train_decoder/{}".format(#
    #         FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    # net = ISD(K=FLAGS.K, m=0.99, Ts=0.1, Tt=0.1, num_classes=FLAGS.num_classes, 
    #             latent_pooling_size=FLAGS.latent_pooling_size, latent_feature_size=FLAGS.latent_feature_size, output_pooling_size=FLAGS.output_pooling_size, 
    #             train_encoder=False, train_decoder=False).cuda()
    net = create_model(ema=False, num_classes=FLAGS.num_classes, train_encoder=False, train_decoder=False)
    save_mode_path = os.path.join(
        snapshot_path, 'iter_{}.pth'.format(FLAGS.epoch))
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    metric = inference(FLAGS, model=net, test_save_path=test_save_path)
    print(metric)
    # print((metric[0]+metric[1])/2)
    cur = None
    for i in metric:
        try:
            if cur == None:
                cur = i
            else:
                cur += i
        except:
            if cur.all() == None:
                cur = i
            else:
                cur += i
    print(cur/len(metric))
    # print((metric[0]+metric[1]+metric[2]+metric[3]+metric[4]+metric[5]+metric[6]+metric[7])/8)
