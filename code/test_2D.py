import argparse
import os
import shutil
import pickle
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from model_2D import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/actionV1_no_aug', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--K', type=int, default=36, help='the size of cache')
parser.add_argument('--latent_pooling_size', type=int, default=1, help='the pooling size of latent vector')
parser.add_argument('--latent_feature_size', type=int, default=512, help='the feature size of latent vectors')
parser.add_argument('--output_pooling_size', type=int, default=8, help='the pooling size of output head')
parser.add_argument('--epoch', type=int,
                    default=30000, help='testing epoch')

parser.add_argument('--startEpoch', type=int,
                    default=30000, help='testing epoch')

parser.add_argument('--endEpoch', type=int,
                    default=30000, help='testing epoch')

parser.add_argument('--fold', type=int,
                    default=5, help='cross validation')
parser.add_argument('--cross_val', type=int,
                    default=1, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
parser.add_argument('--labeled_ratio', type=int, default=20,
                    help='1/labeled_ratio data is provided mask')
parser.add_argument('--seed', type=int,  default=1, help='random seed')

FLAGS = parser.parse_args()

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



def test_single_volume(case, net, classes, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)[0] # , torch.zeros_like(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return metric_list


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    test_save_path = "../model/{}_{}_labeledfinal/{}_predictions/".format(#
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    snapshot_path = "../model/{}_{}_labeledfinal/{}".format(#
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = create_model(ema=False, num_classes=FLAGS.num_classes, train_encoder=False, train_decoder=False)
    
    save_mode_path = os.path.join(
        snapshot_path, 'iter_{}.pth'.format(FLAGS.epoch))
    net.load_state_dict(torch.load(save_mode_path, map_location=lambda storage, loc: storage))

    print("init weight from {}".format(save_mode_path))
    net.eval()
    metric_list = 0.0
    for case in tqdm(image_list):
        metric_i = test_single_volume(
            case, net, FLAGS.num_classes, test_save_path, FLAGS)
        metric_list += np.array(metric_i)
    avg_metric = metric_list / len(image_list)
    return avg_metric


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'
    startEpoch = FLAGS.startEpoch
    endEpoch = FLAGS.endEpoch
    
    record = []
    for i in range(startEpoch, endEpoch+1, 1000):
        FLAGS.epoch = i
        metric_ = Inference(FLAGS)
        print(metric_)
        cur = None
        for i in metric_:
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
        print(cur/len(metric_))
        record.append(cur/len(metric_))
    f = open("../model/{}_{}_labeledfinal/{}".format(#
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)+'test.pkl', 'wb')
    pickle.dump(record, f)
