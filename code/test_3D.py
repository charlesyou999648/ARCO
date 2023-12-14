import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/weicheng/selfLearning/UA-MT-ISD/UA-MT-ISD/data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='LA/Mona_temp_randn_no_consistency_no_uamt_8_labeledfinal/vnet', help='model_name')
parser.add_argument('--epoch', type=int,  default=3000, help='GPU to use')
FLAGS = parser.parse_args()

snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path, nms=0, metric_detail=0)

    return avg_metric


if __name__ == '__main__':
    for i in range(1000, 30100, 1000):
        metric = test_calculate_metric(i)
        print(metric)