import argparse
import logging
import os
import random
import shutil
import sys
import time
import copy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from dataloaders.dataset_synapse import(Synapse_dataset)

from dataloaders.la_heart import LAHeart, RandomRotFlip, RandomCrop, ToTensor, RandomColorJitter, RandomNoise


from utils import losses, metrics, ramps
from model_3D import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/weicheng/selfLearning/DTC/data/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/stage_2_temp', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# step training
parser.add_argument('--resume', type=str, default='ACDC/training_pool_latentF512_K36', help='if we should resume from checkpoint') # 'ACDC/training_pool'
parser.add_argument('--K', type=int, default=36, help='the size of cache')
parser.add_argument('--train_encoder', type=int, default=1, help='is training encoder?')
parser.add_argument('--train_decoder', type=int, default=1, help='is training decoder?')
parser.add_argument('--k1', type=float, default=1.0, help='the weights for latent contrastive loss')
parser.add_argument('--k2', type=float, default=1.0, help='the weights for output contrastive loss')
parser.add_argument('--latent_pooling_size', type=int, default=1, help='the pooling size of latent vector')
parser.add_argument('--latent_feature_size', type=int, default=512, help='the feature size of latent vectors')
parser.add_argument('--output_pooling_size', type=int, default=8, help='the pooling size of output head')
parser.add_argument('--T_s', type=float, default=0.1, help='temperature for student')
parser.add_argument('--T_t', type=float, default=0.1, help='temperature for teacher')
parser.add_argument('--combinations', type=int, default=1, help='the combination of transformation')

parser.add_argument('--temp_high', default=1.0, type=float)

args = parser.parse_args()
# print(args)


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class KLD(nn.Module):
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)
        targets = F.softmax(targets, dim=1)
        return F.kl_div(inputs, targets, reduction='batchmean')

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_t(epoch_num, max_epoch, T_low=0.1, T_high=1.0):
    temp = (T_high - T_low)\
            *(1 + np.cos(2 * np.pi * epoch_num/max_epoch))/2 + T_low
    return temp


transform_student = transforms.Compose([
                        RandomColorJitter(p=0.5, color =(0.02, 0.02, 0.02, 0.01)),
                        RandomNoise(p=0.5),
                    ])

patch_size = [112, 112, 80]

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    train_data_path = args.root_path

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    ##############################################   original dataset: acdc starts here #########################
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))

    total_slices = len(db_train)
    labeled_slice = args.labeled_num
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model = ISD_3d(K=args.K, m=0.99, Ts=args.T_s, Tt=args.T_t, num_classes=num_classes, latent_pooling_size=args.latent_pooling_size, 
                latent_feature_size=args.latent_feature_size, output_pooling_size=args.output_pooling_size, 
                train_encoder=args.train_encoder, train_decoder=args.train_decoder, patch_size=20).cuda() # args.train_encoder # args.train_decoder
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        # tsc.T = T_s
        for i_batch, sampled_batch in enumerate(trainloader):
            ################################################ add weak strong to stu/tea ############################
            if(args.combinations == 0):
                teacher_batch = sampled_batch
                student_batch = sampled_batch
            elif(args.combinations == 1):
                teacher_batch = sampled_batch
                student_batch = sampled_batch
                student_batch = transform_student(student_batch)
            elif(args.combinations == 2):
                teacher_batch = sampled_batch
                student_batch = sampled_batch
                teacher_batch = transform_student(teacher_batch)
            else:
                teacher_batch = sampled_batch
                student_batch = sampled_batch
                teacher_batch = transform_student(teacher_batch)
                student_batch = transform_student(student_batch)

            teacher_batch, _ = teacher_batch['image'].type(torch.FloatTensor).cuda(), teacher_batch['label'].cuda()
            student_batch, student_label = student_batch['image'].type(torch.FloatTensor).cuda(), student_batch['label'].cuda()
            
            if(len(student_batch.shape)==3):
                student_batch = student_batch.unsqueeze(1)
            if(len(teacher_batch.shape)==3):
                teacher_batch = teacher_batch.unsqueeze(1)

            outputs, ema_output, \
            ema_latent_logits, latent_logits, ema_output_logits, output_logits, \
                 = model(student_batch, teacher_batch)
                
            outputs_soft = torch.softmax(outputs, dim=1)
            loss_ce = ce_loss(outputs[:args.labeled_bs],
                              student_label[:args.labeled_bs][:].long())
            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], student_label[:args.labeled_bs].unsqueeze(1))
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            
            kld = KLD().cuda()
            loss_latent = kld(inputs=latent_logits, targets=ema_latent_logits)

            loss_output = kld(inputs=output_logits, targets=ema_output_logits)

            if args.train_encoder == 1 and not args.train_decoder == 1:
                loss = args.k1* loss_latent 
            else:
                loss = supervised_loss  \
                    + args.k1* loss_latent + args.k2* loss_output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            #################################################  Output session ##########################################
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_latent',
                              loss_latent, iter_num)
            writer.add_scalar('info/loss_output',
                              loss_output, iter_num)              
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_latent: %f, loss_output: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_latent.item(), loss_output.item()))

            if iter_num % 20 == 0:
                image = student_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                
                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)
                
                image = student_label[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)
                
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.model.state_dict(), save_mode_path)
                
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '_ema.pth')
                torch.save(model.ema_model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    torch.cuda.empty_cache()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.train_encoder == 1 and args.train_decoder == 1:
        suffix = 'final'

    elif args.train_encoder == 1:
        suffix = '_train_encoder'
    
    elif args.train_decoder == 1:
        suffix = '_train_decoder'


    snapshot_path = "../model/{}_{}_labeled{}/{}".format(
        args.exp, args.labeled_num, suffix, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)