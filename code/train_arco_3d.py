import torch
import logging
import sys
import os
import pickle
import torchvision.models as models
from torchvision.utils import make_grid
import torch.optim as optim
import argparse
import shutil
import random
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import torch.utils.data.sampler as sampler
from tqdm import tqdm
from scipy.ndimage import zoom
from utils import losses, metrics, ramps
from tps.rand_tps_3d import RandTPS

from augment_3d import *
from loss_helper import *
from model_3D import *
from dataloaders.la_heart import LAHeartWithIndex, RandomRotFlip, RandomCrop, ToTensor

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/weicheng/selfLearning/DTC/data/2018LA_Seg_Training Set', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/example_training', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')

parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--strong_threshold_u2pl', default=0.97, type=float)
parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--mona_feature_size', type=int, default=512, help='the feature size of latent vectors')
parser.add_argument('--apply_aug', default='cutmix', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--resume', type=str, default='ACDC/training_pool_latentF512_K36', help='if we should resume from checkpoint') # 'ACDC/training_pool'
parser.add_argument('--K', type=int, default=36, help='the size of cache')                    
parser.add_argument('--k1', type=float, default=0.01, help='the weights for contrastive loss')
parser.add_argument('--k2', type=float, default=1.0, help='the weights for eqv loss')
parser.add_argument('--k3', type=float, default=1.0, help='the weights for unsup loss')
parser.add_argument('--k4', type=float, default=1.0, help='the weights for nn loss')
parser.add_argument('--k5', type=float, default=0.1, help='the weights for nn loss')
parser.add_argument('--topk', type=int, default=5, help='the size of cache')
parser.add_argument('--latent_pooling_size', type=int, default=1, help='the pooling size of latent vector')
parser.add_argument('--latent_feature_size', type=int, default=512, help='the feature size of latent vectors')
parser.add_argument('--output_pooling_size', type=int, default=8, help='the pooling size of output head')
parser.add_argument('--combinations', type=int, default=0, help='0: all, 1: no reco, 2: no unsup')
parser.add_argument('--mask', type=int, default=0, help='0: no mask, 1: use pre_u')
parser.add_argument('--func', type=str, default='asmc', help='asmc or smc')
parser.add_argument('--ref_net', type=str, default='vgg19', help='ref_net')
parser.add_argument('--ref_norm', type=bool, default=False, help='ref_norm')
parser.add_argument('--ref_layer1', type=str, default='relu3_2', help='ref_layer1')
parser.add_argument('--ref_layer2', type=str, default='relu5_4', help='ref_layer2')
parser.add_argument('--ref_weight1', type=float, default=0.33, help='ref_weight1')
parser.add_argument('--ref_weight2', type=float, default=1.0, help='ref_weight2')
parser.add_argument('--temperature', type=float, default=.19, help='temperature')
parser.add_argument('--layer_len', type=int, default=-1, help='layer_len')
parser.add_argument('--tps_sigma', type=float, default=0.01, help='tps_sigma')

args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 23, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "MM" in dataset: # class=8
        ref_dict = {"1": 38, "2": 76, "5": 191, "10": 382}
    elif "Lits" in dataset or "LiTS" in dataset:
        ref_dict = {"1": 167, "5": 835, "10": 1668}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

@torch.no_grad()
def _dequeue_and_enqueue(keys, queue, queue_ptr):
    batch_size = keys.shape[0]

    ptr = int(queue_ptr)
    assert args.K % batch_size == 0 

    # replace the keys at ptr (dequeue and enqueue)
    queue[ptr:ptr + batch_size] = keys
    ptr = (ptr + batch_size) % args.K  # move pointer

    queue_ptr[0] = ptr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_revisiting_loss(random_pool, rep_u, rep_u_teacher, topk=5):
    rep_u = rep_u.view(rep_u.shape[0], -1)
    rep_u = torch.nn.functional.normalize(rep_u, dim=-1)
    rep_u_teacher = rep_u_teacher.view(rep_u_teacher.shape[0], -1)
    rep_u_teacher = torch.nn.functional.normalize(rep_u_teacher, dim=-1)
    dist_t = 2 - 2 * torch.einsum('bc,kc->bk', [rep_u, random_pool])
    dist_q = 2 - 2 * torch.einsum('bc,kc->bk', [rep_u_teacher, random_pool])
    _, nn_index = dist_t.topk(topk, dim=1, largest=False)
    nn_dist_q = torch.gather(dist_q, 1, nn_index)
    loss_q = (nn_dist_q.sum(dim=1) / topk).mean()
    return loss_q


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    record = []

    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(num_classes):
        memobank.append([torch.randn(1, 16)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    random_pool = torch.randn(args.K, 16, *args.patch_size).cuda(0)
    random_pool = random_pool.view(args.K, -1)
    random_pool = nn.functional.normalize(random_pool, dim=1)
    random_pool_ptr = torch.zeros(1, dtype=torch.long)

    db_train_l = LAHeartWithIndex(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(args.patch_size),
                          ToTensor(),
                          ]), index=args.labeled_num, label_type=1)

    db_train_u = LAHeartWithIndex(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(args.patch_size),
                          ToTensor(),
                          ]), index=args.labeled_num, label_type=0)
    

    while(len(db_train_l)<len(db_train_u)):
        db_train_l = torch.utils.data.ConcatDataset([db_train_l, db_train_l])

    train_l_loader = torch.utils.data.DataLoader(
            db_train_l,
            batch_size=batch_size,
            sampler=sampler.RandomSampler(data_source=db_train_l,
                                          replacement=True),
            drop_last=True,
            pin_memory=True
        )

    train_u_loader = torch.utils.data.DataLoader(
            db_train_u,
            batch_size=batch_size,
            sampler=sampler.RandomSampler(data_source=db_train_u,
                                            replacement=True),
            drop_last=True,
            pin_memory=True
        )

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_u_loader)))
      
    isd = ISD_3d(K=args.K, m=0.99, Ts=0.01, Tt=0.1, num_classes=num_classes, latent_pooling_size=args.latent_pooling_size, 
                latent_feature_size=args.latent_feature_size, output_pooling_size=args.output_pooling_size, 
                train_encoder=True, train_decoder=True).cuda() # args.train_encoder # args.train_decoder
    isd.model.load_state_dict(torch.load("../model/{}_{}_labeled{}/{}/iter_30000.pth".format(
        args.resume, args.labeled_num, suffix, args.model), map_location=lambda storage, loc: storage))
    isd.ema_model.load_state_dict(torch.load("../model/{}_{}_labeled{}/{}/iter_30000.pth".format(
        args.resume, args.labeled_num, suffix, args.model), map_location=lambda storage, loc: storage))
    if torch.cuda.device_count() > 1:
        isd.data_parallel()
    ema_model = isd.ema_model
    model = isd.model
    q_representation = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=1, bias=False),
            nn.Conv3d(16, 16, kernel_size=1, bias=False)
        ).cuda()
    
    
    k_feature_extractor = FeatureExtractor_3d(fea_dim=[128, 64, 32, 16, 16], output_dim=16).cuda()
    q_feature_extractor = FeatureExtractor_3d(fea_dim=[128, 64, 32, 16, 16], output_dim=16).cuda()
    
    if torch.cuda.device_count() > 1:
        q_representation = torch.nn.DataParallel(q_representation)
        q_feature_extractor = torch.nn.DataParallel(q_feature_extractor)
        k_feature_extractor = torch.nn.DataParallel(k_feature_extractor)
    
    
    params = [p for p in model.parameters() if p.requires_grad]
    params_rep = [p for p in q_representation.parameters() if p.requires_grad]
    params_fea = [p for p in q_feature_extractor.parameters() if p.requires_grad]
    optimizer = optim.SGD(params+params_rep+params_fea, lr=base_lr, weight_decay=0.0001, momentum=0.9, nesterov=True)
    
    with torch.no_grad():
        for t_params, s_params in zip(k_feature_extractor.parameters(), q_feature_extractor.parameters()):
            t_params.data.copy_(s_params.data)
            t_params.requires_grad = False
            
    tps = RandTPS(args.patch_size[0], args.patch_size[1], args.patch_size[2],
                           batch_size=batch_size*2,
                           sigma=args.tps_sigma,
                           border_padding=False,
                           random_mirror=True,
                           random_scale=(0.8, 1.2),
                           mode='affine').cuda()
    
    model.train()
    ema_model.train()
    q_representation.train()
    k_feature_extractor.train()
    q_feature_extractor.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    iter_num = 0
    max_epoch = max_iterations // len(train_u_loader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        train_l_dataset = iter(train_l_loader)
        train_u_dataset = iter(train_u_loader)

        for i in range(len(train_u_loader)):
            l_next = train_l_dataset.next()
            train_l_data, train_l_label = l_next['image'].cuda(), l_next['label'].cuda()
            u_next = train_u_dataset.next()
            train_u_data, train_u_label = u_next['image'].cuda(), u_next['label'].cuda()

            with torch.no_grad():
                pred_u, _ , _ = ema_model(train_u_data)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u, dim=1), dim=1) # torch.Size([6, 256, 256]) # torch.Size([6, 256, 256])
            _, _, images_cj1_logits_l = \
                batch_transform(train_l_data, train_l_label, logits=torch.ones_like(train_l_label)*255, scale_size=(1.0, 1.0), apply_augmentation=False)
            images_cj2_l, _, _ = \
                batch_transform(train_l_data, train_l_label, logits=torch.ones_like(train_l_label)*255, scale_size=(1.0, 1.0), apply_augmentation=False)
            
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                (train_u_data, pseudo_labels, pseudo_logits)

            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data_3d(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)

            images_cj2_u, _, _ = \
                batch_transform(train_u_aug_data, train_u_aug_label, logits=train_u_aug_logits, scale_size=(1.0, 1.0), apply_augmentation=True)
                
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, logits=train_u_aug_logits, scale_size=(1.0, 1.0), apply_augmentation=True)

            with torch.no_grad():
                for param_q, param_k in zip(q_feature_extractor.parameters(), k_feature_extractor.parameters()):
                    param_k.data = param_k.data * 0.99 + param_q.data * 0.01
            
            pred_l, _, l_feature_map = model(train_l_data)
            pred_u, _, u_feature_map = model(train_u_aug_data)
            
            pred_l_teacher, _, l_feature_map_teacher = ema_model(train_l_data)
            pred_u_teacher, _, u_feature_map_teacher = ema_model(train_u_aug_data)
            
            l_feature_all = q_feature_extractor(l_feature_map)
            u_feature_all = q_feature_extractor(u_feature_map)

            l_feature_all_teacher = k_feature_extractor(l_feature_map_teacher)
            u_feature_all_teacher = k_feature_extractor(u_feature_map_teacher)

            rep_u = q_representation(u_feature_all) 
            rep_l = q_representation(l_feature_all)
            
            rep_u_teacher = (u_feature_all_teacher)
            rep_l_teacher = (l_feature_all_teacher)
            rep_all = torch.cat((rep_l, rep_u))
            pred_all = torch.cat((pred_l, pred_u))

            pred_all_teacher = torch.cat((rep_l_teacher, rep_u_teacher))
            loss_q = get_revisiting_loss(random_pool=random_pool, rep_u=rep_u, rep_u_teacher=rep_u_teacher, topk=args.topk)
            # supervised loss
            outputs_soft = torch.softmax(pred_l, dim=1)
            loss_ce = ce_loss(pred_l, train_l_label.long())
            loss_dice = dice_loss(outputs_soft, train_l_label.unsqueeze(1))
            supervised_loss =  (loss_dice + loss_ce) 
            unsup_loss = compute_unsupervised_loss(pred_u, train_u_aug_label, train_u_aug_logits, args.strong_threshold)
            # print(unsup_loss)
            alpha_t = 20 * (
                1 - epoch_num / max_epoch
            )
            with torch.no_grad():
                label_l = label_onehot(train_l_label, args.num_classes) 
                # train_u_aug_label = 
                label_u = label_onehot(train_u_aug_label, args.num_classes) 

                prob_l = torch.softmax(pred_l, dim=1)
                prob_u = torch.softmax(pred_u, dim=1)

                prob_l_teacher = torch.softmax(pred_l_teacher, dim=1)
                prob_u_teacher = torch.softmax(pred_u_teacher, dim=1)
                prob = prob_u
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
                low_thresh = np.percentile(
                    entropy[train_u_aug_label >= 0].cpu().numpy().flatten(), alpha_t
                )
                low_entropy_mask = (
                    entropy.le(low_thresh).float() * (train_u_aug_label >= 0).bool()
                )
                high_thresh = np.percentile(
                    entropy[train_u_aug_label >= 0].cpu().numpy().flatten(),
                    100 - alpha_t,
                )
                high_entropy_mask = (
                    entropy.ge(high_thresh).float() * (train_u_aug_label >= 0).bool()
                )

                low_mask_all = torch.cat(
                    (
                        (train_l_label.unsqueeze(1) >=0).float(),
                        low_entropy_mask.unsqueeze(1),
                    )
                )

                high_mask_all = torch.cat(
                    (
                        (train_l_label.unsqueeze(1) >=0).float(),
                        high_entropy_mask.unsqueeze(1),
                    )
                )
            
            
            reco_loss = compute_contra_memobank_loss(rep_all, label_l.cuda().long(), label_u.cuda().long(), \
                prob_l_teacher.detach(), prob_u_teacher.detach(), low_mask_all.cuda(), high_mask_all.cuda(), \
                    memobank, queue_ptrlis, queue_size, pred_all_teacher.detach(), delta_n=args.strong_threshold_u2pl, \
                        func=args.func, num_queries=args.num_queries, num_negatives=args.num_negatives
                )[-1]

            
            rep_u_teacher = rep_u_teacher.view(rep_u_teacher.shape[0], -1)
            rep_u_teacher = torch.nn.functional.normalize(rep_u_teacher, dim=-1)
            _dequeue_and_enqueue(keys=rep_u_teacher, queue=random_pool, queue_ptr=random_pool_ptr)
            
            
            # 加一个consistency
            labels = torch.cat((train_l_label, train_u_aug_label), dim=0)
            logits = torch.cat((images_cj1_logits_l, train_u_aug_logits), dim=0)
            mask = torch.ones((rep_all.shape[0], rep_all.shape[2], rep_all.shape[3], rep_all.shape[4]), requires_grad=False).cuda()
            neg = torch.zeros((rep_all.shape[0], rep_all.shape[2], rep_all.shape[3], rep_all.shape[4]), requires_grad=False).cuda()
            mask = torch.where(labels==0, neg, mask)
            mask = torch.where(logits<args.weak_threshold, neg, mask)
            mask = mask.unsqueeze(1)
            images_cj2 = torch.cat((images_cj2_l, images_cj2_u), dim=0)
            tps.reset_control_points()
            images_tps = tps(images_cj2)
            mask_tps = tps(mask.float(), padding_mode='zeros')
            pred_tps = model(images_tps)[0]
            pred_d = pred_all.detach()
            pred_d.requires_grad = False
            pred_tps_org = tps(pred_d.cuda(), padding_mode='zeros')
            kl = nn.KLDivLoss(reduction='none').cuda()
            loss_eqv = kl(F.log_softmax(pred_tps, dim=1),
                           F.softmax(pred_tps_org, dim=1))
            loss_eqv = (loss_eqv * mask_tps).flatten(1).sum(1) / (mask_tps.flatten(1).sum(1) + 1e-7)
            loss_eqv = loss_eqv.mean()
            
            if( iter_num/max_iterations > 0.0):
                loss = args.k1*(reco_loss) + args.k3*unsup_loss + supervised_loss + args.k4*loss_q
            else:
                loss = unsup_loss + supervised_loss + loss_eqv 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            isd._momentum_update_key_encoder()
            # scheduler.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num+=1
            record.append(loss.item())

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/reco_loss', reco_loss, iter_num)
            writer.add_scalar('info/loss_q', loss_q, iter_num)
            writer.add_scalar('info/unsup_loss', unsup_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_eqv', loss_eqv, iter_num)
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, reco_loss: %f, unsup_loss: %f, loss_eqv: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), reco_loss.item(), unsup_loss.item(), loss_eqv.item()))

            if iter_num % 20 == 0:
                image = train_l_data[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                
                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)
                
                image = train_l_label[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num % 1000 == 0:
                if torch.cuda.device_count() > 1:
                    modelToSave = isd.model.module
                else:
                    modelToSave = isd.model
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(modelToSave.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    f = open(snapshot_path+'record.pkl', 'wb')
    pickle.dump(record, f)
    return "Training Finished!"
        

def compute_unsupervised_loss(predict, target, logits, strong_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels

    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None, None] * loss, loss > 0))
    return weighted_loss


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w, im_d = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs).data.cpu().type(torch.int64)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w, im_d]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)



if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    torch.cuda.empty_cache()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    suffix = 'final'
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