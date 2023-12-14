from unittest.mock import patch
import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import math
# from .utils import dequeue_and_enqueue


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    keys = keys.detach().clone().cpu()
    # gathered_list = gather_together(keys)
    # keys = torch.cat(gathered_list, dim=0).cuda()

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size


def as_monte_carlo_sample(high=5233, shape=256, patch=16):
    if(high // patch > shape or high<patch):
        return torch.randint(high, size=(shape,))

    cur_list = []
    if(high % patch != 0):
        sec_high = high - high % patch
        blocks = sec_high // patch
        int_times = shape // blocks
        if (blocks>shape):
            i = 0
            while(len(cur_list) < shape):
                cur_list.append(random.randint(i*patch, (i+1)*patch-1))
                i+=1
        else:
            for i in range(blocks):
                cur_to_add = [random.randint(i*patch, (i+1)*patch-1) for _ in range(int_times//2)]
                cur_list.extend(cur_to_add)
                new_cur_to_add = []
                for item in cur_to_add:
                    new_cur_to_add.append(int((2*i+1)*patch-1) - item)
                # assert new_cur_to_add != cur_to_add
                cur_list.extend(new_cur_to_add)
            while len(cur_list) < shape:
                cur_list.append(random.randint(0, high-1))
    else:
        sec_high = high
        blocks = sec_high // patch
        int_times = shape // (blocks)
        for i in range(blocks):
            cur_to_add = [random.randint(i*patch, (i+1)*patch-1) for _ in range(int_times//2)]
            cur_list.extend(cur_to_add)
            new_cur_to_add = []
            for item in cur_to_add:
                new_cur_to_add.append(int((2*i+1)*patch-1) - item)
            # assert new_cur_to_add != cur_to_add
            cur_list.extend(new_cur_to_add)
        while len(cur_list) < shape:
            cur_list.append(random.randint(0, high-1))
    cur_list = torch.Tensor(cur_list).reshape(shape,).long()

    ind = torch.randperm(shape).long()
    # print(ind)
    # print(cur_list)
    cur_list = cur_list[ind]
    return cur_list


def monte_carlo_sample(high=5233, shape=256, patch=16):
    if(high // patch > shape or high<patch):
        return torch.randint(high, size=(shape,))

    cur_list = []
    if(high % patch != 0):
        sec_high = high - high % patch
        blocks = sec_high // patch
        int_times = shape // blocks
        if (blocks>shape):
            i = 0
            while(len(cur_list) < shape):
                cur_list.append(random.randint(i*patch, (i+1)*patch-1))
                i+=1
        else:
            for i in range(blocks):
                cur_list.extend([random.randint(i*patch, (i+1)*patch-1) for _ in range(int_times)])
            while len(cur_list) < shape:
                cur_list.append(random.randint(0, high-1))

    else:
        sec_high = high
        blocks = sec_high // patch
        int_times = shape // (blocks)
        for i in range(blocks):
            cur_list.extend([random.randint(i*patch, (i+1)*patch-1) for _ in range(int_times)])
        while len(cur_list) < shape:
            cur_list.append(random.randint(0, high-1))
    cur_list = torch.Tensor(cur_list).reshape(shape,).long()

    ind = torch.randperm(shape).long()
    # print(ind)
    # print(cur_list)
    cur_list = cur_list[ind]
    return cur_list


@ torch.no_grad()
def grid_monte_carlo_sample(high=5233, shape=256, cut_count=4):
    # print(high, shape) # 29, 131072
    """_summary_

    Args:
        high (int, optional): highest possible sample. Defaults to 5233.
        shape (int, optional): num_sample . Defaults to 256.
        cut_count (int, optional): cut the overall into cut_count**2 pieces. Defaults to 4.

    Returns:
        torch tensor: sampled indexes
    """
    # if high < shape:
    #     raise ValueError
    try:
        edge = round(math.sqrt(high))
        cur_list = []
        
        per_patch_sample = shape * edge**2 // high  // (cut_count**2)
        img = np.arange(edge**2).reshape(edge, edge)
        patch_edge = edge // cut_count
            
        for i in range(cut_count):
            for j in range(cut_count):
                # we are at the i-th row and j-th col
                if i!= cut_count-1 and j!=cut_count-1:
                    picked_image = img[i*patch_edge:(i+1)*patch_edge, j*patch_edge:(j+1)*patch_edge]
                elif i == cut_count-1 and j!=cut_count-1:
                    picked_image = img[i*patch_edge:, j*patch_edge:(j+1)*patch_edge]
                elif i != cut_count-1 and j==cut_count-1:
                    picked_image = img[i*patch_edge:(i+1)*patch_edge, j*patch_edge:]
                else:
                    picked_image = img[i*patch_edge:, j*patch_edge:]
                    
                picked_image = picked_image.flatten()
                # print('i j: ', (i, j), picked_image.shape)
                ind = torch.randperm(picked_image.shape[0]).long()
                picked_image = picked_image[ind]
                ind = torch.randint(picked_image.shape[0], (per_patch_sample,))
                picked_index = picked_image[ind]
                cur_list.append(picked_index)
            
        cur_list = torch.Tensor(np.array(cur_list)).flatten().long()
        
        mask = cur_list < high
        indices = torch.nonzero(mask)
        
        cur_list = cur_list[indices]

        ind = torch.randperm(cur_list.flatten().shape[0]).long()
        # print(ind)
        # print(cur_list)
        cur_list = cur_list[ind]
        # print(cur_list.shape) # torch.Size([240, 1])
        
        while len(cur_list) < shape:
            cur_list = torch.cat([cur_list, torch.randint(high, (1,1))])
        
        if len(cur_list) > shape:
            cur_list = cur_list[:shape]
        
        return cur_list.squeeze(1)
    except:
        return monte_carlo_sample(high, shape)


@ torch.no_grad()
def grid_as_monte_carlo_sample(high=5233, shape=256, cut_count=4):
    # print(high, shape) # 29, 131072
    """_summary_

    Args:
        high (int, optional): highest possible sample. Defaults to 5233.
        shape (int, optional): num_sample . Defaults to 256.
        cut_count (int, optional): cut the overall into cut_count**2 pieces. Defaults to 4.

    Returns:
        torch tensor: sampled indexes
    """
    # if high < shape:
    #     raise ValueError
    try:
        edge = round(math.sqrt(high))
        cur_list = []
        
        per_patch_sample = shape * edge**2 // high  // (cut_count**2)
        img = np.arange(edge**2).reshape(edge, edge)
        patch_edge = edge // cut_count
            
        for i in range(cut_count):
            for j in range(cut_count):
                # we are at the i-th row and j-th col
                if i!= cut_count-1 and j!=cut_count-1:
                    picked_image = img[i*patch_edge:(i+1)*patch_edge, j*patch_edge:(j+1)*patch_edge]
                elif i == cut_count-1 and j!=cut_count-1:
                    picked_image = img[i*patch_edge:, j*patch_edge:(j+1)*patch_edge]
                elif i != cut_count-1 and j==cut_count-1:
                    picked_image = img[i*patch_edge:(i+1)*patch_edge, j*patch_edge:]
                else:
                    picked_image = img[i*patch_edge:, j*patch_edge:]
                    
                picked_image = picked_image.flatten()
                # print(picked_image)
                center = 2*np.mean(picked_image)
                center = center.astype(np.int64)
                # print('i j: ', (i, j), picked_image.shape)
                ind = torch.randperm(picked_image.shape[0])
                picked_image = picked_image[ind]
                # print(picked_image.shape) # torch.Size([361])
                # print(center) # 1422
                ind = torch.randint(picked_image.shape[0], (per_patch_sample//2,))
                # ano_ind = center-ind
                # if per_patch_sample % 2 != 0 :
                #     ano_ind = torch.cat([ano_ind, torch.randint(picked_image.shape[0], (1,))])
                picked_index = picked_image[ind]
                ano_picked_index = center - picked_index
                cur_list.append(picked_index)
                cur_list.append(ano_picked_index)
                # if per_patch_sample % 2 != 0 :
                #     ind = torch.randint(picked_image.shape[0], (1,1))
                #     extra_index = picked_image[ind]
                #     cur_list.append(extra_index)
            
        # print(cur_list)
        cur_list = torch.Tensor(cur_list)
        cur_list = cur_list.flatten().long()
        
        mask = cur_list < high
        indices = torch.nonzero(mask)
        
        cur_list = cur_list[indices]

        ind = torch.randperm(cur_list.flatten().shape[0]).long()
        # print(ind)
        # print(cur_list)
        cur_list = cur_list[ind]
        # print(cur_list.shape) # torch.Size([240, 1])
        
        while len(cur_list) < shape:
            cur_list = torch.cat([cur_list, torch.randint(high, (1,1))])
        
        if len(cur_list) > shape:
            cur_list = cur_list[:shape]
        
        return cur_list.squeeze(1)
    
    except:
        return as_monte_carlo_sample(high, shape)
    

def compute_contra_memobank_loss(
    rep,
    label_l,
    label_u,
    prob_l,
    prob_u,
    low_mask,
    high_mask,
    memobank,
    queue_prtlis,
    queue_size,
    rep_teacher,
    momentum_prototype=None,
    i_iter=0,
    delta_n = 1.0,
    func = 'asmc', 
    num_queries=256, 
    num_negatives=512,
    temp = 0.5
):
    """_summary_

    Args:
        rep (_type_): _description_: all features stu
        label_l (_type_): _description_: 
        label_u (_type_): _description_:
        prob_l (_type_): _description_: teacher
        prob_u (_type_): _description_: teacher
        low_mask (_type_): _description_: label + low_u
        high_mask (_type_): _description_: label + high_u
        memobank (_type_): _description_
        queue_prtlis (_type_): _description_
        queue_size (_type_): _description_
        rep_teacher (_type_): _description_: all features tea
        
        momentum_prototype (_type_, optional): _description_. Defaults to None.
        i_iter (int, optional): _description_. Defaults to 0.
        delta_n (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    
    current_class_threshold = 0.3
    current_class_negative_threshold = delta_n
    low_rank, high_rank = 3, 20
    # temp = 0.5
    # num_queries = 256
    # num_negatives = 512

    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = label_l.shape[1] # 4
    
    if func =='asmc':
        function_to_use = grid_as_monte_carlo_sample
        num_queries_ = num_queries
        num_negatives_ = num_queries*num_negatives
    elif func == 'smc':
        function_to_use = grid_monte_carlo_sample
        num_queries_ = num_queries
        num_negatives_ = num_queries*num_negatives
    else:
        function_to_use = torch.randint
        num_queries_ = (num_queries, )
        num_negatives_ = (num_queries*num_negatives, )
        

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(
        0, 2, 3, 1
    )  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []
    for i in range(num_segments):
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :]
        rep_mask_low_entropy = (
            prob_seg > current_class_threshold
        ) * low_valid_pixel_seg.bool()
        rep_mask_high_entropy = (
            prob_seg < current_class_negative_threshold
        ) * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # positive sample: center of the class
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        class_mask_u = torch.sum(
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
        # prob_i_classes = prob_indices_l[label_l_mask]
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        )

        negative_mask = rep_mask_high_entropy * class_mask

        keys = rep_teacher[negative_mask].detach()
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if (
        len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        print('a small mini-batch might only contain 1 or no semantic class')
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()

    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).cuda()

        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0
                and memobank[valid_classes[i]][0].shape[0] > 0
            ):
                # select anchor pixel
                # seg_low_entropy_idx = torch.randint(
                #     len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                # )
                seg_low_entropy_idx = function_to_use(\
                    len(seg_feat_low_entropy_list[i]), num_queries_
                    )
                
                # len(seg_feat_low_entropy_list[i]) == 0~1999, num_queries=50, patch=100 (5 patches)
                # => MC per patch: 10 counts (randint)
                # strMC: per patch: 10 counts (0~99, 100~199)
                # asmc: per patch: 5 counts (0~99: 50) => 5 counts
                
                # print("len(seg_feat_low_entropy_list[i]): ", len(seg_feat_low_entropy_list[i])) # len(seg_feat_low_entropy_list[i]):  299680
                # print("seg_low_entropy_idx: ", seg_low_entropy_idx)
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:
                # in some rare cases, all queries in the current query class are easy
                print('all queries in the current query class are easy')
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                # high_entropy_idx = torch.randint(
                #     len(negative_feat), size=(num_queries * num_negatives,)
                # )
                high_entropy_idx = function_to_use(\
                    len(negative_feat), num_negatives_
                    )
                # print("len(negative_feat): ", len(negative_feat)) # len(negative_feat):  15
                # print("high_entropy_idx: ", high_entropy_idx)
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(
                    num_queries, num_negatives, num_feat
                )
                positive_feat = (
                    seg_proto[i]
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)
                    .cuda()
                )  # (num_queries, 1, num_feat)

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )
        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg
        else:
            return prototype, new_keys, reco_loss / valid_seg


def compute_reco_loss(rep, label, mask, prob, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask.cpu()

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg.cpu() < strong_threshold) * valid_pixel_seg.bool().cpu()  # select hard queries
        # print(rep_mask_hard.shape) # torch.Size([20, 81, 81])
        # print(rep.shape) # torch.Size([20, 81, 81, 256])
        # print(rep[rep_mask_hard].shape) # torch.Size([33961, 256])
        # exit()
        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                # seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                seg_hard_idx = monte_carlo_sample(\
                    high=len(seg_feat_hard_list[i]), shape=num_queries, patch=8)
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg

def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index


def get_criterion():
    aux_weight = 0
    ignore_index = 255
    criterion = Criterion(aux_weight, ignore_index=ignore_index, use_weight=False)

    return criterion


class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss()
        else:
            weights = torch.FloatTensor(
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            ).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(
                ignore_index=ignore_index, weight=weights
            )

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(
                    main_pred, target
                )
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss


class CriterionOhem(nn.Module):
    def __init__(
        self,
        aux_weight,
        thresh=0.7,
        min_kept=100000,
        ignore_index=255,
        use_weight=False,
    ):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh, min_kept, use_weight
        )
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred, aux_pred = preds
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert (
                len(preds) == 2
                and main_h == aux_h
                and main_w == aux_w
                and main_h == h
                and main_w == w
            )

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=255, thresh=0.7, min_kept=100000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (
            factor * factor
        )  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if min_kept >= num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = (
            torch.from_numpy(input_label.reshape(target.size()))
            .long()
            .cuda(target.get_device())
        )

        return new_target

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad

        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(predict, target)


class OhemCrossEntropy2dTensor(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=256, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
    
    
def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)


def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).mean()


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            input = input.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input


def mean_tensor(input, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            input = input.mean(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.mean(int(ax))
    return input


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True, smooth_in_nom=True,
                 background_weight=1, rebalance_weights=None):
        """
        hahaa no documentation for you today
        :param smooth:
        :param apply_nonlin:
        :param batch_dice:
        :param do_bg:
        :param smooth_in_nom:
        :param background_weight:
        :param rebalance_weights:
        """
        super(SoftDiceLoss, self).__init__()
        if not do_bg:
            assert background_weight == 1, "if there is no bg, then set background weight to 1 you dummy"
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        self.smooth_in_nom = smooth_in_nom
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.y_onehot = None
        if not smooth_in_nom:
            self.nom_smooth = 0
        else:
            self.nom_smooth = smooth

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x = x.shape
        shp_y = y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        # now x and y should have shape (B, C, X, Y(, Z))) and (B, 1, X, Y(, Z))), respectively
        y_max = torch.max(y)
        y_onehot = torch.zeros(shp_x)
        if x.device.type == "cuda":
            y_onehot = y_onehot.cuda(x.device.index)
        # this is really fancy
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError("nah son")
            l = soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        else:
            l = soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                      background_weight=self.background_weight,
                                      rebalance_weights=self.rebalance_weights)
        return l


def soft_dice_per_batch(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1):
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    weights = torch.ones(intersect.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth)) * weights).mean()
    return result


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1, rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:]  # this is the case when use_bg=False
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    net_output_sqaure = sum_tensor(net_output*net_output, axes, keepdim=False)
    gt_square = sum_tensor(gt*gt, axes, keepdim=False)
    #fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    # fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(intersect.shape)
    weights[0] = background_weight
    if net_output.device.type == "cuda":
        weights = weights.cuda(net_output.device.index)
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float()
        if net_output.device.type == "cuda":
            rebalance_weights = rebalance_weights.cuda(net_output.device.index)
        intersect = intersect * rebalance_weights
        # fn = fn * rebalance_weights
    result = (1 - (2*intersect + smooth_in_nom)/(net_output_sqaure + gt_square + smooth) * weights)
    result = result[result > 0]  # ensure that when there is no target class, the dice loss is not too large
    result = result.mean()
    return result


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    result = (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()  #TODO: Was ist weights and er Stelle?
    return result


class SupConSegLoss(nn.Module):
    # TODO: only support batch size = 1
    def __init__(self, temperature=0.7):
        super(SupConSegLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def forward(self, features, labels=None):
        # input features: [bsz, c, h ,w], h & w are the image size
        shape = features.shape
        img_size = shape[-1]
        if labels is not None:
            f1, f2 = torch.split(features, [1, 1], dim=1)
            features = torch.cat([f1.squeeze(1), f2.squeeze(1)], dim=0)
            l1, l2 = torch.split(labels, [1, 1], dim=1)
            labels = torch.cat([l1.squeeze(1), l2.squeeze(1)], dim=0)
            # features = features.squeeze(dim=1)
            # labels = labels.squeeze(dim=1)
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                for i in range(img_size):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    for j in range(img_size):
                        x = features[b:b + 1, :, i:i + 1, j:j + 1]  # [1,c, 1, 1, 1]
                        x_label = labels[b, i, j] + 1  # avoid cases when label=0
                        if x_label == 1:  # ignore background
                            continue
                        cos_dst = F.conv2d(features, x)  # [2b, 1, 512, 512]
                        cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                        # print("cos_dst:", cos_dst.max(), cos_dst.min())
                        self_contrast_dst = torch.div((x * x).sum(), self.temp)

                        mask = labels + 1
                        mask[mask != x_label] = 0
                        # if mask.sum() < 5:
                        #    print("Not enough same label pixel")
                        #    continue
                        mask = torch.div(mask, x_label)
                        numerator = (mask * cos_dst).sum() - self_contrast_dst
                        denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                        # print("denominator:", denominator.item())
                        # print("numerator:", numerator.max(), numerator.min())
                        loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                        if loss_tmp != loss_tmp:
                            print(numerator.item(), denominator.item())

                        loss.append(loss_tmp)
            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss

        else:
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                tmp_feature = features[b]
                for n in range(tmp_feature.shape[0]):
                    for i in range(img_size):
                        # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                        for j in range(img_size):
                            x = tmp_feature[n:n+1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                            cos_dst = F.conv2d(tmp_feature, x)  # [2b, 1, 512, 512]
                            cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                            # print("cos_dst:", cos_dst.max(), cos_dst.min())
                            self_contrast_dst = torch.div((x * x).sum(), self.temp)

                            mask = torch.zeros((tmp_feature.shape[0], tmp_feature.shape[2], tmp_feature.shape[3]),
                                               device=self.device)
                            mask[0:tmp_feature.shape[0], i, j] = 1
                            numerator = (mask * cos_dst).sum() - self_contrast_dst
                            denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                            # print("numerator:", numerator.max(), numerator.min())
                            loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                            if loss_tmp != loss_tmp:
                                print(numerator.item(), denominator.item())

                            loss.append(loss_tmp)

            loss = torch.stack(loss).mean()
            return loss

class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        # input features shape: [bsz, v, c, w, h]
        # input labels shape: [bsz, v, w, h]
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # of size (bsz*v, c, h, w)
        # print(contrast_feature.shape) # torch.Size([2048, 256, 256])
        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        # kernels = kernels[non_background_idx]
        logits = torch.div(F.conv2d(contrast_feature, kernels), self.temperature)  # of size (bsz*v, bsz*v*h*w, h, w)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)

        if labels is not None:
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            bg_bool = torch.eq(labels.squeeze().cpu(), -torch.ones(labels.squeeze().shape))
            bg_bool_ne = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
            bg_bool = bg_bool | bg_bool_ne
            non_bg_bool = ~ bg_bool
            non_bg_bool = non_bg_bool.int().to(device)
        else:
            mask = torch.eye(logits.shape[0]//contrast_count).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)
            # print(mask.shape)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        if labels is not None:
            # only consider the contrastive loss for non-background pixel
            loss = (loss * non_bg_bool).sum() / (non_bg_bool.sum())
        else:
            loss = loss.mean()
        return loss


class LocalConLoss(nn.Module):
    def __init__(self, temperature=0.7, stride=4):
        super(LocalConLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=self.temp)
        self.stride = stride

    def forward(self, features, labels=None):
        # input features: [bsz, num_view, c, h ,w], h & w are the image size
        features = features[:, :, :, ::self.stride, ::self.stride]  # resample feature maps to reduce memory consumption and running time
        shape = features.shape
        img_size = shape[-1]
        if labels is not None:
            labels = labels[:, :, ::self.stride, ::self.stride]
            if labels.sum() == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss

            loss = self.supconloss(features, labels)
            """
            f1, f2 = torch.split(features, [1, 1], dim=1)
            features = torch.cat([f1.squeeze(1), f2.squeeze(1)], dim=0)
            l1, l2 = torch.split(labels, [1, 1], dim=1)
            labels = torch.cat([l1.squeeze(1), l2.squeeze(1)], dim=0)
            bsz = features.shape[0]
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                for i in range(img_size):
                    # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                    for j in range(img_size):
                        x = features[b:b + 1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                        x_label = labels[b, i, j] + 1  # avoid cases when label=0
                        if x_label == 1:  # ignore background
                            continue
                        cos_dst = F.conv2d(features, x)  # [2b, 1, 512, 512]
                        cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                        self_contrast_dst = torch.div((x * x).sum(), self.temp)

                        mask = labels + 1
                        mask[mask != x_label] = 0
                        mask = torch.div(mask, x_label)
                        numerator = (mask * cos_dst).sum() - self_contrast_dst
                        denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                        # print("denominator:", denominator.item())
                        # print("numerator:", numerator.max(), numerator.min())
                        loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                        if loss_tmp != loss_tmp:
                            print(numerator.item(), denominator.item())

                        loss.append(loss_tmp)

            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            """
            return loss
        else:
            bsz = features.shape[0]
            loss = self.supconloss(features)

            """
            loss = []
            for b in range(bsz):
                # print("Iteration index:", idx, "Batch_size:", b)
                tmp_feature = features[b]
                for n in range(tmp_feature.shape[0]):
                    for i in range(img_size):
                        # print("before ith iteration, the consumption memory is:", torch.cuda.memory_allocated() / 1024**2)
                        for j in range(img_size):
                            x = tmp_feature[n:n+1, :, i:i + 1, j:j + 1]  # [c, 1, 1, 1]
                            cos_dst = F.conv2d(tmp_feature, x)  # [2b, 1, 512, 512]
                            cos_dst = torch.div(cos_dst.squeeze(dim=1), self.temp)
                            # print("cos_dst:", cos_dst.max(), cos_dst.min())
                            self_contrast_dst = torch.div((x * x).sum(), self.temp)

                            mask = torch.zeros((tmp_feature.shape[0], tmp_feature.shape[2], tmp_feature.shape[3]),
                                               device=self.device)
                            mask[0:tmp_feature.shape[0], i, j] = 1
                            numerator = (mask * cos_dst).sum() - self_contrast_dst
                            denominator = torch.exp(cos_dst).sum() - torch.exp(self_contrast_dst)
                            # print("numerator:", numerator.max(), numerator.min())
                            loss_tmp = torch.log(denominator) - numerator / (mask.sum() - 1)
                            if loss_tmp != loss_tmp:
                                print(numerator.item(), denominator.item())

                            loss.append(loss_tmp)

            loss = torch.stack(loss).mean()
            """
            return loss


if __name__ == '__main__':
    # criterion = SupConLoss(temperature=0.1, base_temperature=0.01)
    # criterion = LocalConLoss(temperature=0.7, stride=4)
    # pred = torch.rand(4, 2, 256, 256, 256)
    # pred = torch.nn.functional.normalize(pred, dim=2)
    # label = torch.ones(4, 2, 256, 256)
    # loss = criterion(pred, label)
    # print(loss)
    res = grid_as_monte_carlo_sample(high=6133, shape=256, cut_count=4)
    print(res) # torch.Size([256, 1])
    
    # print(torch.randint(
    #                 511, size=(256,)
    #             ).shape)
    # test = torch.randint(1, (512,))
    # print(test)