import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from networks.net_factory_args import net_factory
from networks.net_factory_3dArgs import net_factory_3d
import numpy as np
# batch_size = 24

# outputs_shape = (4, 256, 256) # -> b_s, 4, 256, 256  -> 256 * (b_s, 4, 16, 16) -> 256 * (b_s, 4, 16, 16) (student)
# https://github.com/xhu248/semi_cotrast_seg       
# https://github.com/tfzhou/ContrastiveSeg/blob/main/lib/models/modules/seg_basic.py segmentation head               

# latent_vector_shape = (256, 16, 16) # -> 4 * (64, b_s, 16, 16)  #64->16->4           -> 4*4, b_s* 16*16 
                                        # 4*4, B_s*16*16
                                        # 4096 (256, 8, 8)  => 256*8*8 => 2048/1024/4096


class FeatureExtractor_3d(nn.Module):
    #### can be modified as moe ####
    def __init__(self, fea_dim=[128, 64, 32, 16, 16], output_dim=128) -> None:
        super().__init__()
        assert len(fea_dim)==5, 'input_dim is not correct'
        cnt = fea_dim[0]
        self.fea0 = nn.Conv3d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[1]
        self.fea1 = nn.Conv3d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[2]
        self.fea2 = nn.Conv3d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[3]
        self.fea3 = nn.Conv3d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        # cnt += fea_dim[4]
        # self.fea4 = nn.Conv3d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[4]
        self.fea4 = nn.Conv3d(in_channels=cnt, out_channels=output_dim, kernel_size=1, bias=False)
        
    def forward(self, fea_list):
        feature0 = fea_list[0]
        feature1 = fea_list[1]
        feature2 = fea_list[2]
        feature3 = fea_list[3]
        feature4 = fea_list[4]
        # feature5 = fea_list[5]
        
        x = self.fea0(feature0) + feature0
        x = nn.Upsample(size = feature1.shape[-3:], mode='trilinear', align_corners=True)(x)
        x = torch.cat((x, feature1), dim=1)
        x = self.fea1(x) + x
        x = nn.Upsample(size = feature2.shape[-3:], mode='trilinear', align_corners=True)(x)
        x = torch.cat((x, feature2), dim=1)
        x = self.fea2(x) + x
        x = nn.Upsample(size = feature3.shape[-3:], mode='trilinear', align_corners=True)(x)
        x = torch.cat((x, feature3), dim=1)
        x = self.fea3(x) + x
        x = nn.Upsample(size = feature4.shape[-3:], mode='trilinear', align_corners=True)(x)
        x = torch.cat((x, feature4), dim=1)
        x = self.fea4(x) # + x
        # x = nn.Upsample(size = feature5.shape[-3:], mode='trilinear', align_corners=True)(x)
        # x = torch.cat((x, feature5), dim=1)
        # x = self.fea5(x)
        # print(x.shape) # ([2, 256, 256, 256])
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, fea_dim=[256, 128, 64, 32, 16], output_dim=256) -> None:
        super().__init__()
        assert len(fea_dim)==5, 'input_dim is not correct'
        cnt = fea_dim[0]
        self.fea0 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[1]
        self.fea1 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[2]
        self.fea2 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[3]
        self.fea3 = nn.Conv2d(in_channels=cnt, out_channels=cnt, kernel_size=1, bias=False)
        cnt += fea_dim[4]
        self.fea4 = nn.Conv2d(in_channels=cnt, out_channels=output_dim, kernel_size=1, bias=False)
        
    def forward(self, fea_list):
        feature0 = fea_list[0]
        feature1 = fea_list[1]
        feature2 = fea_list[2]
        feature3 = fea_list[3]
        feature4 = fea_list[4]
        x = self.fea0(feature0) + feature0
        x = nn.Upsample(size = feature1.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature1), dim=1)
        x = self.fea1(x) + x
        x = nn.Upsample(size = feature2.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature2), dim=1)
        x = self.fea2(x) + x
        x = nn.Upsample(size = feature3.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature3), dim=1)
        x = self.fea3(x) + x
        x = nn.Upsample(size = feature4.shape[-2:], mode='bilinear', align_corners=True)(x)
        x = torch.cat((x, feature4), dim=1)
        x = self.fea4(x) 
        # print(x.shape) # ([2, 256, 256, 256])
        return x

def create_model(ema=False, num_classes=4, train_encoder=True, train_decoder=True):
        # Network definition
        model = net_factory(net_type='unet', in_chns=1,
                            class_num=num_classes, train_encoder=train_encoder, train_decoder=train_decoder)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


def create_model_3d(ema=False, num_classes=4):
    # Network definition
    model = net_factory_3d(net_type="vnet", in_chns=1,
                        class_num=num_classes)
    if ema:
            for param in model.parameters():
                param.detach_()
    return model


class ProjectionHead(nn.Module):
    def __init__(self, dim_in=4, proj_dim=4, output_pooling_size=16, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_pooling_size),
                nn.Conv2d(dim_in, dim_in*2, kernel_size=1),
                nn.Conv2d(dim_in*2, proj_dim, kernel_size=1)
            )
        # 第一种：只有一个linear，https://github.com/tfzhou/ContrastiveSeg/blob/a2f7392d0c50ecdda36fd41205b77a7bc74f7cb8/lib/models/modules/projection.py#L15
        # 第二种： https://github.com/tfzhou/ContrastiveSeg/blob/a2f7392d0c50ecdda36fd41205b77a7bc74f7cb8/lib/models/modules/projection.py#L17
        # no normalize

    def forward(self, x):
        return self.proj(x)


class ProjectionHead_3d(nn.Module):
    def __init__(self, dim_in=4, proj_dim=4, output_pooling_size=16, proj='convmlp'):
        super(ProjectionHead_3d, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv3d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.AdaptiveAvgPool3d(output_pooling_size),
                nn.Conv3d(dim_in, dim_in*2, kernel_size=1),
                nn.Conv3d(dim_in*2, proj_dim, kernel_size=1)
            )
        # 第一种：只有一个linear，https://github.com/tfzhou/ContrastiveSeg/blob/a2f7392d0c50ecdda36fd41205b77a7bc74f7cb8/lib/models/modules/projection.py#L15
        # 第二种： https://github.com/tfzhou/ContrastiveSeg/blob/a2f7392d0c50ecdda36fd41205b77a7bc74f7cb8/lib/models/modules/projection.py#L17
        # no normalize

    def forward(self, x):
        return self.proj(x)


class RepresentationHead(nn.Module):
    def __init__(self, num_classes=256+128+64+32+16, output_channel=512):
        super(RepresentationHead, self).__init__()
        self.proj = nn.Sequential(
                nn.Conv2d(num_classes, output_channel, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(output_channel, output_channel, kernel_size=1)
            )
    def forward(self, x):
        return self.proj(x)


class RepresentationHead_3d(nn.Module):
    def __init__(self, num_classes=256+128+64+32+16, output_channel=512):
        super(RepresentationHead_3d, self).__init__()
        self.proj = nn.Sequential(
                nn.Conv3d(num_classes, output_channel, kernel_size=3, padding=1, bias=False),
                nn.Conv3d(output_channel, output_channel, kernel_size=1)
            )
    def forward(self, x):
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, input_channels=256, num_class=128, pooling_size=1):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool2d(pooling_size)
        self.f1 = nn.Linear(input_channels*pooling_size**2, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        y = self.f1(x)
        y = self.f2(y)

        return y
    
    
class MLP_3d(nn.Module):
    def __init__(self, input_channels=256, num_class=128, pooling_size=1):
        super().__init__()

        self.gap = nn.AdaptiveAvgPool3d(pooling_size)
        self.f1 = nn.Linear(input_channels*pooling_size**2, input_channels)
        self.f2 = nn.Linear(input_channels, num_class)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        y = self.f1(x)
        y = self.f2(y)

        return y


class ISD_3d(nn.Module):
    def __init__(self, K=48, m=0.99, Ts=0.1, Tt = 0.01, num_classes=4, train_encoder=True, train_decoder=True, 
                latent_pooling_size=1, latent_feature_size=128, output_pooling_size=4, patch_size=64): # K=48
        super(ISD_3d, self).__init__()

        self.K = K
        self.m = m
        self.Ts = Ts
        self.Tt = Tt
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.latent_feature_size = latent_feature_size

        self.model = create_model_3d(num_classes=num_classes, )

        self.ema_model = create_model_3d(ema=True, num_classes=num_classes)

        self.k_latent_head = MLP_3d(input_channels=128, num_class=self.latent_feature_size, pooling_size=latent_pooling_size)

        self.q_latent_head = MLP_3d(input_channels=128, num_class=self.latent_feature_size, pooling_size=latent_pooling_size)

        self.latent_predictor = nn.Sequential(
            nn.Linear(self.latent_feature_size, self.latent_feature_size),
            nn.Linear(self.latent_feature_size, self.latent_feature_size),
        )

        
        self.k_outputs_head = ProjectionHead_3d(dim_in=num_classes, proj_dim=num_classes, output_pooling_size=output_pooling_size)
        self.q_outputs_head = ProjectionHead_3d(dim_in=num_classes, proj_dim=num_classes, output_pooling_size=output_pooling_size)
        

        self.outputs_predictor = nn.Sequential(
            nn.Conv3d(num_classes, num_classes, kernel_size=1),
            nn.Conv3d(num_classes, num_classes, kernel_size=1),
        )

        # copy query encoder weights to key encoder
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # setup queue
        self.register_buffer('queue', torch.randn(self.K, self.latent_feature_size))
        self.register_buffer('queue_mask', torch.randn(self.K, 700, num_classes*output_pooling_size**3))

        # normalize the queue
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.queue_mask = nn.functional.normalize(self.queue_mask, dim=-1)

        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('mask_queue_ptr', torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.model.parameters(), self.ema_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.q_outputs_head.parameters(), self.k_outputs_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.q_latent_head.parameters(), self.k_latent_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def data_parallel(self):
        self.model = torch.nn.DataParallel(self.model)
        self.ema_model = torch.nn.DataParallel(self.ema_model)
        self.k_latent_head = torch.nn.DataParallel(self.k_latent_head)
        self.q_latent_head = torch.nn.DataParallel(self.q_latent_head)
        self.latent_predictor = torch.nn.DataParallel(self.latent_predictor)
        self.k_outputs_head = torch.nn.DataParallel(self.k_outputs_head)
        self.q_outputs_head = torch.nn.DataParallel(self.q_outputs_head)
        self.outputs_predictor = torch.nn.DataParallel(self.outputs_predictor)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.K % batch_size == 0 

        # replace the keys at ptr (dequeue and enqueue)
        queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.K  # move pointer

        queue_ptr[0] = ptr


    def forward(self, im_q, im_k, Ts=None, Tt=None):

        if not Ts:
            Ts = self.Ts
        if not Tt:
            Tt = self.Tt
        batch_size = im_q.shape[0]

        if not self.training:
            outputs, latent_vector, _ = self.model(im_q)
            # outputs = self.model_seg(outputs)
            return outputs, latent_vector 

        outputs, latent_vector, _ = self.model(im_q) 
        outputs_tmp = outputs

        with torch.no_grad():
            # noise = torch.clamp(torch.randn_like(im_k) * 0.1, -0.2, 0.2)
            ema_inputs = im_k 
            ema_output_tmp, _, _ = self.ema_model(ema_inputs)


        with torch.no_grad():
            # update the key encoder
            self._momentum_update_key_encoder()
            
            # shuffle keys
            shuffle_ids, reverse_ids = get_shuffle_ids(im_k.shape[0])
            im_k = im_k[shuffle_ids]

            ema_output, ema_latent_vector, _ = self.ema_model(im_k)

            # undo shuffle
            ema_latent_vector = ema_latent_vector[reverse_ids]
            ema_output = ema_output[reverse_ids]

        # calculate similarities
        queue = self.queue.clone().detach() 
        queue_mask = self.queue_mask.clone().detach().transpose(0, 1).contiguous() 
        

        patch_size = self.patch_size
        step = self.patch_size // 2
        output_stu_after_head = []
        output_tea_after_head = []
        # outputs (b_s, 4, 256, 256)
        for i in range(0, outputs.shape[2]-patch_size+1, step):
            for j in range(0, outputs.shape[3]-patch_size+1, step):
                for k in range(0, outputs.shape[4]-patch_size+1, step):
                    output_stu_after_head.append(self.outputs_predictor(self.q_outputs_head(outputs[:, :, i:i+patch_size, j:j+patch_size, k:k+patch_size])))
                    output_tea_after_head.append(self.k_outputs_head(ema_output[:, :, i:i+patch_size, j:j+patch_size, k:k+patch_size]))

        
        output_stu_after_head = torch.cat(output_stu_after_head).reshape(batch_size, -1, 
                                        output_stu_after_head[0].shape[1], 
                                        output_stu_after_head[0].shape[2], 
                                        output_stu_after_head[0].shape[3], 
                                        output_stu_after_head[0].shape[4], ).contiguous() 
 
        
        output_tea_after_head = torch.cat(output_tea_after_head).reshape(batch_size, -1, 
                                        output_tea_after_head[0].shape[1], 
                                        output_tea_after_head[0].shape[2], 
                                        output_tea_after_head[0].shape[3], 
                                        output_tea_after_head[0].shape[4], ).contiguous() # torch.Size([4, 700, 4, 8, 8, 8])
        

        desired_compressed_lat_k  = self.k_latent_head(ema_latent_vector) # torch.Size([24, 256])

        desired_compressed_lat_q = self.latent_predictor(self.q_latent_head(latent_vector)) # torch.Size([24, 256])

        output_tea_after_head_tmp = output_tea_after_head.reshape(output_tea_after_head.shape[0], output_tea_after_head.shape[1], -1).contiguous()
        
        output_stu_after_head = output_stu_after_head.reshape((output_stu_after_head.shape[1], batch_size, -1)).contiguous() 
        output_tea_after_head = output_tea_after_head.reshape((output_tea_after_head.shape[1], batch_size, -1)).contiguous() 

        output_stu_after_head = output_stu_after_head.reshape(-1, output_stu_after_head.shape[0]).contiguous() # xxx*36
        output_tea_after_head = output_tea_after_head.reshape(-1, output_tea_after_head.shape[0]).contiguous() # xxx*36
        
        queue_mask = queue_mask.reshape(-1, queue_mask.shape[0]).contiguous()

        # compute the 4 logits
        ema_latent_logits = compute_logits(desired_compressed_lat_k, queue, Tt) 
        latent_logits = compute_logits(desired_compressed_lat_q, queue, Ts) 

        ema_output_logits = compute_logits(output_tea_after_head, queue_mask, Tt) 
        output_logits = compute_logits(output_stu_after_head, queue_mask, Ts) 

        
        # # dequeue and enqueue
        self._dequeue_and_enqueue(desired_compressed_lat_k, self.queue, self.queue_ptr)
        self._dequeue_and_enqueue(output_tea_after_head_tmp, self.queue_mask, self.mask_queue_ptr)
        

        return outputs_tmp, ema_output_tmp, ema_latent_logits, latent_logits, ema_output_logits, output_logits
        
        

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    # torch.arange(bsz) -> 0~(bsz-1)
    value = torch.arange(bsz).long().cuda()
    # Tensor.index_copy_(dim, index, tensor)
    # if dim == 0 and index[i] == j, then the ith row of tensor is copied to the jth row of self
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def compute_logits(z_anchor, z_positive, temp_fac):
    # z_anchor_norm = torch.linalg.norm(z_anchor.cuda(), dim=1) + 1e-6
    # z_positive_norm = torch.linalg.norm(z_positive.cuda(), dim=1) + 1e-6
    # print(z_anchor.shape) # torch.Size([512])
    z_anchor = nn.functional.normalize(z_anchor, dim=-1)
    z_positive = nn.functional.normalize(z_positive, dim=-1)
    logits_out = torch.matmul(z_anchor.cuda(), z_positive.T.cuda())/temp_fac
    # logits_out = logits_out / z_anchor_norm.view(-1, 1)
    # logits_out = logits_out / z_positive_norm.view(1, -1)
    # logits_out = logits_out - torch.max(logits_out, 1)[0][:, None]
    return logits_out


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    torch.cuda.empty_cache()
    model = ISD_3d(K=24, m=0.99, Ts=0.07, Tt=0.01, num_classes=2, train_encoder=True, train_decoder=True, 
                latent_pooling_size=1, latent_feature_size=128, output_pooling_size=4, patch_size=20).cuda()
    # model.data_parallel()
    input = torch.ones([4, 1, 112, 112, 80]).cuda()
    for i in range(5):
        outputs, ema_output_tmp, uncertainty, \
            ema_latent_logits, latent_logits, ema_output_logits, output_logits, \
                feat_map_stu, feat_map_tea= model(input, input, 0.1, 0.1)
        # exit()
        print(outputs.shape)
        
        # print(ema_output.shape)
        # print(uncertainty.shape)
        # print(ema_latent_logits.shape)
        # print(latent_logits.shape), 
        # print(ema_output_logits.shape)
        # print(output_logits.shape)
        # print(purity.shape)