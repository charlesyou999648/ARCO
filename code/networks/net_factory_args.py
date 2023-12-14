from networks.efficientunet import Effi_UNet
from networks.enet import ENet
from networks.pnet import PNet2D
from networks.unetWithArgs import UNet, UNet_DS, UNet_URPC, UNet_CCT
# from networks.unetWithArgsSTEGO import UNet, UNet_DS, UNet_URPC, UNet_CCT
import argparse
# from networks.vision_transformer import SwinUnet as ViT_seg
from networks.config import get_config
from networks.nnunet import initialize_network
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import torch

def net_factory(net_type="unet", in_chns=1, class_num=3, train_encoder=True, train_decoder=True, unfreeze_seg=True):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num, \
            train_encoder=train_encoder, train_decoder=train_decoder, unfreeze_seg=unfreeze_seg).cuda()
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "ViT_Seg":
        net = ViT_seg(CONFIGS_ViT_seg['R50-ViT-B_16'], img_size=256,
                      num_classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    else:
        net = None
    return net

if __name__ == '__main__':
    model = net_factory(net_type="ViT_Seg", ).cuda()
    inp = torch.randn(4, 1, 256, 256).cuda()
    out, latent, mid = model(inp)
    print([item.shape for item in mid])
    print(out.shape)
    print(latent.shape)