# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# import argparse
# import os
# import torch
# from torch import optim
#
# import model_change  # 这是我自己改进的模型
#
# from util import str2bool
# from utils import *
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # 针对作者已经编写好的模型进行命名
# # model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
# # 针对我自己改进的模型进行命名
# model_names = sorted(name for name in model_change.__dict__ if name.islower() and not name.startswith("__"))

#########################   将模型中的网络层参数进行分组训练网络   #############################
# # Parse arguments
# parser = argparse.ArgumentParser(allow_abbrev=False)
#
# # Model
# parser.add_argument("--arch", type=str, default="bi3dnet_binary_depth")
#
# parser.add_argument("--bi3dnet_featnet_arch", type=str, default="featextractnetspp")
# parser.add_argument("--bi3dnet_featnethr_arch", type=str, default="featextractnethr")
# parser.add_argument("--bi3dnet_segnet_arch", type=str, default="segnet2d")
# parser.add_argument("--bi3dnet_refinenet_arch", type=str, default="segrefinenet")
# parser.add_argument("--bi3dnet_max_disparity", type=int, default=192)
# parser.add_argument("--bi3dnet_disps_per_example_true", type=str2bool, default=True)
#
# parser.add_argument("--featextractnethr_out_planes", type=int, default=16)
# parser.add_argument("--segrefinenet_in_planes", type=int, default=17)
# parser.add_argument("--segrefinenet_out_planes", type=int, default=8)
#
# parser.add_argument("--pretrained", type=str,
#                     default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/sf_binary_depth.pth.tar")
#
# # parse arguments
# args, unknown = parser.parse_known_args()
# options = vars(args)
#
# # Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# network_data = torch.load(args.pretrained, map_location=device)  # 加载预训练模型
# model = model_change.__dict__[args.arch](options, network_data).to(device)  # 没有预训练模型，模型参数随机初始化。
#
# # 只对新增添的网络层和以该网络的输出为输入的网络进行参数反向传播，其余网络层的参数不变。
# for name, parms in model.named_parameters():
#     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data))
#
# # 将某一网络层的参数转化为list打印在终端。
# print(list(model.refinenet.conv2.parameters()))
#
# # 在SegNet网络的最后增添大小为1*1的卷积核，训练时只需让SegNet网络最后新增添的卷积层last_last_conv和优化网络的网络参数加载到优化器中。
# parma_grad = list(model.segnet.last_last_conv.parameters()) + list(model.refinenet.parameters())
# optimizer = optim.Adam(parma_grad, betas=(0.9, 0.999))
#
# # 在SegRefine网络的最后增添大小为3*3的卷积核，并且只将该卷积核的参数加载到优化器中。
# optimizer = optim.Adam(model.refinenet.conv2.parameters(), betas=(0.9, 0.999))
#
# # 将优化器定义时加载的网络层参数打印出来，看加载对了没有。
# for para_groups in optimizer.param_groups:
#     print(para_groups['params'])
#
# # 整个测试过程结束以后，删除所有定义的变量以释放内存。
# del optimizer
# del model
# del network_data

# #########################   使用summary函数将模型可视化   #######################
# # 将模型加载到的设备。
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 引入summary函数
# from torchsummary import summary
#
# # 将提取立体图像对的网络结构可视化
# from model_change.PSMNet import FeatExtractNetSPP
# featnet = FeatExtractNetSPP().to(device)
# summary(featnet, (3, 960, 576))
#
# # 将只提取左图特征的网络使用summary函数进行可视化
# from model_change.FeatExtractNet import FeatExtractNetHR
# featnetr = FeatExtractNetHR(out_planes=16).to(device)
# summary(featnetr, (3, 960, 576))
#
# # 将二元深度估计网络内部的图像分割网络进行可视化
# from model_change.SegNet2D import SegNet2D
# segnet = SegNet2D().to(device)
# summary(segnet, (64, 192, 384))
#
# # 将对输出视差进行优化的网络SegRefine进行可视化
# from model_change.RefineNet2D import SegRefineNet
# refinenet = SegRefineNet(in_planes=17, out_planes=8).to(device)
# summary(refinenet, (17, 576, 960))

#########################   使用tensorboard界面将模型可视化   #######################
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='model_visualization', comment='SegRefine')
from model_change.RefineNet2D import SegRefineNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
refinenet = SegRefineNet(in_planes=17, out_planes=8).to(device)
fake_img = torch.randn(1, 17, 576, 960)
fake_img = fake_img.cuda()
writer.add_graph(refinenet, fake_img)
writer.close()
