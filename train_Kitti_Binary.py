# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

import model_change
import cv2
import numpy as np
from util import disp2rgb, str2bool

import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from utils import *
from torch.utils.data import DataLoader
import gc
import random

model_names = sorted(name for name in model_change.__dict__ if name.islower() and not name.startswith("__"))

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parse arguments
parser = argparse.ArgumentParser(allow_abbrev=False)

# Model
parser.add_argument("--arch", type=str, default="bi3dnet_binary_depth")

parser.add_argument("--bi3dnet_featnet_arch", type=str, default="featextractnetspp")
parser.add_argument("--bi3dnet_featnethr_arch", type=str, default="featextractnethr")
parser.add_argument("--bi3dnet_segnet_arch", type=str, default="segnet2d")
parser.add_argument("--bi3dnet_refinenet_arch", type=str, default="segrefinenet")
parser.add_argument("--bi3dnet_max_disparity", type=int, default=192)
parser.add_argument("--bi3dnet_disps_per_example_true", type=str2bool, default=True)

parser.add_argument("--featextractnethr_out_planes", type=int, default=16)
parser.add_argument("--segrefinenet_in_planes", type=int, default=17)
parser.add_argument("--segrefinenet_out_planes", type=int, default=8)

# Input
parser.add_argument("--pretrained", type=str,
                    default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/kitti15_binary_depth.pth.tar")
parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--kitti15_datapath', default='/media/fhy/My_Passport/data_scene_flow/', help='data path')
parser.add_argument('--trainlist', default='./filenames/kitti15_train.txt', help='training list')

parser.add_argument('--lr', type=float, default=0.00001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lrepochs', default="20,32,40,48,56:2", type=str, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='log_tree', help='the directory to save logs and checkpoints')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=2, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
options = vars(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# 创建保存训练模型的日志文件夹
Time = time.time()
output_dir = "{}/{}".format(args.logdir, str(Time))
os.makedirs(output_dir, exist_ok=True)
del Time

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
# 初始化训练的数据集(执行的为SceneFlowDatset中的__init__部分)
train_dataset = StereoDataset(args.kitti15_datapath, args.trainlist, True)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)

# Model
device = torch.device('cuda')
network_data = torch.load(args.pretrained, map_location=device)
# checkpoint = network_data['state_dict']
# new_checkpoint = {}
# for k, v in checkpoint.items():
#     new_k = k.replace('module.', '') if 'module' in k else k
#     new_checkpoint[new_k] = v
# print("=> using pre-trained model '{}'".format(args.arch))
# 初始化训练模型(执行的是Bi3DNetBinaryDepth中的__init__部分)
# model = models.__dict__[args.arch](options, new_checkpoint).to(device)
model = model_change.__dict__[args.arch](options, network_data).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# 初始化交叉熵损失函数(最后将所有像素点的损失值求平均)
crit = torch.nn.BCELoss(reduction='mean')

start_epoch = 0  # 初始迭代参数
print("start at epoch {}".format(start_epoch))


def train():
    # 模型训练一共迭代epochs次，每一次都将误差反向传播，不断地让拟合视差图接近真实视差图。
    for epoch_idx in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()  # 每次训练之前将缓存清空，防止报cuda out of memery的错误。
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)  # 训练到一定次数时，需要调整学习率。

        # 在每一轮训练中,再记录一组该轮训练的平均损失函数
        Loss = 0.0

        # training
        # 每次抓取batch_size对立体图像对和相应的视差图进行训练。真正抓取图像的时候执行的是Bi3DNetBinaryDepth中的__getitem__部分。
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, crit)
            if do_summary:
                save_scalars(logger, 'loss-batch', scalar_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
            Loss += loss

        Loss = Loss / len(TrainImgLoader)
        print('Epoch {}/{}, train loss = {:.3f}'.format(epoch_idx, args.epochs, Loss))

        # 再记录一组损失函数随训练代数的变化曲线
        Scalar_average_loss = {"Loss": Loss}
        save_scalars(logger, 'loss-epoch', Scalar_average_loss, epoch_idx)
        del Scalar_average_loss

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # id_epoch = (epoch_idx + 1) % 100
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()
    # 写完日志文件后关闭。
    logger.close()


def train_sample(sample, loss_fn):
    model.train()  # 模型处于训练模式，有误差的反向传播
    # 先将批图像读进来
    imgL, imgR, pad_H, pad_W, disp_ref, br_disp_gt, disp_gt = \
        sample["left"], sample["right"], sample["top_pad"], sample["left_pad"], \
        sample["dis_ref"], sample["binary_disparity"], sample["disparity"]

    optimizer.zero_grad()

    # input convert to cuda format
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    pad_H = pad_H.cuda()
    pad_W = pad_W.cuda()
    disp_ref = disp_ref.cuda()
    br_disp_gt = br_disp_gt.cuda()
    disp_gt = disp_gt.cuda()

    # Inference
    output = model(imgL, imgR, disp_ref)[1][:, :, pad_H[torch.argmax(pad_H)]:, pad_W[torch.argmax(pad_W)]:]
    # 将output的psv维度和batch维度互换，并取出维度为B*H*W的置信度disp_ests。
    # 其中，取出的经网络估计的置信度的大小为540*960，把填充的部分通过[:, :, pad_H[0]:, pad_W[0]:]截取掉了。
    disp_ests = output.transpose(0, 1)[0]

    # kitti数据集中图像的大小不尽相同。有的大小为375*1241，有的为376*1242。
    # 将图像通过KittiDataset中的getitem函数抓取进来的时候，先将左、右立体图像对和真值视差图填充到384*1248的分辨率，避免将批图像读进sample的时候报维度不一致的错误。
    # 然后，将计算出的视差结果、真值视差图和二分类的真值视差图裁剪到375*1241的大小，将填充的部分裁剪掉。

    # save result
    br_disp_gt = br_disp_gt[:, pad_H[torch.argmax(pad_H)]:, pad_W[torch.argmax(pad_W)]:]
    disp_gt = disp_gt[:, pad_H[torch.argmax(pad_H)]:, pad_W[torch.argmax(pad_W)]:]

    # disp_ests = output[0, 0][None, :, :].clone().cpu().detach().numpy()
    # disp_ests = np.transpose(disp_ests * 255.0, (1, 2, 0))

    # 计算损失函数并将误差反向传播
    mask = (disp_gt < args.bi3dnet_max_disparity) & (disp_gt > 0)
    loss = loss_fn(disp_ests[mask], br_disp_gt[mask])
    scalar_outputs = {"loss": loss}
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)


if __name__ == '__main__':
    train()
