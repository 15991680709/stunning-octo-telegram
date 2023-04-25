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
from tensorboardX import SummaryWriter
import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets_c import __datasets_c__
from util import str2bool
import gc

from utils import tensor2float, adjust_learning_rate, save_scalars

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))

# Parse arguments
parser = argparse.ArgumentParser(allow_abbrev=False)

# Model
parser.add_argument("--arch", type=str, default="bi3dnet_continuous_depth_2D")
parser.add_argument("--bi3dnet_featnet_arch", type=str, default="featextractnetspp")
parser.add_argument("--bi3dnet_segnet_arch", type=str, default="segnet2d")
parser.add_argument("--bi3dnet_refinenet_arch", type=str, default="disprefinenet")
parser.add_argument("--bi3dnet_regnet_arch", type=str, default="segregnet3d")
parser.add_argument("--bi3dnet_max_disparity", type=int, default=192)
parser.add_argument("--regnet_out_planes", type=int, default=16)
parser.add_argument("--disprefinenet_out_planes", type=int, default=32)
parser.add_argument("--bi3dnet_disps_per_example_true", type=str2bool, default=True)

# Input
parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/kitti15_continuous_depth_no_conf_reg.pth.tar")
parser.add_argument('--dataset_c', default='kitti_c', help='dataset name', choices=__datasets_c__.keys())
parser.add_argument('--kitti15_datapath', default='/media/fhy/My_Passport/data_scene_flow/', help='data path')
parser.add_argument('--trainlist', default='./filenames/kitti15_train.txt', help='training list')

parser.add_argument("--disp_range_min", type=int, default=0)
parser.add_argument("--disp_range_max", type=int, default=192)

parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
parser.add_argument('--lrepochs',default="20,32,40,48,56:2", type=str,  help='the epochs to decay lr: the downscale rate')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')

# save
parser.add_argument('--logdir',default='log_tree', help='the directory to save logs and checkpoints')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=4, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args, unknown = parser.parse_known_args()
options = vars(args)

# 设置随机种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# 创建保存训练模型的日志文件夹
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets_c__[args.dataset_c]
# 初始化训练的数据集(执行的为SceneFlowDatset中的__init__部分)
train_dataset = StereoDataset(args.kitti15_datapath, args.trainlist, True)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)

# 模型初始化
device = torch.device('cuda')
# 将预训练模型的参数加载到GPU上。
network_data = torch.load(args.pretrained,map_location = device)
# 初始化训练模型
# (先执行Bi3DNetContinuousDepth2D中的__init__部分初始化模型结构，再根据每一个网络层对应的名称将预训练模型的参数加载到初始化的模型中)
print("=> using pre-trained model '{}'".format(args.arch))
model = models.__dict__[args.arch](options, network_data).to(device)

# 初始化Adam优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# 损失函数初始化
BCE_crit=torch.nn.BCELoss(reduction='mean')  # 二元交叉熵损失
L1_crit=nn.SmoothL1Loss(reduction='mean')  # SmoothL1损失

start_epoch = 0  # 初始迭代参数
print("start at epoch {}".format(start_epoch))

def train():
    # 模型训练一共迭代epochs次，每一次都将误差反向传播，不断地让拟合视差图接近真实视差图。
    for epoch_idx in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()  # 每次训练之前将缓存清空，防止报cuda out of memery的错误。
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)  # 训练到一定次数时，需要调整学习率。
        # 每次从数据集中读取一批图像进行训练。
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss,scalar_outputs = train_sample(sample, BCE_crit, L1_crit)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader),
                                                                                       loss,
                                                                                       time.time() - start_time))

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.pth".format(args.logdir, epoch_idx))
        gc.collect()

def train_sample(sample, BCE_fn, L1_fn):
    model.train()  # 模型处于训练模式，有误差的反向传播
    optimizer.zero_grad()  # 每次调用backward()函数之前都要将梯度清零，节省内存。不然这次的梯度会将上次的梯度累加。

    # 读取预处理过的批图像。
    imgL_pad, imgR_pad, pad_H, pad_W, disp_list, disp_ref, disp_gt, br_disp_gt \
        = sample["left"], sample["right"], sample["top_pad"], sample["left_pad"], sample["disparity_list"], \
          sample["dis_ref"], sample["disparity"], sample["binary_disparity"]

    # 将所有的输入转化为cuda类型
    imgL_pad = imgL_pad.cuda()
    imgR_pad = imgR_pad.cuda()
    pad_H = pad_H.cuda()
    pad_W = pad_W.cuda()
    disp_list = disp_list.cuda()
    # disp_ref为该批图像中每组图像随即选择的参考视差所组成的一维Tensor
    # 即：disp_ref[i](i=0,1,...,batch_size-1)为该批图像中索引为i的图像组所选择的参考视差。
    disp_ref = disp_ref.cuda()
    disp_gt = disp_gt.cuda()
    br_disp_gt = br_disp_gt.cuda()
    # 因此，大小是B*H*W的批二值视差中的每一个大小是H*W的二值视差图为批图像中相应图像组在该图像组随机选择的参考视差下计算出的二值视差图。
    # 每组图像输入到二元交叉熵的Tensor为该组图像对应的二值视差图和该组图像在相应视差值下的置信度映射。

    # 在将视差序列传入到模型进行视差计算之前，视差序列中的元素值必须为整型。
    # 不然在计算匹配代价体那一步中会报错：视差序列中的元素由于不是整型所以不能转化为索引。
    # 使用连续深度估计的模型计算置信体映射和连续估计的视差图，并把当初填充的区域裁剪掉。
    confidence_v, disp_est = model(imgL_pad, imgR_pad, disp_list)
    # 模型输出的置信体映射的大小为B*D*H*W。批图像中下标为i的图像组在相应的参考视差disp_ref[i]下的置信体映射的二维下标为[i,disp_ref[i]]。
    confidence_v = confidence_v[:, :, pad_H[torch.argmax(pad_H)]:, pad_W[torch.argmax(pad_W)]:]
    disp_est = disp_est[:, :, pad_H[torch.argmax(pad_H)]:, pad_W[torch.argmax(pad_W)]:]

    # 将归一化之后的视差图中每个像素点的视差范围限制到-0.5和192.5之间。
    delta = 1
    d_min_GT = args.disp_range_min - 0.5 * delta  # 视差图中所有像素点对应视差值的下限
    d_max_GT = args.disp_range_max + 0.5 * delta  # 视差图中所有像素点对应视差值的上限
    max_disp_levels = (args.disp_range_max - args.disp_range_min) + 1
    disp_est = torch.clamp(
        disp_est * delta * max_disp_levels + d_min_GT, min=d_min_GT, max=d_max_GT
    )

    # disp_est为维度是B*D*H*W的Tensor张量。
    # 其中，该批连续深度估计视差结果的每一组视差图中每一个像素点的视差为经过了AUC层的计算结果。因此，D=1。
    # 将disp_est中维数为1的视差维删除掉，使得连续深度估计视差图张量disp_est与真值视差图张量的维度相同，传入到smoothL1损失函数中计算。
    disp_est=disp_est.squeeze(1)

    # 以连续估计的结果和真值视差图为输入，计算SmoothL1损失。
    # 将真值视差图中视差值介于0到192之间的像素点筛选出来，使用这些点参与运算。、
    mask_L1 = (disp_gt > args.disp_range_min) & (disp_gt < args.disp_range_max)
    L1_loss = L1_fn(disp_est[mask_L1], disp_gt[mask_L1])
    L1_loss.requires_grad_(True)  # 启动反向传播

    # 每组图像输入到二元交叉熵的Tensor为该组图像对应的二值视差图br_disp_gt[i]和该组图像在相应视差值下的置信度映射confidence_v[i,disp_ref[i]]。
    BCE_loss = 0.0
    for i in range(0, args.batch_size):
        mask_bce = (disp_gt[i] > args.disp_range_min) & (disp_gt[i] < args.disp_range_max)
        bce_loss = BCE_fn(confidence_v[i, disp_ref[i]][mask_bce], br_disp_gt[i][mask_bce])
        bce_loss.requires_grad_(True)
        BCE_loss += bce_loss
    BCE_loss = BCE_loss / args.batch_size

    # 连续深度估计网络中的总损失为二元交叉熵损失和SmoothL1损失分别以0.1和0.9为权重的加权求和。
    loss = 0.1 * BCE_loss + 0.9 * L1_loss
    loss.backward()
    optimizer.step()
    scalar_outputs = {"loss": loss, "BCE_loss":BCE_loss, "L1_loss":L1_loss}

    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    train()
