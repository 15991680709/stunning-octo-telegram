# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
import os
from locale import atoi

import torch.nn.parallel
import torch.optim as optim
# import models  # 这是作者已经编写好的模型
import model_change  # 这是我自己改进的模型
import time
from torch.utils.data import DataLoader
from util import str2bool
from utils import *
import gc
from torch.utils.tensorboard import SummaryWriter
from datasets import __datasets__

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 针对作者已经编写好的模型进行命名
# model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
# 针对我自己改进的模型进行命名
model_names = sorted(name for name in model_change.__dict__ if name.islower() and not name.startswith("__"))

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

# input
parser.add_argument("--pretrained", type=str,
                    default="/home/zhangkexin/Bi3D-master/src/bi3d_models/model_weights/sf_binary_depth.pth.tar")
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/home/zhangkexin/GA-Net_SceneFlowDataset/", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--lr', type=float, default=1e-5, help='base learning rate')
parser.add_argument('--lrepochs', default="20,32,40,48,56:2", type=str,
                    help='the epochs to decay lr: the downscale rate')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')

# save
parser.add_argument('--logdir', default='log_tree', help='the directory to save logs and checkpoints')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=2, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args, unknown = parser.parse_known_args()
options = vars(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# 如果将每次训练所创建的Summary类日志文件均保存到同一个文件夹下,则下一次训练过程中计算的损失函数曲线会覆盖掉本次训练的曲线.
# 于是,为了便于查看每次训练下的损失函数,学习率等变化,以每次开始训练的时间为log_tree文件夹下二级文件夹的名称,将每次训练的日志文件保存到该二级文件夹中.
# 创建保存训练模型的二级日志文件夹
Time = time.time()
output_dir = "{}/{}".format(args.logdir,str(Time))
os.makedirs(output_dir, exist_ok=True)
del Time

# create summary logger
print("creating new summary file")
logger = SummaryWriter(output_dir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
# 初始化训练的数据集(执行的为SceneFlowDatset中的__init__部分)
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network_data = torch.load(args.pretrained, map_location=device)  # 加载预训练模型
model = model_change.__dict__[args.arch](options, network_data).to(device)  # 没有预训练模型，模型参数随机初始化。

# if torch.cuda.device_count()>1:
#     model = torch.nn.DataParallel(model)

# 定义优化器。并且优化器中只传入需要优化的网络层参数。
optimizer = optim.Adam(model.refinenet.conv2_conv.parameters(), lr=args.lr, betas=(0.9, 0.999))

# 从第N代训练的模型的基础上继续往上训练，得将第N代训练模型的optimizer加载进来。
# optimizer.load_state_dict(network_data['optimizer'])
# 如果从第N代已经训练好的模型checkpoint_(N-1).pth上继续训练,为了记录损失函数曲线的方便,直接将训练轮数计数器从当前加载的第N代训练模型开始计数
# start_epoch = atoi(os.path.splitext(os.path.basename(args.pretrained))[0].split('_')[1]) + 1

start_epoch = 0  # 从原始模型的基础上开始训练，为了描绘损失函数曲线和学习率曲线的方便，训练代收从0开始计数。
print("start at epoch {}".format(start_epoch))

# 该函数中last_epoch参数表示上一次训练的过程中恰好断掉的那一代。
# 如果此次训练模型是在上一次已经训练好的第N代模型checkpoint_(N-1).pth的基础上开始继续训练，那么last_epoch应为N、N-1还是N+1？
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# 初始化交叉熵损失函数(最后将所有像素点的损失值求平均)
crit = torch.nn.BCELoss(reduction='mean')


def train():
    # 模型训练一共迭代epochs次，每一次都将误差反向传播，不断地让拟合视差图接近真实视差图。
    # range(start_epoch, start_epoch + args.epochs)相比于range(start_epoch, args.epochs)写法的优势在于：
    # 可以兼容训练的过程中在第N代停下了，从第N代的基础上继续训练n代，一直训练到第N+n代的情形。其中：N=0表示从原始模型开始训练。

    for epoch_idx in range(start_epoch, start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        torch.cuda.empty_cache()  # 每次训练之前将缓存清空，防止报cuda out of memery的错误。

        # 上一代训练完之后，调整之后的学习率作为这一代的学习率。如果是从头训练，则为初始学习率。
        # 如果是在第N代的基础上继续训练，则这里的optimizer.param_groups[0]["lr"]应该为上一次训练过程中，训练到第N代模型时，将数据集中的数据都计算完了之后经过函数scheduler.step()调整之后的结果。
        # Scalar_learning_rate = {"lr": optimizer.param_groups[0]["lr"]}
        # save_scalars(logger, 'learning_rate', Scalar_learning_rate, epoch_idx)
        # del Scalar_learning_rate

        # 在每一轮训练中,再记录一组该轮训练的平均损失函数
        Loss = 0.0

        # training
        # 每次抓取batch_size对立体图像对和相应的视差图进行训练。真正抓取图像的时候执行的是Bi3DNetBinaryDepth中的__getitem__部分。
        for batch_idx, sample in enumerate(TrainImgLoader):
            loss, scalar_outputs = train_sample(sample, crit)
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            if do_summary:
                save_scalars(logger, 'loss-batch', scalar_outputs, global_step)
            del scalar_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))

            Loss += loss  # 将这一代中,每个batch训练的损失函数累加

        # scheduler.step()  # 这一代的模型训练完之后，对学习率进行调整。

        # 这一代的数据集中的所有数据训练完之后,将这一代所有数据集中每一批数据计算的损失函数值取一个平均,并在终端进行打印.
        Loss = Loss / len(TrainImgLoader)
        print('Epoch {}/{}, train loss = {:.3f}'.format(epoch_idx, args.epochs, Loss))

        # 再记录一组损失函数随训练代数的变化曲线
        Scalar_average_loss = {"Loss": Loss}
        save_scalars(logger, 'loss-epoch', Scalar_average_loss, epoch_idx)
        del Scalar_average_loss

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'state_dict': model.state_dict(),
                               'optimizer': optimizer.state_dict()}
            # 如果从我训练的某一代模型开始训练,则需要修改保存模型时模型的命名方式.
            # 比如:我之前已经训练好了5代,每一代的模型序号分别为0到4.
            # 则当我把序号为4的第5代模型当作预训练模型加载进来之后,从Epoch=0开始训练的模型的序号应该从5开始.
            # 即:将原来训练的第5代模型当作预训练模型开始训练之后每一代的训练模型分别对应于原来训练的第6代及其之后的模型.因此,需要将这些模型的序号改为从5开始.
            # 即:Epoch=0对应的模型序号为5,Epoch=1对应的模型序号为6,..以此类推.
            # if 'checkpoint' in os.path.splitext(os.path.basename(args.pretrained))[0]:
            #     torch.save(checkpoint_data,'{}/{}_{:0>6}{}'.format(args.logdir, os.path.splitext(os.path.basename(args.pretrained))[0].split('_')[0],
            #                       atoi(os.path.splitext(os.path.basename(args.pretrained))[0].split('_')[1]) + epoch_idx + 1,
            #                       os.path.splitext(os.path.basename(args.pretrained))[1]))
            # else:
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.pth".format(args.logdir, epoch_idx))
        gc.collect()
    # 写完日志文件后关闭。
    logger.close()


# train one sample
def train_sample(sample, loss_fn):
    model.train()  # 模型处于训练模式，有误差的反向传播

    # 先将批图像读进来
    imgL_pad, imgR_pad, pad_H, pad_W, disp_ref, br_disp_gt, disp_gt = \
        sample["left"], sample["right"], sample["top_pad"], sample["left_pad"], sample["dis_ref"], sample[
            "binary_disparity"], sample["disparity"]
    optimizer.zero_grad()

    # input convert to cuda format
    imgL_pad = imgL_pad.cuda()
    imgR_pad = imgR_pad.cuda()
    pad_H = pad_H.cuda()
    pad_W = pad_W.cuda()
    disp_ref = disp_ref.cuda()
    br_disp_gt = br_disp_gt.cuda()
    disp_gt = disp_gt.cuda()

    # Inference
    output = model(imgL_pad, imgR_pad, disp_ref)[1][:, :, pad_H[0]:, pad_W[0]:]
    # 将output的psv维度和batch维度互换，并取出维度为B*H*W的置信度disp_ests。
    # 其中，取出的经网络估计的置信度的大小为540*960，把填充的部分通过[:, :, pad_H[0]:, pad_W[0]:]截取掉了。
    disp_ests = output.clone().detach().transpose(0, 1)[0]

    # 计算损失函数并将误差反向传播
    mask = (disp_gt < args.bi3dnet_max_disparity) & (disp_gt > 0)
    loss = loss_fn(disp_ests[mask], br_disp_gt[mask])
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()
    scalar_outputs = {"loss": loss}

    return tensor2float(loss), tensor2float(scalar_outputs)

if __name__ == '__main__':
    train()
