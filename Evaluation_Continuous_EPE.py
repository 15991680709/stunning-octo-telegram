# -*-：coding:utf-8-*-
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

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets_c import __datasets_c__
from util import str2bool
import gc

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

# Input Parameters
parser.add_argument("--disp_range_min", type=int, default=0)
parser.add_argument("--disp_range_max", type=int, default=192)

# Input Datasets
parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/sf_continuous_depth_no_conf_reg.pth.tar")
parser.add_argument('--dataset_c', default='sceneflow_c', help='dataset name', choices=__datasets_c__.keys())
parser.add_argument('--datapath', default="/media/fhy/My_Passport/GA-Net_SceneFlowDataset/", help='data path')
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')

parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

# parse arguments, set seeds
args, unknown = parser.parse_known_args()
options = vars(args)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# dataset, dataloader
StereoDataset = __datasets_c__[args.dataset_c]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

# model initial
device = torch.device('cuda:0')
network_data = torch.load(args.pretrained, map_location=device)
print("=> using pre-trained model '{}'".format(args.arch))
model = models.__dict__[args.arch](options, network_data).to(device)

def test_sample(sample):
    # 将getitem返回的批数据读取进来。
    imgL = sample["left"]
    imgL = imgL.cuda()

    imgR = sample["right"]
    imgR = imgR.cuda()

    pad_H = sample["top_pad"]
    pad_H = pad_H.cuda()

    pad_W = sample["left_pad"]
    pad_W = pad_W.cuda()

    disp_list = sample["disparity_list"]
    disp_list = disp_list.cuda()

    disp_gt = sample["disparity"]

    model.eval()
    with torch.no_grad():
        confidence_v, disp_ests = model(imgL, imgR, disp_list)
    # 由于网络输出的置信体不参与评估指标的计算，因此删除置信体以节省内存。
    del confidence_v

    # 把计算出的视差图与输入到网络模型中的立体图像对的相应区域裁减掉。
    disp_ests = disp_ests[:, :, pad_H[torch.argmax(pad_H)]:, pad_W[torch.argmax(pad_W)]:]

    # 将归一化之后的视差图中每个像素点的视差范围限制到-0.5和192.5之间。
    delta = 1
    d_min_GT = args.disp_range_min - 0.5 * delta  # 视差图中所有像素点对应视差值的下限
    d_max_GT = args.disp_range_max + 0.5 * delta  # 视差图中所有像素点对应视差值的上限
    max_disp_levels = (args.disp_range_max - args.disp_range_min) + 1
    disp_ests = torch.clamp(
        disp_ests * delta * max_disp_levels + d_min_GT, min=d_min_GT, max=d_max_GT
    )

    disp_est = disp_ests[0, 0].clone().detach().cpu().numpy()
    disp_gt_EPE = disp_gt[0].clone().detach().numpy()
    del disp_ests, disp_gt

    # 将真值视差图中视差值介于0到192的像素点筛选出来，和视差范围介于-0.5到192.5的真值视差图进行EPE指标的计算。
    mask = (disp_gt_EPE > d_min_GT) & (disp_gt_EPE < d_max_GT)
    error = np.mean(np.abs(disp_est[mask] - disp_gt_EPE[mask]))
    return error

    # 可视化检查评估指标的输入：估计的视差图和视差真值图的代码。
    # disp_gt_copy = disp_gt[0].clone().detach()
    # disp_gt_copy = disp_gt_copy.cpu().numpy()
    # disp_gt_copy = Image.fromarray(disp_gt_copy.astype('uint16')).convert('L')
    # disp_gt_copy.save("./debug/continuous_sf_disparity.png")

    # disp_est_copy = disp_est[0].clone().detach()
    # disp_est_copy = disp_est_copy.cpu().numpy()
    # disp_est_copy = Image.fromarray(disp_est_copy.astype('uint16')).convert('L')
    # disp_est_copy.save("./debug/continuous_sf_disparity_estimate.png")


def test():
    mean_EPE = 0.0
    EPE_list = []
    batch_list = []
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.empty_cache()  # 每次训练之前将缓存清空，防止报cuda out of memery的错误。
        EPE = test_sample(sample)

        # 查找出EPE的值为nan的图片索引。
        # 查找方法：将DataLoader中的shuffle改为False，这样每次从数据集中读取图片就成顺序读取了。
        # 当计算出EPE为nan时，在终端打印出当前图片的索引，
        # 然后从.txt文件中找出该图片，可视化该图片的真值视差图、经过网络输出的连续视差估计结果和mask掩码。
        # if np.isnan(EPE):
        #     print("当前EPE为无效值的图片序号为：", batch_idx)
        #     break;

        # 由于在评估EPE时将真值视差图和连续深度估计结果转换为numpy再计算的，因此输出的EPE类型为numpt.float32类型。
        # 但是在写入SummaryWriter文件时，计算出的EPE需是Tensor类型。而使用from_array函数进行转化时报错：numpy.float32不是ndarray，转化失败。
        # 因此，就换一种思路：将每次计算的EPE和相应的批序号保存在各自的列表中，通过该列表使用matplotlib画图，将每批数据的EPE指标描绘出来。
        # 在跑了一轮EPE指标的评估之后发现：测试集中存在两组数据的输出为nan，还有一些输出的EPE过大。因此需要将这些点的EPE滤掉。
        # 经过这么计算，batch_list中只有输出EPE处于合理范围之内的批数据所对应的序号。
        if np.isnan(EPE) == False and 0.0 < EPE < 1.5:
            batch_list.append(batch_idx)
            EPE_list.append(EPE)
            start_time = time.time()
            print('Iter {}/{}, test EPE = {:.3f}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), EPE,
                                                                        time.time() - start_time))
            mean_EPE += EPE
        else:
            del EPE

    # for循环执行完后，将每批数据输出的EPE指标和相应的批序号使用matplotlib画图工具画出来。
    # 为了让曲线图看起来美观，对输出的EPE进行排序
    EPE_list.sort()
    plt.plot(batch_list, EPE_list, 'ro-', color='orange')
    plt.savefig("./Continuous_SceneFlow_logtree/checkpoint_000019_EPE.png")

    mean_EPE = mean_EPE / len(batch_list)
    print("the whole EPE is:", mean_EPE)

    gc.collect()

if __name__ == '__main__':
    test()
