# -*-：coding:utf-8-*-
import argparse
import os
import time

import numpy as np
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

# Input Datasets Kitti2015
# parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/kitti15_continuous_depth_no_conf_reg.pth.tar")
# parser.add_argument('--dataset_c', default='kitti_c', help='dataset name', choices=__datasets_c__.keys())
# parser.add_argument('--kitti15_datapath', default='/media/fhy/My_Passport/data_scene_flow/', help='data path')
# parser.add_argument('--testlist', default='./filenames/kitti15_train.txt', help='testing list')

# Input SceneFlowDatasets
parser.add_argument("--pretrained", type=str,default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/sf_continuous_depth_no_conf_reg.pth.tar")
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
# test_dataset = StereoDataset(args.kitti15_datapath, args.testlist, False)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=True, num_workers=0, drop_last=True)

# model initial
device = torch.device('cuda')
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

    # 连续深度估计网络的输出为：A.由输入视差序列所构成的大小为的置信体映射。B.视差值范围为的连续视差图像。
    # 而在对连续深度估计网络进行评估的过程中，只需要对网络输出的连续视差图进行评估，置信体不参与评估计算。
    # 因此，当执行到网络输出的置信体映射时，为了节省cuda显存，需要删除输出的置信体映射，在对输出的连续视差图进行EPE或者Dl指标的评估。
    # 相应地，对于测试集，在数据预处理getitem中无需从视差序列中任意选择一组待分类视差并计算相应的二值视差图，最终的返回也没有待分类视差值和相应的二值视差图。
    # 除此之外，数据其余的预处理流程与训练时候相同。
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

    # 将真值视差图中视差值介于0到192的像素点筛选出来，和视差范围介于-0.5到192.5的真值视差图进行Dl指标的计算。
    mask = (disp_gt_EPE > d_min_GT) & (disp_gt_EPE < d_max_GT)
    abolute_disp_error = np.abs(disp_est[mask] - disp_gt_EPE[mask])
    array_3 = np.full(disp_gt_EPE[mask].shape, 3, dtype='float32')
    disp_dl_big = np.array([0.05 * disp_gt_EPE[mask], array_3])
    disp_dl = np.max(disp_dl_big, axis=0)
    outline_error = np.mean(abolute_disp_error > disp_dl)
    del disp_dl, disp_dl_big, array_3, abolute_disp_error
    return outline_error


def test():
    # 根据网上使用matplotlib画图的有关教程：对于自变量和因变量分别生成一组列表(总之一定是可以迭代的序列)，然后将自变量和因变量列表中对应的自变量和因变量的取值所组成的数对在图像中描点。
    # 每次计算完一批数据的Dl值以后，将该Dl值追加在该列表的尾部。EPE指标值同理。
    # 相应地，在Dl_list和EPE_list的尾部每追加一次计算出的指标值时，在存放自变量(本程序中为批图像的序号batch_idx)的列表batch_list中也要存放相应批数据的索引，保证批数据和相应的指标在各自的列表中要一一对应。
    mean_Dl = 0.0
    Dl_list = []
    batch_list = []
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.empty_cache()  # 每次训练之前将缓存清空，防止报cuda out of memery的错误。

        Dl = test_sample(sample)

        if np.isnan(Dl) == False and 0.0 < Dl < 0.1:
            batch_list.append(batch_idx)
            Dl_list.append(Dl)
            start_time = time.time()
            print('Iter {}/{}, test Dl = {:.3f}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), Dl,
                                                                        time.time() - start_time))
            mean_Dl += Dl
        else:
            del Dl

    # for循环执行完后，将每批数据输出的EPE指标和相应的批序号使用matplotlib画图工具画出来。
    # 为了让曲线图看起来美观，对输出的EPE进行排序
    Dl_list.sort()
    plt.plot(batch_list, Dl_list, 'ro-', color='orange')
    plt.savefig("./Continuous_SceneFlow_logtree/Dl.png")

    mean_Dl = mean_Dl / len(batch_list)
    print("the whole Dl is:", mean_Dl)

    gc.collect()

if __name__ == '__main__':
    test()
