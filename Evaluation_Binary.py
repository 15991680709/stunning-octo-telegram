# -*-：coding:utf-8-*-
import argparse
import os
import time

from PIL import Image
from matplotlib import pyplot as plt

import model_change  # 这是我自己改进的模型
import numpy as np
from torch.utils.data import DataLoader
from util import str2bool
from utils import *
import gc
from tensorboardX import SummaryWriter
from datasets import __datasets__

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
parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/sf_binary_depth.pth.tar")
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/media/fhy/My_Passport/GA-Net_SceneFlowDataset/", help='data path')
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='training list')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')

# save
parser.add_argument('--logdir', default='log_tree_miou', help='the directory to save logs')
parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')

# parse arguments
args, unknown = parser.parse_known_args()
options = vars(args)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
# 初始化测试的数据集(执行的为SceneFlowDatset中的__init__部分)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network_data = torch.load(args.pretrained, map_location=device)  # 加载预训练模型
model = model_change.__dict__[args.arch](options, network_data).to(device)  # 没有预训练模型，模型参数随机初始化。

# create summary logger
output_dir = "{}".format(args.logdir)
os.makedirs(output_dir, exist_ok=True)
print("creating new summary file")
logger = SummaryWriter(output_dir)

# 该函数内部的分析见实验报告《mIou指标的具体调试过程》
# 计算置信度映射的miou指标的自定义函数。
def compute_miou(pred, target):
    # 计算置信度映射中被估计为输入深度之前的像素点和实际位于输入深度之前的像素点之间的重叠区域。
    intersection = pred * (pred == target)
    # 二元深度估计网络将场景内部的像素点划分为位于输入深度平面之前或者之后两类。
    # 在输入该函数时，无论是置信度映射还是二值视差图，位于输入深度之前的像素点被标记为1，反之标记为0。
    # 因此，控制分类范围的区间bins只要内部被划分为两段，一段的中心值为0，另一段为1即可。
    area_inter, _ = np.histogram(intersection, bins=[-0.5, 0.5, 1.5])
    area_pred, _ = np.histogram(pred, bins=[-0.5, 0.5, 1.5])
    area_target, _ = np.histogram(target, bins=[-0.5, 0.5, 1.5])
    area_union = area_pred + area_target - area_inter  # 置信度映射和二值视差图的并集。

    assert area_inter[1] <= area_union[1] and area_inter[0] >= area_union[
        0], "Intersection area should be smaller than union area"
    # 当前置信度映射和二值视差图之间的mIou值。
    if area_union[1] != 0.0 and area_inter[0] != 0.0:
        rate = round(((area_inter[1] / area_union[1]) + (area_union[0] / area_inter[0])) / 2.0, 4)
    elif area_union[1] != 0.0 and area_inter[0] == 0.0:
        rate = round((area_inter[1] / area_union[1]), 4)
    elif area_union[1] == 0.0 and area_inter[0] != 0.0:
        rate = round((area_union[0] / area_inter[0]), 4)
    else:
        print("异常：miou的两种分母同时为0")
        return 0.0
    return rate


def test_sample(sample):
    imgL = sample["left"]
    imgL = imgL.cuda()

    imgR = sample["right"]
    imgR = imgR.cuda()

    pad_H = sample["top_pad"]
    pad_H = pad_H.cuda()

    pad_W = sample["left_pad"]
    pad_W = pad_W.cuda()

    # 经过调试，br_disp_gt为一个包含5个元素list，每个list为相应的待分类视差下的二值视差图，维度为B*H*W。
    br_disp_gt = sample["br_disp_gt"]

    # disp_ref为一个包含5个元素的list，每个list为相应的待分类视差值除以三之后的数值转化为二维Tensor后的结果。
    disp_ref = sample["disp_ref"]
    # 由于list对象没有cuda属性，因此需要把disp_ref中的每个元素沿列表长度的方向进行拼接，得到维度是5*1*1*1大小的Tensor。
    # 其中，第二个维度代表了值为1的test_batch_size。
    disp_ref = torch.stack(disp_ref)
    disp_ref = disp_ref.cuda()

    # 对5组待分类的每组视差所对应的置信度映射进行miou指标计算。
    miou_batch = 0.0
    for i in range(0, len(br_disp_gt)):
        # 读取该待分类视差所对应的二值视差图，并将该视差图中多余的batch维去掉。
        # 最终读取到的二值视差图br_disp_gt_random为大小是H*W的ndarray。
        br_disp_gt_random = br_disp_gt[i].clone().detach().squeeze()
        br_disp_gt_random = br_disp_gt_random.cpu().numpy()

        # 通过模型得到该待分类视差所对应的置信度映射。
        model.eval()
        with torch.no_grad():
            # output为输出的置信度映射，该置信度映射的尺寸为B*psv_size*H*W。
            # 其中，由于test_batch_size=1，并且该网络为二元深度估计网络，因此B=psv_size=1。
            # 所以，输出的output的尺寸为1*1*H*W。
            output = model(imgL, imgR, disp_ref[i])[1][:, :, pad_H[0]:, pad_W[0]:]
        # 将output的前两个值为1的多余维度去掉，得到的disp_ests为output的深拷贝，并转化为numpy类型。
        disp_ests = output[0, 0].clone().cpu().detach().numpy()

        # 置信度映射值大于等于0.5的像素点位于该视差平面上或者位于该视差平面之前，为了便于miou计算，这些像素点的置信值设为1。
        # 同理：置信度映射值小于0.5的像素点位于该视差平面之后，这些像素点的置信值设为0。
        # 然后，将二值化后的置信度映射输入到miou函数中进行计算。
        disp_ests[disp_ests >= 0.5] = 1.0
        disp_ests[disp_ests < 0.5] = 0.0

        # 将转化为numpy类型的置信度映射和二值视差图输入到miou中进行计算，得到该批数据在每个待分类视差下的miou指标值。
        miou_disp = compute_miou(disp_ests, br_disp_gt_random)
        miou_batch += miou_disp

    # 该批数据的所有待分类视差下的miou均值。
    miou_batch = miou_batch / len(br_disp_gt)
    scalar_outputs = {"miou": miou_batch}
    return tensor2float(miou_batch), tensor2float(scalar_outputs)


def test():
    mean_miou = 0.0 # 对该模型参数进行miou指标评估在所有批数据上的均值。
    for batch_idx, sample in enumerate(TestImgLoader):
        torch.cuda.empty_cache()  # 每次训练之前将缓存清空，防止报cuda out of memery的错误。

        miou, scalar_outputs = test_sample(sample)
        start_time = time.time()
        do_summary = batch_idx % args.summary_freq == 0
        if do_summary:
            save_scalars(logger, 'miou-batch', scalar_outputs, batch_idx)
        del scalar_outputs
        print('Iter {}/{}, test miou = {:.3f}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), miou, time.time() - start_time))
        mean_miou += miou

    mean_miou = mean_miou / len(TestImgLoader)
    print("the whole miou is:", mean_miou)
    gc.collect()

    # 写完日志文件后关闭。
    logger.close()


if __name__ == '__main__':
    test()
