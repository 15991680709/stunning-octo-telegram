# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import models
import cv2
import numpy as np
from util import disp2rgb, str2bool

from datasets_c.data_io_c import read_all_lines

import random

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))


# Parse Arguments
parser = argparse.ArgumentParser(allow_abbrev=False)

# Experiment Type
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

# kitti15
# parser.add_argument("--pretrained", type=str, default="/home/zhangkexin/Bi3D-master/bi3d_models/model_weights/kitti15_continuous_depth_no_conf_reg.pth.tar")
parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/Continuous_Kitti15_log_tree/checkpoint_000015.pth")
parser.add_argument('--testlist',default='./filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--datapath', default="/media/fhy/My_Passport/data_scene_flow/", help='data path')

# sceneflow
# parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/sf_continuous_depth_no_conf_reg.pth.tar")
# parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
# parser.add_argument('--datapath', default="/media/fhy/My_Passport/GA-Net_SceneFlowDataset/", help='data path')

parser.add_argument("--disp_range_min", type=int, default=0)
parser.add_argument("--disp_range_max", type=int, default=192)
parser.add_argument("--crop_height", type=int,default=384)
parser.add_argument("--crop_width", type=int,default=1248)

args, unknown = parser.parse_known_args()

##############################################################################################################
def main():

    options = vars(args)
    # print("==> ALL PARAMETERS")
    # for key in options:
    #     print("{} : {}".format(key, options[key]))

    # 创建存储连续深度估计结果的文件夹
    out_dir = "out"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

        # 模仿Datasets数据结构中的load_path函数，将所有图片的路径均读进来。
    lines = read_all_lines(args.testlist)
    splits = [line.split() for line in lines]
    left_images = [x[0] for x in splits]  # 数据集中所有左图像的路径
    right_images = [x[1] for x in splits]  # 数据集中所有右图像的路径
    # 在数据集中任选一对左右立体图像做可视化evaluation。
    index = random.randrange(0, len(lines), 1)
    imgleft = os.path.join(args.datapath, left_images[index])
    imgright = os.path.join(args.datapath, right_images[index])

    # 保存做可视化evaluation的立体图像对的名称(在数据集中是哪对立体图像对)
    # kitti Datasets
    image_name = "%s" % (os.path.splitext(splits[index][0].split('/')[-1])[0])
    # sceneFlow Datasets
    # image_name = "%s-%s-%s" % (splits[index][0].split('/')[1], splits[index][0].split('/')[4], os.path.splitext(splits[index][0].split('/')[-1])[0])

    # 为了查看训练的效果如何，需要在保存输出结果的过程中保存evaluation模型的名字。
    model_name = os.path.splitext(os.path.basename(args.pretrained))[0]

    # Model
    if args.pretrained:
        # 加载预训练模型的参数
        device = torch.device('cuda')
        network_data = torch.load(args.pretrained,map_location=device)
    else:
        print("Need an input model")
        exit()

    print("=> using pre-trained model '{}'".format(args.arch))
    # 使用预训练模型的参数将模型初始化
    model = models.__dict__[args.arch](options, network_data).cuda()

    # Inputs
    # 将左、右立体图像对分别以Image变量的形式读取进来，转换为Tensor类型的张量以后分别在每个维度上以0.5的均值和方差做归一化计算。
    # 然后，在Tensor类型的立体图像对的第0维增加批处理维度，在转换为cuda上的Tensor。
    img_left = Image.open(imgleft).convert("RGB")
    img_right = Image.open(imgright).convert("RGB")
    img_left = transforms.functional.to_tensor(img_left)
    img_right = transforms.functional.to_tensor(img_right)
    img_left = transforms.functional.normalize(img_left, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_right = transforms.functional.normalize(img_right, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_left = img_left.type(torch.cuda.FloatTensor)[None, :, :, :]
    img_right = img_right.type(torch.cuda.FloatTensor)[None, :, :, :]

    # Prepare Disparities
    max_disparity = args.disp_range_max  # 最大视差值对应最近的深度
    min_disparity = args.disp_range_min  # 最小视差值对应最远的深度

    assert max_disparity % 3 == 0 and min_disparity % 3 == 0, "disparities should be divisible by 3"

    if args.arch == "bi3dnet_continuous_depth_3D":
        assert (
            max_disparity - min_disparity
        ) % 48 == 0, "for 3D regularization the difference in disparities should be divisible by 48"

    # 最近深度平面和最远深度平面之间一共存在多少组深度平面
    max_disp_levels = (max_disparity - min_disparity) + 1

    # 计算匹配代价体的时候使用了左右立体图像进行三倍降采样之后的左右特征映射的结果，因此传入网络中计算的视差上下限相应地变为原来的三分之一。
    max_disparity_3x = int(max_disparity / 3)   # 传入到连续深度估计的网络中进行计算的视差值上限
    min_disparity_3x = int(min_disparity / 3)   # 传入到连续深度估计的网络中进行计算的视差值下限
    max_disp_levels_3x = (max_disparity_3x - min_disparity_3x) + 1  # 传入到深度估计网络中计算的视差平面的数量。
    # disp_3x为视差序列[0,1,...,64]
    disp_3x = np.linspace(min_disparity_3x, max_disparity_3x, max_disp_levels_3x, dtype=np.int32)
    # 将视差序列转换为可以运行在cuda上的Tensor类型。
    disp_long_3x_main = torch.from_numpy(disp_3x).type(torch.LongTensor).cuda()
    disp_float_main = np.linspace(min_disparity, max_disparity, max_disp_levels, dtype=np.float32)
    disp_float_main = torch.from_numpy(disp_float_main).type(torch.float32).cuda()
    delta = 1
    d_min_GT = min_disparity - 0.5 * delta  # 视差图中所有像素点对应视差值的下限
    d_max_GT = max_disparity + 0.5 * delta  # 视差图中所有像素点对应视差值的上限
    # 将转换为Tensor的视差序列扩展一个批处理的维度，将这二维的视差序列输入到2D连续深度估计网络中。
    disp_long_3x = disp_long_3x_main[None, :].expand(img_left.shape[0], -1)
    disp_float = disp_float_main[None, :].expand(img_left.shape[0], -1)

    # Pad Inputs
    # 针对立体图像对进行裁剪和填充
    tw = args.crop_width
    th = args.crop_height
    assert tw % 96 == 0, "image dimensions should be multiple of 96"
    assert th % 96 == 0, "image dimensions should be multiple of 96"
    h = img_left.shape[2]
    w = img_left.shape[3]
    x1 = random.randint(0, max(0, w - tw))
    y1 = random.randint(0, max(0, h - th))
    pad_w = tw - w if tw - w > 0 else 0
    pad_h = th - h if th - h > 0 else 0
    pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
    img_left = img_left[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]
    img_right = img_right[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]
    img_left_pad = pad_opr(img_left)
    img_right_pad = pad_opr(img_right)

    # Inference
    model.eval()  # 模型评估
    with torch.no_grad():
        # 2D连续深度估计网络输出连续估计的视差图与视差序列[0,1,...,64]中每一个参考视差所对应的置信度映射
        if args.arch == "bi3dnet_continuous_depth_2D":
            output_seg_low_res_upsample, output_disp_normalized = model(
                img_left_pad, img_right_pad, disp_long_3x
            )
            output_seg = output_seg_low_res_upsample
        else:
            (
                output_seg_low_res_upsample,
                output_seg_low_res_upsample_refined,
                output_disp_normalized_no_reg,
                output_disp_normalized,
            ) = model(img_left_pad, img_right_pad, disp_long_3x)
            output_seg = output_seg_low_res_upsample_refined

        # 把当初填充进来的部分裁剪掉。
        output_seg = output_seg[:, :, pad_h:, pad_w:]
        output_disp_normalized = output_disp_normalized[:, :, pad_h:, pad_w:]
        
            # visiualization test
            # index = random.randint(0,192)
            # output_seg_com = output_seg[0,index].clone().detach().cpu().numpy()
            # output_seg_com = output_seg_com * 255
            # output_seg_com = Image.fromarray(output_seg_com).convert('L')
            # output_seg_com.save("/media/fhy/My_Passport/ZKX/Bi3D-master/src/train_debug/continuous_confidence.png")
            # plt.imshow(output_seg_com)
            # plt.show()
            # plt.close()
            # output_disp_normalized_com = output_disp_normalized[0,0].clone().detach().cpu().numpy()
            # output_disp_normalized_com = output_disp_normalized_com * delta * max_disp_levels + d_min_GT
            # output_disp_normalized_com = Image.fromarray(output_disp_normalized_com).convert('L')
            # output_disp_normalized_com.save("/media/fhy/My_Passport/ZKX/Bi3D-master/src/train_debug/continuous_nor_disp.png")
            # plt.imshow(output_disp_normalized_com)
            # plt.show()
            # plt.close()

        # 将输出视差图上每个像素点的视差值的范围限制到-0.5和192.5之间。
        output_disp = torch.clamp(
            output_disp_normalized * delta * max_disp_levels + d_min_GT, min=d_min_GT, max=d_max_GT
        )

    # Write Results
    max_disparity_color = 192
    output_disp_clamp = output_disp[0, 0, :, :].cpu().clone().numpy()
    # 大于视差值上限的像素点的视差值限制为视差值上限，小于视差值下限的像素点的视差值限制为视差值下限。
    output_disp_clamp[output_disp_clamp < min_disparity] = 0
    output_disp_clamp[output_disp_clamp > max_disparity] = max_disparity_color
    # 视差图转换为彩色图像。
    disp_np_ours_color = disp2rgb(output_disp_clamp / max_disparity_color) * 255.0
    cv2.imwrite(
        os.path.join(out_dir, "%s_%s_%s_%d_%d.png" % (model_name, image_name, args.arch, min_disparity, max_disparity)),
        disp_np_ours_color,
    )

    return


if __name__ == "__main__":
    main()
