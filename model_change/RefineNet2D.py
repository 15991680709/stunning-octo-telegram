# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import time
import torch.backends.cudnn as cudnn

from model_change.PSMNet import conv2d
from model_change.PSMNet import conv2d_lrelu

from model_change.DispRefine2D import DispRefineNet

__all__ = ["disprefinenet", "segrefinenet"]


"""
Disparity refinement network.
Takes concatenated input image and the disparity map to generate refined disparity map.
Generates refined output using input image as guide.
"""


def disprefinenet(options, data=None):

    # print("==> USING DispRefineNet")
    # for key in options:
    #     if "disprefinenet" in key:
    #         print("{} : {}".format(key, options[key]))

    model = DispRefineNet(out_planes=options["disprefinenet_out_planes"])

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model


"""
Binary segmentation refinement network.
Takes as input high resolution features of input image and the disparity map.
Generates refined output using input image as guide.
"""


class SegRefineNet(nn.Module):
    def __init__(self, in_planes=17, out_planes=8):

        super(SegRefineNet, self).__init__()

        self.conv1 = nn.Sequential(conv2d_lrelu(in_planes, out_planes, kernel_size=3, stride=1, pad=1))

        self.classif1 = nn.Conv2d(out_planes, 1, kernel_size=3, padding=1, stride=1, bias=False)

        # 除了优化网络中新添加的卷积层需要参数反向传播以外，其余网络层的参数可以不变。
        for p in self.parameters():
            p.requires_grad = False

        # self.conv2 = nn.Sequential(conv2d_lrelu(1, 1, kernel_size=3, stride=1, pad=1))

        ################   对conv2的输出进行数字统计   ################
        # 将conv2卷积层拆成两部分：
        # a.使用大小为3*3的权重矩阵和偏置项的卷积核对output进行卷积。
        # b.将卷积之后的结果使用LeakyReLU函数进行线性激活。
        self.conv2_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)
        self.activate = nn.LeakyReLU(0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, input):
        output0 = self.conv1(input)
        output = self.classif1(output0)

        # output_copy = output[0, 0].clone().detach().cpu().numpy()
        # print("the output is:\n",output_copy)
        # print("the max value of the output is:", np.max(output_copy))
        # print("the min value of the output is:", np.min(output_copy))
        # print("the mean value of the output is:", np.mean(output_copy))
        # print("the var value of the output is:", np.var(output_copy))
        # del output_copy

        # 对原来的Bi3D网络输出的output和增加了conv2之后的Bi3D网络输出的output_conv2进行数学特性的统计，包括均值，方差等。
        output_conv2_conv = self.conv2_conv(output)

        # output_conv2_conv_copy = output_conv2_conv[0, 0].clone().detach().cpu().numpy()
        # print("the output_conv2_conv is:\n", output_conv2_conv_copy)
        # print("the max value of the output_conv2_conv is:", np.max(output_conv2_conv_copy))
        # print("the min value of the output_conv2_conv is:", np.min(output_conv2_conv_copy))
        # print("the mean value of the output_conv2_conv is:", np.mean(output_conv2_conv_copy))
        # print("the var value of the output_conv2_conv is:", np.var(output_conv2_conv_copy))
        # del output_conv2_conv_copy

        output_conv2 = self.activate(output_conv2_conv)
        #
        # output_conv2_copy = output_conv2[0, 0].clone().detach().cpu().numpy()
        # print("the output_conv2_conv is:\n", output_conv2_copy)
        # print("the max value of the output_conv2 is:", np.max(output_conv2_copy))
        # print("the min value of the output_conv2 is:", np.min(output_conv2_copy))
        # print("the mean value of the output_conv2 is:", np.mean(output_conv2_copy))
        # print("the var value of the output_conv2 is:", np.var(output_conv2_copy))
        # del output_conv2_copy

        return output, output_conv2

def segrefinenet(options, data=None):

    # print("==> USING SegRefineNet")
    # for key in options:
    #     if "segrefinenet" in key:
    #         print("{} : {}".format(key, options[key]))

    model = SegRefineNet(
        in_planes=options["segrefinenet_in_planes"], out_planes=options["segrefinenet_out_planes"]
    )

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model
