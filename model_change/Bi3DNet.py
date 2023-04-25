# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from einops.einops import rearrange

import model_change.FeatExtractNet as FeatNet
import model_change.SegNet2D as SegNet
import model_change.RefineNet2D as RefineNet
import model_change.RefineNet3D as RefineNet3D

# 对二分类网络和连续深度估计网络增添位置编码和attention计算网络层。
from .pos_attention.position_encoding import PositionEncodingSine  # 引入位置编码网络层的构造函数
from .pos_attention.transformer import LocalFeatureTransformer  # 引入有关attention计算的transformer网络架构的构造函数。

__all__ = ["bi3dnet_binary_depth", "bi3dnet_continuous_depth_2D", "bi3dnet_continuous_depth_3D"]


def compute_cost_volume(features_left, features_right, disp_ids, max_disp, is_disps_per_example):

    batch_size = features_left.shape[0]  # batch_size=1
    feature_size = features_left.shape[1]  # feature_size=32
    H = features_left.shape[2]  # H=192
    W = features_left.shape[3]  # W=320

    psv_size = disp_ids.shape[1]  # psv_size=1

    psv = Variable(features_left.new_zeros(batch_size, psv_size, feature_size * 2, H, W + max_disp)).cuda()
    # psv is a batch_size*psv_size*(feature_size * 2)*H*(W + max_disp) Variable
    if is_disps_per_example:
        for i in range(batch_size):
            psv[i, 0, :feature_size, :, 0:W] = features_left[i]
            psv[i, 0, feature_size:, :, disp_ids[i, 0] : W + disp_ids[i, 0]] = features_right[i]
        psv = psv.contiguous()
    else:
        for i in range(psv_size):
            psv[:, i, :feature_size, :, 0:W] = features_left
            psv[:, i, feature_size:, :, disp_ids[0, i] : W + disp_ids[0, i]] = features_right
        psv = psv.contiguous()

    return psv


"""
Bi3DNet for continuous depthmap generation. Doesn't use 3D regularization.
"""


class Bi3DNetContinuousDepth2D(nn.Module):
    def __init__(self, options, featnet_arch, segnet_arch, refinenet_arch=None, max_disparity=192):

        super(Bi3DNetContinuousDepth2D, self).__init__()

        self.max_disparity = max_disparity
        self.max_disparity_seg = int(self.max_disparity / 3)
        self.is_disps_per_example = False
        self.is_save_memory = False

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False

        self.featnet = FeatNet.__dict__[featnet_arch](options, data=None)
        self.segnet = SegNet.__dict__[segnet_arch](options, data=None)
        if self.is_refine:
            self.refinenet = RefineNet.__dict__[refinenet_arch](options, data=None)

        return

    def forward(self, img_left, img_right, disp_ids):

        batch_size = img_left.shape[0]
        psv_size = disp_ids.shape[1]

        if psv_size == 1:
            self.is_disps_per_example = True
        else:
            self.is_disps_per_example = False

        # Feature Extraction
        features_left = self.featnet(img_left)
        features_right = self.featnet(img_right)
        feature_size = features_left.shape[1]
        H = features_left.shape[2]
        W = features_left.shape[3]

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, disp_ids, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]
        seg_raw_low_res = seg_raw_low_res.view(batch_size, 1, psv_size, H, W)

        # Upsampling
        seg_prob_low_res_up = torch.sigmoid(
            F.interpolate(
                seg_raw_low_res,
                size=[psv_size * 3, img_left.size()[-2], img_left.size()[-1]],
                mode="trilinear",
                align_corners=False,
            )
        )
        seg_prob_low_res_up = seg_prob_low_res_up[:, 0, 1:-1, :, :]

        # Projection
        disparity_normalized = torch.mean((seg_prob_low_res_up), dim=1, keepdim=True)

        # Refinement
        if self.is_refine:
            refine_net_input = torch.cat((disparity_normalized, img_left), dim=1)
            disparity_normalized = self.refinenet(refine_net_input)

        return seg_prob_low_res_up, disparity_normalized


def bi3dnet_continuous_depth_2D(options, data=None):

    # print("==> USING Bi3DNetContinuousDepth2D")
    # for key in options:
    #     if "bi3dnet" in key:
    #         print("{} : {}".format(key, options[key]))

    model = Bi3DNetContinuousDepth2D(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
    )

    if data is not None:
        # train=false,evaluation=True
        model.load_state_dict(data["state_dict"],True)

    return model


"""
Bi3DNet for continuous depthmap generation. Uses 3D regularization.
"""


class Bi3DNetContinuousDepth3D(nn.Module):
    def __init__(
        self,
        options,
        featnet_arch,
        segnet_arch,
        refinenet_arch=None,
        refinenet3d_arch=None,
        max_disparity=192,
    ):

        super(Bi3DNetContinuousDepth3D, self).__init__()

        self.max_disparity = max_disparity
        self.max_disparity_seg = int(self.max_disparity / 3)
        self.is_disps_per_example = False
        self.is_save_memory = False

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False

        self.featnet = FeatNet.__dict__[featnet_arch](options, data=None)
        self.segnet = SegNet.__dict__[segnet_arch](options, data=None)
        if self.is_refine:
            self.refinenet = RefineNet.__dict__[refinenet_arch](options, data=None)
            self.refinenet3d = RefineNet3D.__dict__[refinenet3d_arch](options, data=None)

        return

    def forward(self, img_left, img_right, disp_ids):

        batch_size = img_left.shape[0]
        psv_size = disp_ids.shape[1]

        if psv_size == 1:
            self.is_disps_per_example = True
        else:
            self.is_disps_per_example = False

        # Feature Extraction
        features_left = self.featnet(img_left)
        features_right = self.featnet(img_right)
        feature_size = features_left.shape[1]
        H = features_left.shape[2]
        W = features_left.shape[3]

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, disp_ids, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]  # cropped to remove excess boundary
        seg_raw_low_res = seg_raw_low_res.view(batch_size, 1, psv_size, H, W)

        # Upsampling
        seg_prob_low_res_up = torch.sigmoid(
            F.interpolate(
                seg_raw_low_res,
                size=[psv_size * 3, img_left.size()[-2], img_left.size()[-1]],
                mode="trilinear",
                align_corners=False,
            )
        )

        seg_prob_low_res_up = seg_prob_low_res_up[:, 0, 1:-1, :, :]

        # Upsampling after 3D Regularization
        seg_raw_low_res_refined = seg_raw_low_res
        seg_raw_low_res_refined[:, :, 1:, :, :] = self.refinenet3d(
            features_left, seg_raw_low_res_refined[:, :, 1:, :, :]
        )

        seg_prob_low_res_refined_up = torch.sigmoid(
            F.interpolate(
                seg_raw_low_res_refined,
                size=[psv_size * 3, img_left.size()[-2], img_left.size()[-1]],
                mode="trilinear",
                align_corners=False,
            )
        )

        seg_prob_low_res_refined_up = seg_prob_low_res_refined_up[:, 0, 1:-1, :, :]

        # Projection
        disparity_normalized_noisy = torch.mean((seg_prob_low_res_refined_up), dim=1, keepdim=True)

        # Refinement
        if self.is_refine:
            refine_net_input = torch.cat((disparity_normalized_noisy, img_left), dim=1)
            disparity_normalized = self.refinenet(refine_net_input)

        return (
            seg_prob_low_res_up,
            seg_prob_low_res_refined_up,
            disparity_normalized_noisy,
            disparity_normalized,
        )


def bi3dnet_continuous_depth_3D(options, data=None):

    # print("==> USING Bi3DNetContinuousDepth3D")
    # for key in options:
    #     if "bi3dnet" in key:
    #         print("{} : {}".format(key, options[key]))

    model = Bi3DNetContinuousDepth3D(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        refinenet3d_arch=options["bi3dnet_regnet_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
    )

    if data is not None:
        model.load_state_dict(data["state_dict"])

    return model


"""
Bi3DNet for binary depthmap generation.
"""


class Bi3DNetBinaryDepth(nn.Module):
    def __init__(
        self,
        options,
        featnet_arch,
        segnet_arch,
        refinenet_arch=None,
        featnethr_arch=None,
        max_disparity=192,
        is_disps_per_example=False,
        # config=None
    ):

        super(Bi3DNetBinaryDepth, self).__init__()
        self.max_disparity = max_disparity
        self.max_disparity_seg = int(max_disparity / 3)
        self.is_disps_per_example = is_disps_per_example

        self.is_refine = True
        if refinenet_arch == None:
            self.is_refine = False

        self.featnet = FeatNet.__dict__[featnet_arch](options, data=None)

        # 初始化针对FeatNet提取出分辨率为原图的1/3的特征映射进行位置编码的网络层。
        # 位置编码构造函数的输入参数为：d_model是输入图像的通道数，FeatNet网络提取出的特征映射具有32个通道。因此，d_model的值为32。
        # 第二个输入参数为输入图像的尺寸。
        # 对于该网络的训练和评估，若网络的输入图像为分辨率是960*576的SceneFlow数据集的图像，则经过FeatNet网络的三倍降采样之后，图像的分辨率变为320*192；j
        # 即：如果网络的输入图像为SceceFlow数据集的图像，位置编码构造函数的第二个参数为(192,320)。
        # 同理：对于输入是kitti2015数据集的图像，位置编码构造函数的第二个参数为(384/3,1248/3)，即(128,416)。
        # self.pos_encoding = PositionEncodingSine(32, (192, 320), temp_bug_fix=True)

        # 新增attention网络层。该网络层的构造函数的输入为一个字典，该字典中保存了attention计算过程中的所有属性。
        # 对于增加了attention计算的模型，还需要增加有关attention网络层有关的字典config。
        # config属性中：d_model表示特征的维数。在该算法中，每个像素点所构成的32维向量为一个特征。
        # 在多头注意力机制中，每个头的特征维数为d_model//nhead。
        # 在LoFTR算法中，输入到transformer中进行计算的像素点维度为256，nhead设置为8；因此每个“头”中像素点的维度是64。
        # 该网络中，输入到该transformer中进行计算的特征图像的通道数为32，如果nhead设置成8或者4，那么每个“头“中像素点的维度是4或者8，维度过小反而不具备区分性。
        # 因此，这里改为单头注意力机制。
        # 考虑到显卡内存的承受情况还有输入特征图的大小，这里的transformer只需两次连续的自注意力和交叉注意力的交替计算。
        # if config is None:
        #     config = {'nhead': 1, 'd_model': 32, 'layer_names':['self', 'cross'] * 2,'attention':'linear', 'temp_bug_fix':True, 'd_ffn':32}
        # self.loftr_coarse = LocalFeatureTransformer(config)

        self.featnethr = FeatNet.__dict__[featnethr_arch](options, data=None)

        self.segnet = SegNet.__dict__[segnet_arch](options, data=None)

        # 本网络中，只有SegRefine网络中新添的卷积层参数可以参与反向传播，其余网络层的参数无需变化。
        for p in self.parameters():
            p.requires_grad = False

        if self.is_refine:
            self.refinenet = RefineNet.__dict__[refinenet_arch](options, data=None)
        return

    def forward(self, img_left, img_right, disp_ids):

        batch_size = img_left.shape[0]
        psv_size = disp_ids.shape[1]

        if psv_size == 1:
            self.is_disps_per_example = True
        else:
            self.is_disps_per_example = False

        features = self.featnet(torch.cat((img_left, img_right), dim=0))

        features_left = features[:batch_size, :, :, :]
        features_right = features[batch_size:, :, :, :]

        feature_size = features_left.shape[1]  # 特征图的通道数
        H = features_left.shape[2]  # 特征图的高度
        W = features_left.shape[3]  # 特征图的宽度

        # 对左、右图特征映射分别做位置编码并展平。
        # features_left = self.pos_encoding(features_left)
        # features_left = rearrange(features_left, 'n c h w -> n (h w) c')
        # features_right = self.pos_encoding(features_right)
        # features_right = rearrange(features_right, 'n c h w -> n (h w) c')
        # 将展平之后的特征映射输入到transformer中进行计算。
        # 其中，对左特征图进行self-attention计算和cross-attention计算时，forward函数的输入x为左图特征映射，sourse为右图特征映射，换成右图同理。
        # features_left, features_right= self.loftr_coarse(features_left, features_right, None, None)
        # 做完transformer后，需要将左右图像的维度还原，以便进行代价体计算。
        # features_left = rearrange(features_left, 'n (h w) c -> n c h w',h=H,w=W)
        # features_right = rearrange(features_right, 'n (h w) c -> n c h w',h=H,w=W)

        if self.is_refine:
            features_lefthr = self.featnethr(img_left)

        # Cost Volume Generation
        psv = compute_cost_volume(
            features_left, features_right, disp_ids, self.max_disparity_seg, self.is_disps_per_example
        )

        psv = psv.view(batch_size * psv_size, feature_size * 2, H, W + self.max_disparity_seg)

        # Segmentation Network
        seg_raw_low_res = self.segnet(psv)[:, :, :, :W]  # cropped to remove excess boundary
        seg_prob_low_res = torch.sigmoid(seg_raw_low_res)
        seg_prob_low_res = seg_prob_low_res.view(batch_size, psv_size, H, W)

        seg_prob_low_res_up = F.interpolate(
            seg_prob_low_res, size=img_left.size()[-2:], mode="bilinear", align_corners=False
        )
        out = []
        out.append(seg_prob_low_res_up)

        # Refinement
        if self.is_refine:
            seg_raw_high_res = F.interpolate(
                seg_raw_low_res, size=img_left.size()[-2:], mode="bilinear", align_corners=False
            )
            # Refine Net
            features_left_expand = (
                features_lefthr[:, None, :, :, :].expand(-1, psv_size, -1, -1, -1).contiguous()
            )
            features_left_expand = features_left_expand.view(
                -1, features_lefthr.size()[1], features_lefthr.size()[2], features_lefthr.size()[3]
            )
            refine_net_input = torch.cat((seg_raw_high_res, features_left_expand), dim=1)
            # seg_raw_high_res = self.refinenet(refine_net_input)
            seg_raw_high_res, seg_raw_high_res_conv2 = self.refinenet(refine_net_input)

            seg_prob_high_res = torch.sigmoid(seg_raw_high_res)

            # seg_prob_high_res_copy = seg_prob_high_res[0, 0].clone().detach().cpu().numpy()
            # print("the seg_prob_high_res is:\n", seg_prob_high_res_copy)
            # print("the max value of the seg_prob_high_res is:", np.max(seg_prob_high_res_copy))
            # print("the min value of the seg_prob_high_res is:", np.min(seg_prob_high_res_copy))
            # print("the mean value of the seg_prob_high_res is:", np.mean(seg_prob_high_res_copy))
            # print("the var value of the seg_prob_high_res is:", np.var(seg_prob_high_res_copy))
            # del seg_prob_high_res_copy

            seg_prob_high_res_conv2 = torch.sigmoid(seg_raw_high_res_conv2)

            # seg_prob_high_res_conv2_copy = seg_prob_high_res_conv2[0, 0].clone().detach().cpu().numpy()
            # print("the seg_prob_high_res_conv2 is:\n", seg_prob_high_res_conv2_copy)
            # print("the max value of the seg_prob_high_res_conv2 is:", np.max(seg_prob_high_res_conv2_copy))
            # print("the min value of the seg_prob_high_res_conv2 is:", np.min(seg_prob_high_res_conv2_copy))
            # print("the mean value of the seg_prob_high_res_conv2 is:", np.mean(seg_prob_high_res_conv2_copy))
            # print("the var value of the seg_prob_high_res_conv2 is:", np.var(seg_prob_high_res_conv2_copy))
            # del seg_prob_high_res_conv2_copy

            seg_prob_high_res = seg_prob_high_res.view(
                batch_size, psv_size, img_left.size()[-2], img_left.size()[-1]
            )
            seg_prob_high_res_conv2 = seg_prob_high_res_conv2.view(
                batch_size, psv_size, img_left.size()[-2], img_left.size()[-1]
            )

            out.append(seg_prob_high_res)
            out.append(seg_prob_high_res_conv2)

        else:
            out.append(seg_prob_low_res_up)

        return out


def bi3dnet_binary_depth(options, data=None):

    # print("==> USING Bi3DNetBinaryDepth")
    # for key in options:
    #     if "bi3dnet" in key:
    #         print("{} : {}".format(key, options[key]))

    model = Bi3DNetBinaryDepth(
        options,
        featnet_arch=options["bi3dnet_featnet_arch"],
        segnet_arch=options["bi3dnet_segnet_arch"],
        refinenet_arch=options["bi3dnet_refinenet_arch"],
        featnethr_arch=options["bi3dnet_featnethr_arch"],
        max_disparity=options["bi3dnet_max_disparity"],
        is_disps_per_example=options["bi3dnet_disps_per_example_true"],
    )

    if data is not None:
        # 由于新增了transformer网络层,因此训练的时候设置为False.
        # 在evaluation的时侯,为了避免由于网络层的名称不匹配导致训练模型的参数未加载进来的问题,需要将strict选项设置为True.
        model.load_state_dict(data["state_dict"], False)

    return model
