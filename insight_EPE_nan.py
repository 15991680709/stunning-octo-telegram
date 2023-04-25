# -*-：coding:utf-8-*-
import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import models
import cv2
import numpy as np
from util import disp2rgb, str2bool
from datasets_c.data_io_c import pfm_imread
# Parse Arguments
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

# Input Pretrained Model
parser.add_argument("--pretrained", type=str, default="/media/fhy/My_Passport/ZKX/Bi3D-master/src/bi3d_models/model_weights/sf_continuous_depth_no_conf_reg.pth.tar")

# Input Data
# 本代码旨在对SceneFlowDatasets测试集中输出的EPE值为nan的数据进行真值视差图、连续深度估计结果和mask掩码进行可视化输出，分析该数据的EPE为什么是nan。
# 经过对SceneFlowDatasets测试集的再一次筛查，得到值为nan的索引为2261。
# 即：原SceneFlowDatasets测试集中的第2262组数据的EPE值为nan。
# 根据sceneflow_test.txt文件：第2262组数据的相对路径为：frames_finalpass/TEST/A/0005/left/0009.png frames_finalpass/TEST/A/0005/right/0009.png disparity/TEST/A/0005/left/0009.pfm
parser.add_argument("--img_left", type=str,
                    default="/media/fhy/My_Passport/GA-Net_SceneFlowDataset/frames_finalpass/TEST/A/0005/left/0010.png")
parser.add_argument("--img_right", type=str,
                    default="/media/fhy/My_Passport/GA-Net_SceneFlowDataset/frames_finalpass/TEST/A/0005/right/0010.png")
parser.add_argument("--img_disp", type=str,
                    default="/media/fhy/My_Passport/GA-Net_SceneFlowDataset/disparity/TEST/A/0005/left/0010.pfm")

# Input Parameters
parser.add_argument("--disp_range_min", type=int, default=0)
parser.add_argument("--disp_range_max", type=int, default=192)
parser.add_argument("--crop_height", type=int, default=270)
parser.add_argument("--crop_width", type=int, default=480)

# parse arguments, set seeds
args, unknown = parser.parse_known_args()
options = vars(args)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# model initial
device = torch.device('cuda:0')
network_data = torch.load(args.pretrained, map_location=device)
print("=> using pre-trained model '{}'".format(args.arch))
model = models.__dict__[args.arch](options, network_data).to(device)

# 将输入的立体图像对读取进来。
img_left = Image.open(args.img_left).convert("RGB")
img_left = transforms.functional.to_tensor(img_left)
img_left = transforms.functional.normalize(img_left, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_left = img_left.type(torch.cuda.FloatTensor)[None, :, :, :]

img_right = Image.open(args.img_right).convert("RGB")
img_right = transforms.functional.to_tensor(img_right)
img_right = transforms.functional.normalize(img_right, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_right = img_right.type(torch.cuda.FloatTensor)[None, :, :, :]

# 求解原图像的宽度和高度
w, h = img_left.shape[3], img_right.shape[2]
# 由于连续深度估计模型占用的cuda内存比二元深度估计网络占用的cuda内存多，
# 因此先对左右立体图像和真值视差图做裁剪，使得这些图像的分辨率变为原来的一半。
# 即：原来960*540的分辨率经过裁剪以后变为480*270。
crop_w = args.crop_width
crop_h = args.crop_height

# 裁剪图像的顶点选择为：以0.8的概率从顶点(0,0)、(480,0)、(0,270)、(480,270)围成的左上角区域中任意选择一个点；
# 以0.2的概率从顶点(0,162)、(480,162)、(0,270)、(480,270)围成的矩形区域中任意选择一个点。
x1 = random.randint(0, w - crop_w)
if random.randint(0, 10) >= int(8):
    y1 = random.randint(0, h - crop_h)
else:
    y1 = random.randint(int(0.3 * h), h - crop_h)

# 根据设置的裁剪宽度和在区域中随机选择的像素点作为裁剪后图像的左上角顶点对图像进行裁剪。
img_left = img_left[:, :, y1: y1 + min(crop_h, h), x1: x1 + min(crop_w, w)]
img_right = img_right[:, :, y1: y1 + min(crop_h, h), x1: x1 + min(crop_w, w)]

# 原来分辨率为960*540时，左侧填充宽度为0而上侧填充宽度为36，使得图像的分辨率变为960*576。再将填充之后的图像输入到网络中。
# 为了防止在segNet网络里解码器中相应层的升采样图像和编码器中对应的降采样图像沿着图像通道的方向进行拼接时由于尺寸不对应而报错，
# 让裁剪后的图像上侧填充18个像素宽度，左侧填充0个像素宽度，使得图像的分辨率变为480*288，并将填充后的图像作为网络的输入。
# 经过这一步，网络输入的立体图像的分辨率刚好为未经裁剪操作的，输入立体图像分辨率的一半。
pad_w, pad_h = 0, 18
pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
img_left_pad = pad_opr(img_left)
img_right_pad = pad_opr(img_right)

img_left_pad_copy = img_left_pad[0].clone().detach().cpu().numpy()
img_left_pad_copy = np.transpose(img_left_pad_copy, (1, 2, 0))
img_left_pad_copy = Image.fromarray(np.uint8(img_left_pad_copy)).convert('RGB')
img_left_pad_copy.save("./debug/left_img_crop.png")
print("左图像保存完毕")
img_right_pad_copy = img_right_pad[0].clone().detach().cpu().numpy()
img_right_pad_copy = np.transpose(img_right_pad_copy, (1, 2, 0))
img_right_pad_copy = Image.fromarray(np.uint8(img_right_pad_copy)).convert('RGB')
img_right_pad_copy.save("./debug/right_img_crop.png")
print("右图像保存完毕")

# 加载视差图
disparity, scale = pfm_imread(args.img_disp)
del scale
disparity = np.ascontiguousarray(disparity, dtype=np.float32)
disparity = disparity[y1:y1 + min(crop_h, h), x1:x1 + min(crop_w, w)]

# Prepare Disparities
max_disparity = args.disp_range_max  # 最大视差值对应最近的深度
min_disparity = args.disp_range_min  # 最小视差值对应最远的深度

assert max_disparity % 3 == 0 and min_disparity % 3 == 0, "disparities should be divisible by 3"

# 最近深度平面和最远深度平面之间一共存在多少组深度平面
max_disp_levels = (max_disparity - min_disparity) + 1

# 计算匹配代价体的时候使用了左右立体图像进行三倍降采样之后的左右特征映射的结果，因此传入网络中计算的视差上下限相应地变为原来的三分之一。
max_disparity_3x = int(max_disparity / 3)  # 传入到连续深度估计的网络中进行计算的视差值上限
min_disparity_3x = int(min_disparity / 3)  # 传入到连续深度估计的网络中进行计算的视差值下限
max_disp_levels_3x = (max_disparity_3x - min_disparity_3x) + 1  # 传入到深度估计网络中计算的视差平面的数量。

# disp_3x为视差序列[0,1,...,64]
disp_3x = np.linspace(min_disparity_3x, max_disparity_3x, max_disp_levels_3x, dtype=np.int32)
# 将视差序列转换为可以运行在cuda上的Tensor类型。
disp_long_3x_main = torch.from_numpy(disp_3x).type(torch.LongTensor).cuda()
# 将转换为Tensor的视差序列扩展一个批处理的维度，将这二维的视差序列输入到2D连续深度估计网络中。
disp_long_3x = disp_long_3x_main[None, :].expand(img_left.shape[0], -1)

# 通过立体图像对和待分类视差序列输出连续深度估计结果
model.eval()
with torch.no_grad():
    output_seg_low_res_upsample, output_disp_normalized = model(
        img_left_pad, img_right_pad, disp_long_3x
    )
del output_seg_low_res_upsample
output_disp_normalized = output_disp_normalized[:, :, pad_h:, pad_w:]

delta = 1
d_min_GT = min_disparity - 0.5 * delta  # 视差图中所有像素点对应视差值的下限
d_max_GT = max_disparity + 0.5 * delta  # 视差图中所有像素点对应视差值的上限
# 将输出视差图上每个像素点的视差值的范围限制到-0.5和192.5之间。
output_disp_normalized = torch.clamp(output_disp_normalized * delta * max_disp_levels + d_min_GT, min=d_min_GT, max=d_max_GT)

output_disp = output_disp_normalized[0, 0].clone().detach().cpu().numpy()
# 将真值视差图中视差值介于0到192的像素点筛选出来，和视差范围介于-0.5到192.5的真值视差图进行EPE指标的计算。
mask = (disparity > d_min_GT) & (disparity < d_max_GT)

error = np.mean(np.abs(output_disp[mask] - disparity[mask]))
print("the EPE error is:",error)

output_disp = Image.fromarray(output_disp.astype('uint16')).convert('L')
output_disp.save("./debug/disp_crop_estimate.png")
print("连续深度估计视差结果保存完毕")
disparity = Image.fromarray(disparity.astype('uint16')).convert('L')
disparity.save("./debug/disparity_crop.png")
print("真值视差图保存完毕")
mask = mask * 255
mask = Image.fromarray(mask.astype('uint16')).convert('L')
mask.save("./debug/mask.png")
print("掩码保存完毕")