import os
import random

import numpy
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch


class KITTIDataset(Dataset):
    def __init__(self, kitti15_datapath, list_filename, training):
        self.datapath_15 = kitti15_datapath
        # self.datapath_12 = kitti12_datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        # 判断当前抓取到的图像是kitti12数据集的还是kitti15数据集的。
        self.datapath = self.datapath_15
        # left_name = self.left_filenames[index].split('/')[1]
        # if left_name.startswith('image'):
        #     self.datapath = self.datapath_15
        # else:
        #     self.datapath = self.datapath_12

        # 从数据集中加载左右图像，并将图像转换为array类型的矩阵。
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        left_img = numpy.array(left_img)
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        right_img = numpy.array(right_img)

        # crop and pad image 截取图像的顶点坐标为(0,0)，宽度和高度分别为w和h
        # 求解原图像的宽度和高度
        w, h = left_img.shape[1], left_img.shape[0]
        # 设置与裁剪和填充有关的宽度和高度
        crop_w, crop_h = 1248, 384
        # 求解要裁剪的图像的顶点(0,0)
        x1 = random.randint(0, max(0, w - crop_w))  # x1=0
        y1 = random.randint(0, max(0, h - crop_h))  # y1=0
        # random crop (裁了跟没裁一样，就是把原来的图像取出来了)
        left_img= left_img[ y1: y1 + min(crop_h,h), x1: x1 + min(crop_w,w), :]
        right_img = right_img[ y1: y1 + min(crop_h,h), x1: x1 + min(crop_w,w), :]

        # 将图像转换为C * H * W类型的张量。然后，沿着C、H、W三个维度将图像以均值和方差均为0.5进行归一化。
        left_img = transforms.functional.to_tensor(left_img)
        left_img = transforms.functional.normalize(left_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        right_img = transforms.functional.to_tensor(right_img)
        right_img = transforms.functional.normalize(right_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # 对图像做左边界和上边界的填充。先计算填充宽度。
        pad_w = crop_w - w if crop_w - w > 0 else 0
        pad_h = crop_h - h if crop_h - h > 0 else 0
        pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
        # pad image：将左右立体图像对均填充为384*1248的分辨率
        left_img_pad = pad_opr(left_img)
        right_img_pad = pad_opr(right_img)

        # 如果当前抓取的立体图像对有ground_truth disparity，则单独对disparity进行处理。
        if self.disp_filenames:
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index])) # 加载
            disparity = disparity[y1:y1 + min(crop_h,h), x1:x1 + min(crop_w,w)] # 裁剪(取原图)
            # 将ground_truth disparity也填充为384 * 1248的分辨率
            disparity_pad = np.lib.pad(disparity, ((pad_h, 0), (pad_w, 0)), mode='constant', constant_values=0)

            # 随机生成一个视差值为disp_val所处的深度平面
            disp_val = np.random.choice(range(0, 193, 3))
            assert disp_val % 3 == 0, "disparity value should be a multiple of 3 as we downsample the image by 3"
            # 输入到网络中进行计算的视差张量disp_long
            disp_long = torch.Tensor([[disp_val / 3]]).type(torch.LongTensor)

            # disp_reference为所有像素点的视差值为disp_val，且大小和ground_truth disparity一样大的视差图。
            # 即：disp_reference为disp_val所处的深度平面上所有像素点的视差所构成的视差图
            disp_reference = np.full_like(disparity_pad, disp_val)
            # 从相机方向看过去，如果某一个点位于参考视差所处深度平面之前(该点的视差值大于参考视差)，则该点的置信度为1；反之为0。
            mask = disparity_pad > disp_reference
            disp_br_crop = numpy.zeros((disparity_pad.shape[0], disparity_pad.shape[1]), dtype='float32')
            disp_br_crop[mask] = 1

            return {"left": left_img_pad,
                    "right": right_img_pad,
                    "left_pad":pad_w,
                    "top_pad":pad_h,
                    "dis_ref": disp_long,
                    "binary_disparity": disp_br_crop,
                    "disparity":disparity_pad}

        else:
            return {"left": left_img_pad,
                    "right": right_img_pad,
                    "left_pad": pad_w,
                    "top_pad": pad_h,
                    "dis_ref": None,
                    "binary_disparity":None,
                    "disparity":None}

# if self.training:
        #     # random cropping
        #     crop_w, crop_h = 576, 384
        #     x1 = random.randint(0, max(0,w - crop_w))
        #     y1 = random.randint(0, max(0,h - crop_h))
        #
        #     # random crop
        #     left_img = left_img[y1:y1 + min(crop_h,h), x1:x1 + min(crop_w,w),:]
        #     right_img = right_img[y1:y1 + min(crop_h,h), x1:x1 + min(crop_w,w),:]
        #
        #     # transform to Tensor
        #     left_img = transforms.functional.to_tensor(left_img)
        #     left_img = transforms.functional.normalize(left_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #     right_img = transforms.functional.to_tensor(right_img)
        #     right_img = transforms.functional.normalize(right_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #
        #     # top padding to crop_h
        #     pad_h = crop_h - h if crop_h - h > 0 else 0
        #     pad_opr = torch.nn.ZeroPad2d((0, 0, pad_h, 0))
        #     left_img_pad = pad_opr(left_img)
        #     right_img_pad = pad_opr(right_img)
        #
        #     # load and crop disparity
        #     disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        #     disparity = disparity[y1:y1 + min(crop_h,h), x1:x1 + min(crop_w,w)]
        #     disparity_pad = np.lib.pad(disparity, ((pad_h, 0), (0, 0)), mode='constant', constant_values=0)
        #
        #     # reference disp plane
        #     disp_val = np.random.choice(range(0, 193, 3))
        #     assert disp_val % 3 == 0, "disparity value should be a multiple of 3 as we downsample the image by 3"
        #     disp_long = torch.Tensor([[disp_val / 3]]).type(torch.LongTensor)
        #
        #     disp_reference = np.full_like(disparity_pad, disp_val)
        #     mask = disparity_pad > disp_reference
        #     disp_br = numpy.zeros((disparity_pad.shape[0], disparity_pad.shape[1]), dtype='float32')
        #     disp_br[mask] = 1
        #
        #     return {"left": left_img_pad,
        #             "right": right_img_pad,
        #             "top_pad": pad_h,
        #             "dis_ref": disp_long,
        #             "binary_disparity": disp_br,
        #             "disparity": disparity_pad}
