import copy
import os
import random

import numpy
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')


    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        # 从数据集中加载左右图像、将图像转换为C*H*W类型的张量。然后，沿着C、H、W三个维度将图像以均值和方差均为0.5进行归一化。
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        left_img = transforms.functional.to_tensor(left_img)
        left_img = transforms.functional.normalize(left_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        right_img = transforms.functional.to_tensor(right_img)
        right_img = transforms.functional.normalize(right_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # 从数据集中加载视差图
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        # 对图像做裁剪。截取图像的顶点坐标为(0,0)，宽度和高度分别为w和h
        # 求解原图像的宽度和高度
        w, h = left_img.shape[2], left_img.shape[1]
        # 设置与裁剪和填充有关的宽度和高度
        # crop_w, crop_h = 480, 270
        # 裁剪图像的顶点选择为：以0.8的概率从顶点(0,0)、(480,0)、(0,270)、(480,270)围成的左上角区域中任意选择一个点；
        # 以0.2的概率从顶点(0,162)、(480,162)、(0,270)、(480,270)围成的矩形区域中任意选择一个点。
        # x1 = random.randint(0, w - crop_w)
        # if random.randint(0, 10) >= int(8):
        #     y1 = random.randint(0, h - crop_h)
        # else:
        #     y1 = random.randint(int(0.3 * h), h - crop_h)
        crop_w, crop_h = 960, 576
        assert crop_w % 96 == 0, "image dimensions should be a multiple of 96"
        assert crop_h % 96 == 0, "image dimensions should be a multiple of 96"
        # 求解要裁剪的图像的顶点
        x1 = random.randint(0, max(0, w - crop_w))  # w=crop_w，因此x1=0。
        y1 = random.randint(0, max(0, h - crop_h))  # h<crop_h，因此y1=0。
        # random crop (裁了跟没裁一样，就是把原来的图像取出来了)
        left_img = left_img[ :, y1: y1 + min(crop_h, h), x1: x1 + min(crop_w, w)]  # min(crop_h, h) = h
        right_img = right_img[ :, y1: y1 + min(crop_h, h), x1: x1 + min(crop_w, w)]  # min(crop_w, w) = w
        disparity = disparity[y1:y1 + min(crop_h, h), x1:x1 + min(crop_w, w)]

        # 对图像做左边界和上边界的填充。先计算填充宽度。
        pad_w = crop_w - w if crop_w - w > 0 else 0
        pad_h = crop_h - h if crop_h - h > 0 else 0
        # pad_w, pad_h = 0, 18
        pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
        # 填充图像：左边界填充pad_w个像素宽度，上边界填充pad_h个像素宽度。
        left_img_pad = pad_opr(left_img)
        right_img_pad = pad_opr(right_img)

        # 如果在SceneFlowDatasets的训练集上对数据训练，则在代分类视差范围{0,1,..,192}内随机选取一组视差。
        if self.training:
            # 随机生成一个视差值为disp_val所处的深度平面
            disp_val = np.random.choice(range(0,193,3))
            assert disp_val % 3 == 0, "disparity value should be a multiple of 3 as we downsample the image by 3"
            # 输入到网络中进行计算的视差张量disp_long
            disp_long = torch.Tensor([[disp_val / 3]]).type(torch.LongTensor)

            # disp_reference为所有像素点的视差值为disp_val，且大小和ground_truth disparity一样大的视差图。
            # 即：disp_reference为disp_val所处的深度平面上所有像素点的视差所构成的视差图
            disp_reference=np.full_like(disparity,disp_val)
            # 从相机方向看过去，如果某一个点位于参考视差所处深度平面之前(该点的视差值大于参考视差)，则该点的置信度为1；反之为0。
            mask=disparity>disp_reference
            disp_br = numpy.zeros((disparity.shape[0],disparity.shape[1]),dtype='float32')
            disp_br[mask]=1

            return {"left": left_img_pad,
                    "right": right_img_pad,
                    "top_pad": pad_h,
                    "left_pad": pad_w,
                    "dis_ref": disp_long,
                    "binary_disparity": disp_br,
                    "disparity": disparity}

        # 如果在SceneFlowDatasets的测试集上进行评估，则需要对5组代分类视差{24, 36, 54, 96, 144}进行mIou指标的评估。
        else:

            disp_vals = [24, 36, 54, 96, 144]
            # disp_ref_test = [tensor[[24]], tensor[[36]], tensor[[54]], tensor[[96]], tensor[[144]]]
            # br_disp_gt_test = [br_disp_gt(24), br_disp_gt(36), br_disp_gt(54), br_disp_gt(96), br_disp_gt(144)]
            disp_ref_test, br_disp_gt_test = [], []

            for disp_val in disp_vals:
                # 先将视差序列disp_vals中的每个视差disp_val转换为二维的Tensor。
                assert disp_val % 3 == 0, "disparity value should be a multiple of 3 as we downsample the image by 3"
                disp_long = torch.Tensor([[disp_val / 3]]).type(torch.LongTensor)
                disp_ref_test.append(disp_long)

                # 依次求解每组代分类视差所对应的二值视差图。
                disp_reference = np.full_like(disparity, disp_val)
                mask = disparity > disp_reference
                disp_br = numpy.zeros((disparity.shape[0], disparity.shape[1]), dtype='float32')
                disp_br[mask] = 1
                # 将求解出来的二值视差图依次放到数组br_disp_gt_test中。
                br_disp_gt_test.append(disp_br)

            del disp_vals, disparity

            return {"left": left_img_pad,
                    "right": right_img_pad,
                    "top_pad": pad_h,
                    "left_pad": pad_w,
                    "br_disp_gt": br_disp_gt_test,
                    "disp_ref": disp_ref_test,
            }
