import os
import random
import numpy
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets_c.data_io_c import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch


class KITTIDataset_c(Dataset):
    def __init__(self, kitti15_datapath, list_filename, training):
        self.datapath_15 = kitti15_datapath
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
        self.datapath = self.datapath_15  # kitti15的数据集路径。

        # 从数据集中加载左右图像，将图像转换为array类型的矩阵进行裁剪。
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        left_img = numpy.array(left_img)
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        right_img = numpy.array(right_img)

        # 求解原图像的宽度和高度(kitti15中图片的分辨率并不完全相同，有的为1241*375；有的为1242*376。)
        # 因此，统一设置裁剪区域的宽度为：max(1241,1242)/2=621；裁剪区域的高度为：max(375,376)/2=188。
        w, h = left_img.shape[1], left_img.shape[0]
        # 由于连续深度估计模型占用的cuda内存比二元深度估计网络占用的cuda内存多，
        # 因此先对左右立体图像和真值视差图做裁剪，使得这些图像的分辨率变为原来的一半。
        # 即：原来960*540的分辨率经过裁剪以后变为621*188。
        crop_w, crop_h = 480, 270
        # 裁剪图像的顶点选择为：从顶点(0,0)、(w - 621,0)、(0,h - 188)、(w - 621,h - 188)围成的左上角区域中任意选择一个点。
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        # 根据设置的裁剪宽度和在区域中随机选择的像素点作为裁剪后图像的左上角顶点对图像进行裁剪。
        left_img = left_img[y1: y1 + min(crop_h, h), x1: x1 + min(crop_w, w), :]
        right_img = right_img[y1: y1 + min(crop_h, h), x1: x1 + min(crop_w, w), :]

        # 将图像转换为C * H * W类型的张量。然后，沿着C、H、W三个维度将图像以均值和方差均为0.5进行归一化。
        left_img = transforms.functional.to_tensor(left_img)
        left_img = transforms.functional.normalize(left_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        right_img = transforms.functional.to_tensor(right_img)
        right_img = transforms.functional.normalize(right_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        # 原来分辨率为1241*375或者1242*376时，将图像填充到分辨率为1248*384的大图像。再将填充之后的图像输入到网络中。
        # 为了防止在segNet网络里解码器中相应层的升采样图像和编码器中对应的降采样图像沿着图像通道的方向进行拼接时由于尺寸不对应而报错，
        # 让裁剪后的图像上侧填充4个像素宽度，左侧填充3个像素宽度，使得图像的分辨率变为624*192，并将填充后的图像作为网络的输入。
        # 经过这一步，网络输入的立体图像的分辨率刚好为未经裁剪操作的，输入立体图像分辨率的一半。
        pad_w, pad_h = 0, 18
        pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))
        left_img_pad = pad_opr(left_img)
        right_img_pad = pad_opr(right_img)

        # 如果当前抓取的立体图像对有ground_truth disparity，则单独对disparity进行处理。
        if self.disp_filenames:
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index])) # 加载
            # 对视差图也进行同样的裁剪操作，使得每组图像中的视差图的分辨率均为480*270。
            disparity = disparity[y1:y1 + min(crop_h, h), x1:x1 + min(crop_w, w)]

            # 在二元深度估计中，对场景中像素点的深度进行分类的视差平面是视差搜寻范围[min_disp,max_disp]内一个参考视差值对应的视差平面。
            # 而在连续深度估计中，对场景中像素点的深度进行分类的视差平面是视差搜寻范围[min_disp,max_disp]内一组参考视差值对应的视差平面簇。
            max_disp, min_disp = 192, 0
            assert max_disp % 3 == 0 and min_disp % 3 == 0, "disparity value should be a multiple of 3 as we downsample the image by 3"

            # 在连续深度估计网络中，使用FeatNet提取出的左右立体图像的特征映射和视差序列计算匹配代价体。
            # 在FeatNet网络的一开始，对左右立体图像进行了3倍的降采样。因此，通过FeatNet网络提取出的特征映射的分辨率为左右立体图像的三分之一。
            max_disp_3x = int(max_disp / 3)  # 传入到连续深度估计的网络中进行计算的视差值上限
            min_disp_3x = int(min_disp / 3)  # 传入到连续深度估计的网络中进行计算的视差值下限
            level_num_3x = max_disp_3x - min_disp_3x + 1
            disparity_level_3x = np.linspace(min_disp_3x, max_disp_3x, level_num_3x, dtype=np.int32)

            # kitti2015数据集中同理：在视差序列{0,1,..,192}中任选一组视差并计算相应视差的二值视差图只需在训练过程中进行，评估过程不需要。
            if self.training:
                # 对于每一组图像，在视差搜索范围内任意选择一个视差值，计算该组图像在该视差值下的二值视差图。
                # 计算二值视差图的思路同二元深度估计网络的训练代码。
                disp_val = random.randint(min_disp, max_disp)
                disp_reference = np.full_like(disparity, disp_val)
                mask = disparity > disp_reference
                disp_br = numpy.zeros((disparity.shape[0], disparity.shape[1]), dtype='float32')
                disp_br[mask] = 1

                return {"left": left_img_pad,
                        "right": right_img_pad,
                        "top_pad": pad_h,
                        "left_pad": pad_w,
                        "disparity_list": disparity_level_3x,
                        "dis_ref": disp_val,
                        "disparity": disparity,
                        "binary_disparity": disp_br
                        }

            else:
                return {"left": left_img_pad,
                        "right": right_img_pad,
                        "top_pad": pad_h,
                        "left_pad": pad_w,
                        "disparity_list": disparity_level_3x,
                        "disparity": disparity,
                        }

        else:
            return {"left": left_img_pad,
                    "right": right_img_pad,
                    "top_pad": pad_h,
                    "left_pad": pad_w,
                    }
