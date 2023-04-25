import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(192, 320), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        # y_position本来为维度是(192,320)并且元素值全为1的Tensor,经过对行为进行累加的函数之后,y_position的每一行元素均为之前所有行的累加.
        # 即:经过了cumsum(0)函数以后,y_position的第1行元素值全为1,第2行值全为2,..,第192行的值全为192.
        # 经过了这个操作后,y_position上每个元素的值代表了这个元素在像素坐标系下的纵坐标.
        # 同理:张量x_position = torch.ones(max_shape).cumsum(1)上的每个元素的值代表了这个元素在像素坐标系下的横坐标.
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        # d_model在该函数中表示输入到位置编码函数的图像的通道数.在LoFTR中,以图像上的每一个像素点为特征向量,则d_model表示特征向量的维数.
        # 令:图像通道的序号从0开始.
        # 则数组torch.arange(0, d_model//2, 2)表示了:将输入图像中所有图像通道的前d_model//2个通道每两个通道分为一组,每一组通道的起始序号.
        # 若将图像通道中前d_model//2个通道每两个通道分为一组,则一共可以分为d_model//4组.
        # 令:每组的序号从0开始,则torch.arange(0, d_model//2, 2)数组的内容为:[0*2,1*2,..,(d_model//4-1)*2].
        # 即:torch.arange(0, d_model//2, 2)数组的内容为:[0,2,..,d_model//2-2].
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]  # div_term为维度是(d_model//4,1,1)的张量.
        # 张量x_position * div_term的维度为[d_model//4,*(max_shape)].
        # 其中,三维张量x_position * div_term沿着第0维顺序的任意第i个维度是max_shape的二维张量(i=0,1,...,d_model//4-1)中的任意一个像素点的像素值为该点的横坐标与第i组图像通道的起始图像通道序号2i经过对数函数和指数函数变换后的积.
        # 同样,将输入图像的整个图像通道沿着通道序号分为d_model//4组,那么每组具有4个图像通道.
        # 以下代码的含义为:按照图像通道划分后,这8组图像的每组图像中第一通道图像中的像素值为像素点的横坐标与按照将图像通道的前d_model//2
        # 个通道每两个分组中相应的组对应的起始图像通道序号经过对数函数与指数函数变换结果的积的正弦变换;
        # 这8组图像的每组图像中第二通道图像中的像素值为像素点的横坐标与按照将图像通道的前d_model // 2
        # 个通道每两个分组中相应的组对应的起始图像通道序号经过对数函数与指数函数变换结果的积的余弦变换;
        # 这8组图像的每组图像中第三通道图像中的像素值为像素点的纵坐标与按照将图像通道的前d_model // 2
        # 个通道每两个分组中相应的组对应的起始图像通道序号经过对数函数与指数函数变换结果的积的正弦变换;
        # 这8组图像的每组图像中第四通道图像中的像素值为像素点的纵坐标与按照将图像通道的前d_model // 2
        # 个通道每两个分组中相应的组对应的起始图像通道序号经过对数函数与指数函数变换结果的积的余弦变换;
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
