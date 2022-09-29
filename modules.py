import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class innerBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super(innerBlock, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv0 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding='same',
                               bias=False)
        # self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding='same',
        #                        bias=False)
        # self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding='same',
        #                        bias=False)

    def forward(self, x):
        x_1 = self.leaky_relu(self.conv0(x))
        # x_2 = self.conv1(x_1)
        # out = self.conv2(x_2)
        # out = torch.cat((x, out), 1)
        return x_1


# # 残差块
# class ResB(nn.Module):
#     def __init__(self, n_layer, channels_in, channels_out):
#         super(ResB, self).__init__()
#         modules = []
#         channels_buffer = channels_in
#         for i in range(n_layer):
#             modules.append(innerBlock(channels_buffer, channels_out))
#             channels_buffer += channels_in
#         self.innerBlocknet = nn.Sequential(*modules)  # 按顺序加入到Sequential容器,构成内部的残差块
#         # self.BN = nn.BatchNorm2d(channels_out, affine=True)  # batchnormal替代1*1卷积
#         self.conv = nn.Conv2d(channels_buffer, channels_in, kernel_size=1, padding=0, bias=False)
#
#     def forward(self, x):
#         out = self.innerBlocknet(x)
#         out = self.BN(out)
#         out = out + x
#         return out

class ResB(nn.Module):
    def __init__(self, n_layer, channels_in, channels_out):
        super(ResB, self).__init__()
        modules = []
        channels_buffer = channels_in
        for i in range(n_layer):
            modules.append(innerBlock(channels_buffer, channels_out))
            channels_buffer += channels_in
        self.innerBlocknet = nn.Sequential(*modules)  # 按顺序加入到Sequential容器,构成内部的残差块
        self.conv = nn.Conv2d(channels_buffer, channels_in, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.innerBlocknet(x)
        # out = self.BN(out)
        out = out + x
        return out


class OFRnet(nn.Module):
    def __init__(self, upscale_factor, is_training):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final_upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')   #*4
        self.shuffle = nn.PixelShuffle(upscale_factor)    #*4
        self.upscale_factor = upscale_factor
        self.is_training = is_training

        self.conv_L1_1 = nn.Conv2d(2, 32, 3, 1, 1, bias=False)
        # 残差网络
        self.ResB_L1_1 = ResB(1, 32, 32)
        self.ResB_L1_2 = ResB(1, 32, 32)

        self.bottleneck_L1 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L1_2 = nn.Conv2d(2, 32, 3, 1, 1, bias=False)   #32,32s
        self.BN_L1 = nn.BatchNorm2d(32, affine=True)


    # OFRnet模型的输入是input_01和input_21:   16*2*32*32
    def forward(self, x):
        x_L1 = self.pool(x)       #16*2*16*16
        _, _, h, w = x_L1.size()
        input_L1 = self.conv_L1_1(x_L1)      #16*32*16*16
        buffer_1 = self.ResB_L1_1(input_L1)  #16*32*16*16
        # print("buffer",buffer_1.size())
        buffer_2 = self.ResB_L1_2(buffer_1)  #16*32*16*16
        # print("buffer",buffer_2.size())
        buffer = torch.cat((buffer_1, buffer_2), 1)  # 拼接：16*64*16*16
        # print("buffer", buffer.size())

        optical_flow_L1 = self.bottleneck_L1(buffer)    #16*2*16*16
        optical_flow_L1 = self.conv_L1_2(optical_flow_L1)   #16*32*16*16
        # print("optical_flow",optical_flow_L1.size())
        optical_flow_L1 = self.BN_L1(optical_flow_L1)
        # print("optical_flow",optical_flow_L1.size())    #16, 32, 16, 16]
        optical_flow_L1_upscaled = self.upsample(optical_flow_L1)
        # print("optical_flow",optical_flow_L1_upscaled.size())    #16, 32, 32, 32]
        optical_flow_L1_shuffle = self.shuffle(optical_flow_L1_upscaled)   #16*2*128*128
        # print("optical_flow_L1_shuffle",optical_flow_L1_shuffle.size())
        return optical_flow_L1_shuffle     #




class SRnet(nn.Module):
    def __init__(self, upscale_factor, is_training):
        super(SRnet, self).__init__()
        self.conv = nn.Conv2d(35, 64, 3, 1, 1, bias=False)  # 输入35，输出64

        self.bottleneck = nn.Conv2d(64, upscale_factor ** 2, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(upscale_factor ** 2, upscale_factor ** 2, 3, 1, 1, bias=True)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.is_training = is_training

    def forward(self, x):
        input = self.conv(x)        #35-64
        output = self.bottleneck(input)     #64-16
        output = self.conv_2(output)        #16-16
        output = self.shuffle(output)         #16-1   16*16-128*128
        # print("output:",output.size())
        return output


class SOFVSR(nn.Module):
    def __init__(self, upscale_factor, is_training=True):
        super(SOFVSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.is_training = is_training
        self.OFRnet = OFRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.SRnet = SRnet(upscale_factor=upscale_factor, is_training=is_training)

    # 输x[:,0,:,:]输入为16*3*32*32,第二维分别是3帧的y通道，取第二维的0维,即第一帧的一通道，x[:,0,:,:]维度：16*1*32*23，
    def forward(self, x):
        input_01 = torch.cat((torch.unsqueeze(x[:, 0, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)),1)
        input_21 = torch.cat((torch.unsqueeze(x[:, 2, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)

        flow_01_L3 = self.OFRnet(input_01)  #16*2*32*32
        flow_21_L3 = self.OFRnet(input_21)  #16*2*32*32
        draft_cube = x  # 16*3*32*32

        for i in range(self.upscale_factor):
            for j in range(self.upscale_factor):
                draft_01 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1),
                                             flow_01_L3[:, :, i::self.upscale_factor,   #flow_01_L3取的是i行和4行，j列和4列
                                             j::self.upscale_factor] / self.upscale_factor)
                draft_21 = optical_flow_warp(torch.unsqueeze(x[:, 2, :, :], dim=1),
                                             flow_21_L3[:, :, i::self.upscale_factor,
                                             j::self.upscale_factor] / self.upscale_factor)
                draft_cube = torch.cat((draft_cube, draft_01, draft_21), 1)

        output = draft_cube      #draft torch.Size([16, 35, 32, 32])

        output = self.SRnet(output)

        return torch.squeeze(output)


def optical_flow_warp(image, image_optical_flow):  #
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    # print("image_optical_flow",image_optical_flow.size())   #16*2*32*32
    b, _, h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))  # 复制数组
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
    grid = grid + torch.cat((flow_0, flow_1), 1)
    # print(grid.size())   #16*2*16*16
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    output = F.grid_sample(image, grid, padding_mode='border', align_corners=False)  # 16*1*32*32
    # print(output.size())
    return output
