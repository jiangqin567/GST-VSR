import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from motion_estimate import ResB, MEnet
from alignment import Resample_net, optical_flow_warp

class MCnet(nn.Module):

    def __init__(self, upscale_factor):
        super(MCnet, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x, optical_flow_resample_01, optical_flow_resample_21):  # 利用重采样后的光流对帧进行运动补偿,输入为x原始LR帧，预测的光流
        draft_cube = x

        for i in range(self.upscale_factor):
            for j in range(self.upscale_factor):
                draft_01 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1),
                                             optical_flow_resample_01[:, :, i::self.upscale_factor,
                                             j::self.upscale_factor] / self.upscale_factor)     #1

                draft_21 = optical_flow_warp(torch.unsqueeze(x[:, 2, :, :], dim=1),
                                             optical_flow_resample_21[:, :, i::self.upscale_factor,
                                             j::self.upscale_factor] / self.upscale_factor)       #1

                draft_cube = torch.cat((draft_cube, draft_01, draft_21), 1)       #


        return draft_cube    #[16, 35, 64, 64]


class SRnet(nn.Module):
    def __init__(self, upscale_factor):
        super(SRnet, self).__init__()
        self.conv = nn.Conv2d(35, 32, 3, 1, 1, bias=False)
        self.ResB_1 = ResB(32, 32)
        self.ResB_2 = ResB(32, 32)
        self.ResB_3 = ResB(64, 64)

        self.bottleneck = nn.Conv2d(160, upscale_factor ** 2, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(upscale_factor ** 2, upscale_factor ** 2, 3, 1, 1, bias=True)
        self.BN = nn.BatchNorm2d(upscale_factor ** 2,affine=True)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        input = self.conv(x)     #16, 32, 64, 64
        buffer_1 = self.ResB_1(input)
        buffer_2 = self.ResB_2(buffer_1)
        buffer_12 = torch.cat((buffer_1,buffer_2),1)   #64


        buffer_3 = self.ResB_3(buffer_12)   #64
        output = torch.cat((buffer_12, buffer_3, input), 1)  # 64,64,32=160
        output = self.bottleneck(output)  # 16
        output = self.conv_2(output)  # 16
        output = self.shuffle(output)  # 1

        return output  # 16, 1, 256, 256


class VSR(nn.Module):
    def __init__(self, upscale_factor):
        super(VSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.MEnet = MEnet(upscale_factor=upscale_factor)
        self.Resample = Resample_net(upscale_factor=upscale_factor)
        self.MCnet = MCnet(upscale_factor=upscale_factor)
        self.SRnet = SRnet(upscale_factor=upscale_factor)

    def forward(self, x):

        input_01 = torch.cat((torch.unsqueeze(x[:, 0, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)),
                             1)
        input_21 = torch.cat((torch.unsqueeze(x[:, 2, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)

        flow_01 = self.MEnet(input_01)
        flow_21 = self.MEnet(input_21)

        flow_01_resample = self.Resample(input_01, flow_01)  # #16*2*256*256
        flow_21_resample = self.Resample(input_21, flow_21)

        frame_mc = self.MCnet(x, flow_01_resample, flow_21_resample)    #16, 35, 64, 64

        output = self.SRnet(frame_mc)       #16, 1, 256, 256

        return output