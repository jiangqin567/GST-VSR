import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from motion_estimate import ResB, MEnet
from alignment import Align_net, optical_flow_warp


class SRnet(nn.Module):
    def __init__(self, upscale_factor, frame_num):
        super(SRnet, self).__init__()

        self.conv = nn.Conv2d(frame_num + (frame_num - 1) * upscale_factor ** 2, 128, 3, 1, 1, bias=False)  # 35,69,103
        self.ResB = ResB(5, 128, 128)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.bottleneck = nn.Conv2d(128, upscale_factor ** 3, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(upscale_factor ** 3, upscale_factor ** 2, 3, 1, 1, bias=False)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        input = self.conv(x)  # 16, 128, 64, 64
        buffer_1 = self.ResB(input)  # 16, 128, 64, 64
        buffer_1 = self.leaky_relu(buffer_1)
        output = self.bottleneck(buffer_1)  # 64
        output = self.leaky_relu(output)
        output = self.conv_2(output)  # 32
        output = self.shuffle(output)  # 1

        return output  # 16, 1, 256, 256


class VSR(nn.Module):
    def __init__(self, upscale_factor, frame_num):
        super(VSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.MEnet = MEnet(upscale_factor=upscale_factor)
        self.Align = Align_net(upscale_factor=upscale_factor)
        self.SRnet = SRnet(upscale_factor=upscale_factor, frame_num=frame_num)

    def forward(self, x):

        _, c, _, _ = x.size()
        flow_draft = x
        ME_res_list = []
        Ali_res_list = []

        # input_01 = torch.cat((torch.unsqueeze(x[:, 0, :, :], dim=1), torch.unsqueeze(x[:,  1, :, :], dim=1)),
        #                   1)  # input:16*2*64*64
        # flow_01 = self.MEnet(input_01)
        # flow_align_01 = self.Align(input_01, flow_01)
        # for i in range(self.upscale_factor):
        #     for j in range(self.upscale_factor):
        #         flow_mc = optical_flow_warp(torch.unsqueeze(x[:, id, :, :], dim=1),
        #                                     flow_align_01[:, :, i::self.upscale_factor,
        #                                     j::self.upscale_factor] / self.upscale_factor)
        #         flow_draft = torch.cat((flow_draft, flow_mc), 1)

        for id in range(0, c - 1):
            input = torch.cat((torch.unsqueeze(x[:, id, :, :], dim=1), torch.unsqueeze(x[:, id + 1, :, :], dim=1)),
                              1)  # input:16*2*64*64
            flow = self.MEnet(input)
            ME_res_list.append(flow)
            flow_resample = self.Align(input, flow)
            Ali_res_list.append(flow_resample)

            for i in range(self.upscale_factor):
                for j in range(self.upscale_factor):
                    flow_mc = optical_flow_warp(torch.unsqueeze(x[:, id, :, :], dim=1),
                                                flow_resample[:, :, i::self.upscale_factor,
                                                j::self.upscale_factor] / self.upscale_factor)

                    flow_draft = torch.cat((flow_draft, flow_mc), 1)

        # print("flow",flow_draft.size())   #[16, 35, 64, 64]
        output = self.SRnet(flow_draft)  ##torch.Size([16, 1, 256, 256])
        return output, ME_res_list, Ali_res_list
