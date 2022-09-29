import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from motion_estimate import ResB


def optical_flow_warp(image, image_optical_flow):  #
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b,2, h, w)
    """
    b, _, h, w = image.size()
    #print(image.size())
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] , dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] , dim=1)

    grid = grid + torch.cat((flow_0, flow_1), 1)

    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    # print(grid.size())  #b,h,w,2
    output = F.grid_sample(image, grid, padding_mode='border', align_corners=False)  # 16*1*64*64
    return output

class Align_net(nn.Module):

    def __init__(self, upscale_factor):
        super(Align_net, self).__init__()
        self.upscale_factor = upscale_factor
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bicubic')
        self.shuffle = nn.PixelShuffle(upscale_factor)  # *4
        self.conv_1 = nn.Conv2d(6, 64, 3, 1, 1, bias=False)

        # Resnet
        self.ResB_1 = ResB(5,64,64)
        self.conv_2 = nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.bottleneck = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(32, 2, 3, 1, 1, bias=False)

    def forward(self, x, optical_flow_up):

        x_w = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1), optical_flow_up)   #16*1*64*64
        x_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_w
        x_cat = torch.cat((x, x_w, x_res, optical_flow_up), 1)  # 16*6*64*64：2+1+1+2=6

        input = self.conv_1(x_cat)  # 16*64*64*64
        buffer_1 = self.ResB_1(input)   #16*64*64*64
        buffer_1 = self.leaky_relu(buffer_1)
        buffer = torch.cat((buffer_1, input), 1)  #16*128*64*64
        buffer_3 = self.conv_2(buffer)     #16*64*64*64
        buffer_3 = self.leaky_relu(buffer_3)

        optical_flow_res = self.bottleneck(buffer_3)  # 16*32*64*64
        optical_flow_shuffle = self.shuffle(optical_flow_res) + self.upsample(optical_flow_up)   #16*2*256*256

        return optical_flow_shuffle





