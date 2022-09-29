import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class make_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(make_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, stride=1, padding=1,
                               bias=False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.leaky_relu(res)
        res = self.conv2(res)

        return x + res  # 64


class ResB(nn.Module):
    def __init__(self, layers, channels_in, channels_out):
        super(ResB, self).__init__()
        modules = []
        for i in range(layers):
            modules.append(make_layer(channels_in, channels_out))
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(channels_in, channels_out, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class MEnet(nn.Module):
    def __init__(self, upscale_factor):
        super(MEnet, self).__init__()
        self.upscale_factor = upscale_factor
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bicubic')
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
        self.shuffle = nn.PixelShuffle(upscale_factor)  # *4

        # feature extra
        self.conv_1 = nn.Conv2d(2, 64, 3, 1, 1, bias=False)

        # Resnet
        self.ResB = ResB(7, 64, 64)

        self.conv_2 = nn.Conv2d(64, 64 * 2, 3, 1, 1, bias=False)
        self.bottleneck = nn.Conv2d(64 * 2, 32, 3, 1, 1, bias=False)
        self.conv_3 = nn.Conv2d(32, 2, 3, 1, 1, bias=False)


    def forward(self, x):
        input = self.conv_1(x)  # 16*64*64*64
        out_1 = self.ResB(input)  # [16, 64, 64, 64]
        out_1 = self.leaky_relu(out_1)
        out_2 = self.conv_2(out_1)  # 16*128*64*64
        out_2 = self.leaky_relu(out_2)
        out_3 = self.bottleneck(out_2)  # [16, 32, 64, 64]
        out_3 = self.leaky_relu(out_3)
        out_4 = self.conv_3(out_3)  # [16, 2, 64, 64]
        out_4 = self.leaky_relu(out_4)
        optical_flow = out_4  # [16, 2, 64, 64]

        return optical_flow
