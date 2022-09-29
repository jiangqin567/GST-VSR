import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

def optical_flow_warp(image, image_optical_flow):  #
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    # print(image_optical_flow.size())   #16*2*32*32
    b, _, h, w = image.size()
    grid = np.meshgrid(range(w), range(h))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1
    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))  #复制数组
    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :] * 31 / (w - 1), dim=1)
    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 31 / (h - 1), dim=1)
    grid = grid + torch.cat((flow_0, flow_1), 1)
    # print(grid.size())   #16*2*16*16
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    output = F.grid_sample(image, grid, padding_mode='border',align_corners=False)  #16*1*32*32
    # print(output.size())
    return output


class make_dense(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super(make_dense, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = self.leaky_relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

#残差块
class RDB(nn.Module):
    def __init__(self, nDenselayer, channels, growth):   #4，32，32
        super(RDB, self).__init__()
        modules = []
        channels_buffer = channels
        for i in range(nDenselayer):
            modules.append(make_dense(channels_buffer, growth))    #一层卷积：输入32输出32，卷积核3
            channels_buffer += growth
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(channels_buffer, channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x

        return out


class OFRnet(nn.Module):
    def __init__(self, upscale_factor, is_training):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.final_upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.upscale_factor = upscale_factor
        self.is_training = is_training

        # Level 1
        self.conv_L1_1 = nn.Conv2d(2, 32, 3, 1, 1, bias=False)
        #残差网络
        self.RDB1_1 = RDB(4, 32, 32)
        self.RDB1_2 = RDB(4, 32, 32)

        self.bottleneck_L1 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L1_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)

        # Level 2
        self.conv_L2_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)

        self.RDB2_1 = RDB(4, 32, 32)
        self.RDB2_2 = RDB(4, 32, 32)

        self.bottleneck_L2 = nn.Conv2d(64, 2, 3, 1, 1, bias=False)
        self.conv_L2_2 = nn.Conv2d(2, 2, 3, 1, 1, bias=True)

        # Level 3
        self.conv_L3_1 = nn.Conv2d(6, 32, 3, 1, 1, bias=False)

        self.RDB3_1 = RDB(4, 32, 32)
        self.RDB3_2 = RDB(4, 32, 32)

        self.bottleneck_L3 = nn.Conv2d(64, 2 * upscale_factor ** 2, 3, 1, 1, bias=False)
        self.conv_L3_2 = nn.Conv2d(2 * upscale_factor ** 2, 2 * upscale_factor ** 2, 3, 1, 1, bias=True)

    #OFRnet模型的输入是input_01和input_21:   16*2*32*32
    def forward(self, x):

        # Level 1
        # print("x:",x.size())
        x_L1 = self.pool(x)    #平均池化降维:16*2*16*16
        # print("x_L1:",x_L1.size())
        _, _, h, w = x_L1.size()
        # print("h,w",h,w)
        input_L1 = self.conv_L1_1(x_L1)   #16*32*16*16
        # print(input_L1.size())
        buffer_1 = self.RDB1_1(input_L1)   #不改变尺寸大小：16*32*16*16
        # print("buffer_1:",buffer_1.size())
        buffer_2 = self.RDB1_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)    #拼接：16*64*16*16
        # print(buffer.size())
        optical_flow_L1 = self.bottleneck_L1(buffer)     #16*2*16*16
        optical_flow_L1 = self.conv_L1_2(optical_flow_L1)    #16*2*16*16
        optical_flow_L1_upscaled = self.upsample(optical_flow_L1)  # 上采样16*2*32*32，双线性插值
        if self.is_training is True:
            x_L1_res = optical_flow_warp(torch.unsqueeze(x_L1[:, 0, :, :], dim=1), optical_flow_L1) - torch.unsqueeze(
                x_L1[:, 1, :, :], dim=1)


        # Level 2
        x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1), optical_flow_L1_upscaled)
        # print(x_L2.size())  #16*1*32*32
        x_L2_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L2
        # print(x_L2_res.size())   #16*1*32*32
        x_L2 = torch.cat((x, x_L2, x_L2_res, optical_flow_L1_upscaled), 1)     #16*6*32*32

        input_L2 = self.conv_L2_1(x_L2)       #16*32*32*32
        # print(input_L2.size())

        buffer_1 = self.RDB2_1(input_L2)
        buffer_2 = self.RDB2_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L2 = self.bottleneck_L2(buffer)
        optical_flow_L2 = self.conv_L2_2(optical_flow_L2)
        optical_flow_L2 = optical_flow_L2 + optical_flow_L1_upscaled
        if self.is_training is True:
            x_L2_res = optical_flow_warp(torch.unsqueeze(x_L2[:, 0, :, :], dim=1), optical_flow_L2) - torch.unsqueeze(
                x_L2[:, 1, :, :], dim=1)

        # Level 3
        x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1), optical_flow_L2)
        x_L3_res = torch.unsqueeze(x[:, 1, :, :], dim=1) - x_L3
        x_L3 = torch.cat((x, x_L3, x_L3_res, optical_flow_L2), 1)
        input_L3 = self.conv_L3_1(x_L3)
        buffer_1 = self.RDB3_1(input_L3)
        buffer_2 = self.RDB3_2(buffer_1)
        buffer = torch.cat((buffer_1, buffer_2), 1)
        optical_flow_L3 = self.bottleneck_L3(buffer)
        optical_flow_L3 = self.conv_L3_2(optical_flow_L3)
        optical_flow_L3 = self.shuffle(optical_flow_L3) + self.final_upsample(optical_flow_L2)  # *4
        if self.is_training is False:
            return optical_flow_L3
        if self.is_training is True:
            return x_L1_res, x_L2_res, optical_flow_L1, optical_flow_L2, optical_flow_L3


class SRnet(nn.Module):
    def __init__(self, upscale_factor, is_training):
        super(SRnet, self).__init__()
        self.conv = nn.Conv2d(35, 64, 3, 1, 1, bias=False)
        self.RDB_1 = RDB(5, 64, 32)      #5层密集层，输入通道64 输出通道32
        self.RDB_2 = RDB(5, 64, 32)
        self.RDB_3 = RDB(5, 64, 32)
        self.RDB_4 = RDB(5, 64, 32)
        self.RDB_5 = RDB(5, 64, 32)
        self.bottleneck = nn.Conv2d(384, upscale_factor ** 2, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(upscale_factor ** 2, upscale_factor ** 2, 3, 1, 1, bias=True)
        self.shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.is_training = is_training

    def forward(self, x):
        input = self.conv(x)
        buffer_1 = self.RDB_1(input)
        buffer_2 = self.RDB_2(buffer_1)
        buffer_3 = self.RDB_3(buffer_2)
        buffer_4 = self.RDB_4(buffer_3)
        buffer_5 = self.RDB_5(buffer_4)
        output = torch.cat((buffer_1, buffer_2, buffer_3, buffer_4, buffer_5, input), 1)
        output = self.bottleneck(output)
        output = self.conv_2(output)
        output = self.shuffle(output)
        return output


class SOFVSR(nn.Module):
    def __init__(self, upscale_factor, is_training=True):
        super(SOFVSR, self).__init__()
        self.upscale_factor = upscale_factor
        self.is_training = is_training
        self.OFRnet = OFRnet(upscale_factor=upscale_factor, is_training=is_training)
        self.SRnet = SRnet(upscale_factor=upscale_factor, is_training=is_training)

    #输入是LR：LR0,LR1,LR2的组合,size:16,3,32,32
    def forward(self, x):
        # print("x",x.size())  [16, 3, 32, 32]
        input_01 = torch.cat((torch.unsqueeze(x[:, 0, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)  #[16, 2, 32, 32]
        input_21 = torch.cat((torch.unsqueeze(x[:, 2, :, :], dim=1), torch.unsqueeze(x[:, 1, :, :], dim=1)), 1)


        #将input_01,input_21输入OFRnet模型16*2*32*32
        if self.is_training is False:
            flow_01_L3 = self.OFRnet(input_01)
            flow_21_L3 = self.OFRnet(input_21)

        if self.is_training is True:
            res_01_L1, res_01_L2, flow_01_L1, flow_01_L2, flow_01_L3 = self.OFRnet(input_01)
            res_21_L1, res_21_L2, flow_21_L1, flow_21_L2, flow_21_L3 = self.OFRnet(input_21)


        draft_cube = x
        for i in range(self.upscale_factor):
            for j in range(self.upscale_factor):
                draft_01 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], dim=1),
                                             flow_01_L3[:, :, i::self.upscale_factor,
                                             j::self.upscale_factor] / self.upscale_factor)
                draft_21 = optical_flow_warp(torch.unsqueeze(x[:, 2, :, :], dim=1),
                                             flow_21_L3[:, :, i::self.upscale_factor,
                                             j::self.upscale_factor] / self.upscale_factor)
                draft_cube = torch.cat((draft_cube, draft_01, draft_21), 1)



        output = self.SRnet(draft_cube)
        if self.is_training is False:
            return torch.squeeze(output)
        if self.is_training is True:
            return (res_01_L1, res_01_L2, flow_01_L1, flow_01_L2, flow_01_L3), \
                   (res_21_L1, res_21_L2, flow_21_L1, flow_21_L2, flow_21_L3), output
