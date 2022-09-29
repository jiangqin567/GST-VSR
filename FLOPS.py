import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils import TestsetLoader, ycbcr2rgb, rgb2y
import numpy as np
from torchvision.transforms import ToPILImage
import os
import argparse
from motion_compensation_revise import VSR
import torch.nn as nn
import math
from thop import profile
from torchsummary import summary


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default='data/test')
    parser.add_argument("--video_list", type=str, default=os.listdir('data/test'))
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--chop_forward', type=bool, default=False)
    parser.add_argument('--frame_num', type=int, default=5)
    return parser.parse_args()


def main(cfg):
    video_dir = cfg.video_dir
    upscale_factor = cfg.upscale_factor
    use_gpu = cfg.gpu_mode
    frame_num = cfg.frame_num
    video_list = cfg.video_list

    for video in video_list:
        print(video)

        test_set = TestsetLoader('data/test/' + video, upscale_factor, frame_num)
        test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)

        net = VSR(upscale_factor=upscale_factor, frame_num=frame_num)
        input = torch.randn(1,5,64,64)
        flops, params = profile(net, inputs=(input,))
        print('flops:{}'.format(flops / 1e9))
        print('params:{}'.format(params / 1e6))
        # if use_gpu:
        #     net.cuda()
        #     net = nn.DataParallel(net)
        # ckpt = torch.load('log/BI_x4_iter92700.pth')
        # total = sum([param.nelement() for param in net.parameters()])
        # print("Number of parameters: %.2fM" % (total / 1e6))
        # net.load_state_dict(ckpt)
        # summary(net.to('gpu'),input_size=(1,5,540,960),batch_size=-1)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)