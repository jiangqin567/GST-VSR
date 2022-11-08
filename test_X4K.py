import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_utils_X4K import TestsetLoader, ycbcr2rgb, rgb2y
import numpy as np
from torchvision.transforms import ToPILImage
import os
import argparse
from motion_compensation_revise import VSR
import torch.nn as nn
import math
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default='data/X4K1000FPS')
    parser.add_argument("--video_list",type=str, default =os.listdir('data/X4K1000FPS'))
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--chop_forward', type=bool, default=False)
    parser.add_argument('--frame_num', type=int, default=5)
    return parser.parse_args()


def main(cfg):
    video_list = cfg.video_list
    upscale_factor = cfg.upscale_factor
    use_gpu = cfg.gpu_mode
    frame_num = cfg.frame_num
    
    for video_name in video_list:
        print(video_name)
    
        test_set = TestsetLoader('data/X4K1000FPS/' + video_name, upscale_factor,frame_num)
        test_loader = DataLoader(test_set, num_workers=1, batch_size=1, shuffle=False)

        net = VSR(upscale_factor=upscale_factor,frame_num=frame_num)
        if use_gpu:
            net.cuda()
            net = nn.DataParallel(net,device_ids=[0,1,2,3,4,5,6,7])
    # print(list(net.parameters()))
        ckpt = torch.load('log2/BI_x4_iter600000.pth')
        total = sum([param.nelement() for param in net.parameters()])
        print("Number of parameters:%.2fM" % (total / 1e6))
        net.load_state_dict(ckpt)
    # print(list(net.parameters()))
        if use_gpu:
            net.cuda()

        with torch.no_grad():
            for idx_iter, (LR_y_cube, SR_cb, SR_cr) in enumerate(test_loader):   #0~49

            # print(idx_iter)
            # print(LR_y_cube.shape)
                LR_y_cube = Variable(LR_y_cube)
                if use_gpu:
                    LR_y_cube = LR_y_cube.cuda()
                # if cfg.chop_forward:
                #     # crop borders to ensure each patch can be divisible by 2
                #     _, _, h, w = LR_y_cube.size()
                #     h = int(h // 16) * 16
                #     w = int(w // 16) * 16
                #     LR_y_cube = LR_y_cube[:, :, :h, :w]
                #     SR_cb = SR_cb[:, :h * upscale_factor, :w * upscale_factor]
                #     SR_cr = SR_cr[:, :h * upscale_factor, :w * upscale_factor]
                #     SR_y = chop_forward(LR_y_cube, net, cfg.upscale_factor)
                # else:
                #     SR_y = net(LR_y_cube)
            # else:

                SR_y,_,_ = net(LR_y_cube)
                SR_y = torch.squeeze(SR_y)
            # print(SR_y.shape)

                SR_y = np.array(SR_y.data.cpu())
                SR_y = SR_y[np.newaxis, :, :]
            # print(SR_y.shape, SR_cr.shape, SR_cb.shape)

                SR_ycbcr = np.concatenate((SR_y, SR_cb, SR_cr), axis=0).transpose(1, 2, 0)
                SR_rgb = ycbcr2rgb(SR_ycbcr) * 255.0
                SR_rgb = np.clip(SR_rgb, 0, 255)
                SR_rgb = ToPILImage()(SR_rgb.astype(np.uint8))
                results_dir = 'results_log2_600K_X4K1000FPS'
                if not os.path.exists(results_dir):
                    os.mkdir(results_dir)

                if not os.path.exists(results_dir + '/' + video_name):
                    os.mkdir(results_dir+ '/' + video_name)
                print("saving*********************************************",idx_iter+2)
                SR_rgb.save(results_dir + '/' + video_name + '/sr_' + str(idx_iter + 2).rjust(2, '0') + '.png')

    

def chop_forward(x, model, scale, shave=16, min_size=5000, nGPUs=1):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            output_batch = model(input_batch)
            outputlist.append(output_batch.data)
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = Variable(x.data.new(h, w), volatile=True)
    output[0:h_half, 0:w_half] = outputlist[0][0:h_half, 0:w_half]
    output[0:h_half, w_half:w] = outputlist[1][0:h_half, (w_size - w + w_half):w_size]
    output[h_half:h, 0:w_half] = outputlist[2][(h_size - h + h_half):h_size, 0:w_half]
    output[h_half:h, w_half:w] = outputlist[3][(h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
