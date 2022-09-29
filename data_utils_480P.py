import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data.dataset import Dataset
import math
import argparse
import random
from torch.utils.data import DataLoader
from torch.autograd import Variable

class TrainsetLoader(Dataset):

    def __init__(self, trainset_dir, upscale_factor, patch_size, n_iters, frame_num):
        super(TrainsetLoader).__init__()
        self.trainset_dir = trainset_dir
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.n_iters = n_iters
        self.video_list = os.listdir(trainset_dir)
        self.frame_num = frame_num

    def __getitem__(self, idx):
        idx_video = random.randint(0, self.video_list.__len__() - 1)  # 随机产生一个训练样本
        lr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/lr_x' + str(16)
        hr_dir = self.trainset_dir + '/' + self.video_list[idx_video] + '/lr_x4'
        l = len(self.video_list[idx_video])
        idx_frame = random.randint(0, l - 3)
        num = self.frame_num

        # read HR & LR frames，num = 5
        HR = []
        LR = []
        for id in range(0, num):
            lr = Image.open(lr_dir + '/lr' + str(idx_frame + id) + '.png')
            lr = np.array(lr, dtype=np.float32) / 255.0
            lr_y = rgb2y(lr)
            LR.append(lr_y)
            hr = Image.open(hr_dir + '/lr' + str(idx_frame + id) + '.png')
            hr = np.array(hr, dtype=np.float32) / 255.0
            hr_y = rgb2y(hr)
            HR.append(hr_y)

        HR_new = []
        LR_new = []
        for hr, lr in zip(HR, LR):
            hr, lr = random_crop(hr, lr, self.patch_size, self.upscale_factor)
            HR_new.append(hr)
            LR_new.append(lr)

        for id in range(0, len(HR_new)):
            HR_new[id] = HR_new[id][:, :, np.newaxis]
            LR_new[id] = LR_new[id][:, :, np.newaxis]

        if num == 3:
            HR_T = np.concatenate((HR_new[0], HR_new[1], HR_new[2]), axis=2)
            LR_T = np.concatenate((LR_new[0], LR_new[1], LR_new[2]), axis=2)
        if num == 5:
            HR_T = np.concatenate((HR_new[0], HR_new[1], HR_new[2], HR_new[3], HR_new[4]), axis=2)
            LR_T = np.concatenate((LR_new[0], LR_new[1], LR_new[2], LR_new[3], LR_new[4]), axis=2)
        if num == 7:
            HR_T = np.concatenate((HR_new[0], HR_new[1], HR_new[2], HR_new[3], HR_new[4], HR_new[5], HR_new[6]),
                                  axis=2)
            LR_T = np.concatenate((LR_new[0], LR_new[1], LR_new[2], LR_new[3], LR_new[4], LR_new[5], LR_new[6]), axis=2)

        # data augmentation
        LR_T, HR_T = augmentation()(LR_T, HR_T)  # 64*64  256*256

        # print(HR_T.shape)

        return toTensor(LR_T), toTensor(HR_T)

    def __len__(self):
        return self.n_iters


class TestsetLoader(Dataset):

    def __init__(self, dataset_dir,upscale_factor, frame_num):
        super(TestsetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = 4
        self.frame_list = os.listdir(self.dataset_dir + '/lr_x' + str(16))
        self.frame_num = frame_num

    def __getitem__(self, idx):
        dir = self.dataset_dir + '/lr_x' + str(16)
        num = self.frame_num
        LR_list = []

        LR0 = Image.open(dir + '/' + 'lr' + str(idx) + '.png')
        LR1 = Image.open(dir + '/' + 'lr' + str(idx + 1) + '.png')
        LR2 = Image.open(dir + '/' + 'lr' + str(idx + 2) + '.png')
        LR3 = Image.open(dir + '/' + 'lr' + str(idx + 3) + '.png')
        LR4 = Image.open(dir + '/' + 'lr' + str(idx + 4) + '.png')

        W, H = LR2.size

        # H and W should be divisible by 2
        W = int(W // 2) * 2
        H = int(H // 2) * 2
        LR0 = LR0.crop([0, 0, W, H])
        LR1 = LR1.crop([0, 0, W, H])
        LR2 = LR2.crop([0, 0, W, H])
        LR3 = LR3.crop([0, 0, W, H])
        LR4 = LR4.crop([0, 0, W, H])


        LR2_bicubic = LR2.resize((W * self.upscale_factor, H * self.upscale_factor), Image.BICUBIC)
        LR2_bicubic = np.array(LR2_bicubic, dtype=np.float32) / 255.0

        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        LR1 = np.array(LR1, dtype=np.float32) / 255.0
        LR2 = np.array(LR2, dtype=np.float32) / 255.0
        LR3 = np.array(LR3, dtype=np.float32) / 255.0
        LR4 = np.array(LR4, dtype=np.float32) / 255.0


        # extract Y channel for LR inputs
        LR0_y, _, _ = rgb2ycbcr(LR0)
        LR1_y, _, _ = rgb2ycbcr(LR1)
        LR2_y, _, _ = rgb2ycbcr(LR2)
        LR3_y, _, _ = rgb2ycbcr(LR3)
        LR4_y, _, _ = rgb2ycbcr(LR4)


        LR0_y = LR0_y[:, :, np.newaxis]
        LR1_y = LR1_y[:, :, np.newaxis]
        LR2_y = LR2_y[:, :, np.newaxis]
        LR3_y = LR3_y[:, :, np.newaxis]
        LR4_y = LR4_y[:, :, np.newaxis]
        LR = np.concatenate((LR0_y, LR1_y, LR2_y,LR3_y,LR4_y), axis=2)

        LR = toTensor(LR)

        # generate Cr, Cb channels using bicubic interpolation
        _, SR_cb, SR_cr = rgb2ycbcr(LR2_bicubic)

        return LR, SR_cb, SR_cr

        # for i in range(0, num):
        #
        #     lr = Image.open(dir + '/' + 'lr' + str(idx + i + 1) + '.png')
        #     # lr = Image.open(dir + '/' + 'lr_' + str(idx + i + 1).rjust(2, '0') + '.png')
        #     W, H = lr.size
        #     W = int(W // 2) * 2
        #     H = int(H // 2) * 2
        #     lr = lr.crop([0, 0, W, H])
        #     LR_list.append(lr)
        #
        # # 对中间的帧进行插值
        # H, W = LR_list[0].size
        # W = int(W // 2) * 2
        # H = int(H // 2) * 2
        #
        # LR_mid_bicubic = LR_list[int(num / 2)].resize((H * self.upscale_factor, W * self.upscale_factor), Image.BICUBIC)
        # LR_mid_bicubic = np.array(LR_mid_bicubic, dtype=np.float32) / 255.0
        #
        # LR_new = []
        #
        # for LR in LR_list:
        #     LR = np.array(LR, dtype=np.float32) / 255.0
        #     LR_y, _, _ = rgb2ycbcr(LR)
        #     LR_y = LR_y[:, :, np.newaxis]
        #     LR_new.append(LR_y)
        #
        # if num == 3:
        #     LR_T = np.concatenate((LR_new[0], LR_new[1], LR_new[2]), axis=2)
        # if num == 5:
        #     LR_T = np.concatenate((LR_new[0], LR_new[1], LR_new[2], LR_new[3], LR_new[4]), axis=2)
        # if num == 7:
        #     LR_T = np.concatenate((LR_new[0], LR_new[1], LR_new[2], LR_new[3], LR_new[4], LR_new[5], LR_new[6]), axis=2)
        #
        # LR = toTensor(LR_T)
        # # generate Cr, Cb channels using bicubic interpolation
        # _, SR_cb, SR_cr = rgb2ycbcr(LR_mid_bicubic)  # torch.Size([3, 540, 960]) (3840, 2160) (3840, 2160)
        # # print("LR,SR_cb,SR_cr",LR.size(),SR_cb.shape,SR_cr.shape)
        # return LR, SR_cb, SR_cr

    def __len__(self):
        return self.frame_list.__len__() - 4     #48


class augmentation(object):
    def __call__(self, input, target):

        if random.random() < 0.5:
            input = input[:, ::-1, :]
            target = target[:, ::-1, :]
        if random.random() < 0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random() < 0.5:
            input = input.transpose(1, 0, 2)
            target = target.transpose(1, 0, 2)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)

#patch_size = 64
def random_crop(HR, LR, patch_size_lr, upscale_factor):
    h_hr, w_hr = HR.shape  # 540*960

    h_lr = h_hr // upscale_factor  # 135
    w_lr = w_hr // upscale_factor  # 240

    idx_h = random.randint(10, h_lr - patch_size_lr - 10)  # [0,103]
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)  # [0,208]

    # HR的随机起始点,h_end_hr - h_start_hr = 64*4=256
    h_start_hr = (idx_h - 1) * upscale_factor
    h_end_hr = (idx_h - 1 + patch_size_lr) * upscale_factor
    w_start_hr = (idx_w - 1) * upscale_factor
    w_end_hr = (idx_w - 1 + patch_size_lr) * upscale_factor

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    # h_end_hr - h_start_hr = 256; w_end_hr - w_start_hr = 256
    # h_end_lr - h_start_lr = 64; w_end_lr - w_start_lr = 64

    HR = HR[h_start_hr:h_end_hr, w_start_hr:w_end_hr]
    LR = LR[h_start_lr:h_end_lr, w_start_lr:w_end_lr]

    return HR, LR


def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img


def rgb2ycbcr(img_rgb):
    ## the range of img_rgb should be (0, 1)
    img_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    img_cb = -0.148 * img_rgb[:, :, 0] - 0.291 * img_rgb[:, :, 1] + 0.439 * img_rgb[:, :, 2] + 128 / 255.0
    img_cr = 0.439 * img_rgb[:, :, 0] - 0.368 * img_rgb[:, :, 1] - 0.071 * img_rgb[:, :, 2] + 128 / 255.0
    return img_y, img_cb, img_cr


def ycbcr2rgb(img_ycbcr):
    ## the range of img_ycbcr should be (0, 1)
    img_r = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 1.596 * (img_ycbcr[:, :, 2] - 128 / 255.0)
    img_g = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) - 0.392 * (img_ycbcr[:, :, 1] - 128 / 255.0) - 0.813 * (
            img_ycbcr[:, :, 2] - 128 / 255.0)
    img_b = 1.164 * (img_ycbcr[:, :, 0] - 16 / 255.0) + 2.017 * (img_ycbcr[:, :, 1] - 128 / 255.0)
    img_r = img_r[:, :, np.newaxis]
    img_g = img_g[:, :, np.newaxis]
    img_b = img_b[:, :, np.newaxis]
    img_rgb = np.concatenate((img_r, img_g, img_b), 2)
    return img_rgb


# rgb2ycbcr中只取y通道
def rgb2y(img_rgb):
    ## the range of img_rgb should be (0, 1)
    image_y = 0.257 * img_rgb[:, :, 0] + 0.504 * img_rgb[:, :, 1] + 0.098 * img_rgb[:, :, 2] + 16 / 255.0
    return image_y
