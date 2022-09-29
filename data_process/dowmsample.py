import cv2
import os.path
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch
from torch.autograd import Variable
def countFile(dir):
    # 输入文件夹
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp

def upsample(image_tensor, width, height, mode):
    # mode可用：最近邻插值"nearest"，双线性插值"bilinear"，双三次插值"bicubic"，如mode="nearest"
    image_upsample_tensor = nn.functional.interpolate(image_tensor.unsqueeze_(0), size=[width, height], mode=mode)
    image_upsample_tensor.squeeze_(0)
    # 将数据归一到正常范围，尺寸改变过程可能数值范围溢出，此处浮点数据位[0,1]，整数数据为[0,255]
    image_upsample_tensor = image_upsample_tensor.clamp(0, 1)
    return image_upsample_tensor


# 下采用函数，输入数据格式示例：tensor维度[3,300,300]，即3通道RGB，大小300×300，当然4通道图像也能做
def downsample(image_tensor, width, height,mode):
    image_downsample_tensor = nn.functional.interpolate(image_tensor.unsqueeze_(0), size=[width, height],mode= mode)
    image_downsample_tensor.squeeze_(0)
    # 将数据归一到正常范围，尺寸改变过程可能数值范围溢出，此处浮点数据位[0,1]，整数数据为[0,255]
    image_downsample_tensor = image_downsample_tensor.clamp(0, 1)
    return image_downsample_tensor



# filenum
if __name__ == '__main__':

    dir = "../data/test"
    dir_list = os.listdir(dir)
    filenum = countFile(dir)
    n = 4

    num = 0  # 处理图片计数
    for dir_name in dir_list:
        print(dir_name)
        path0 = dir + '/'+ dir_name + '/' + 'hr'
        path = dir + '/'+ dir_name + '/' + 'lr_x'+ str(n)

        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, filenum):
            ########################################################
            if i == 32:
                break
            filename = path0 + "/"+ "hr" + str(i) + ".png"
            original_image = Image.open(filename)

            w,h = original_image.size
            img_1 = original_image.resize((w//4, h//4), Image.BICUBIC)
            img_1.save(path +'/'+ "lr" +  str(i) + ".png")

            num = num + 1
            print("正在为第" + str(num) + "图片下采样......")


