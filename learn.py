
import torchvision.transforms as transforms
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image

def optical_flow_warp(image, image_optical_flow):
    """
    Arguments
        image_ref: reference images tensor, (b, c, h, w)
        image_optical_flow: optical flow to image_ref (b, 2, h, w)
    """
    # print(image_optical_flow.size())   #16*2*32*32
    b, _, h, w = image.size()

    grid = np.meshgrid(range(w), range(h))
    # print(grid)
    grid = np.stack(grid, axis=-1).astype(np.float64)

    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) - 1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) - 1

    grid = grid.transpose(2, 0, 1)
    grid = np.tile(grid, (b, 1, 1, 1))


    grid = Variable(torch.Tensor(grid))
    if image_optical_flow.is_cuda == True:
        grid = grid.cuda()

    flow_0 = torch.unsqueeze(image_optical_flow[:, 0, :, :]  * 3 / (w - 1), dim=1)
    print("*********************************************************************")

    flow_1 = torch.unsqueeze(image_optical_flow[:, 1, :, :] * 4 / (h - 1), dim=1)
    print("*********************************************************************")
    # print(flow_1)
    grid = grid + torch.cat((flow_0, flow_1), 1)
    print("*********************************************************************")
    # print(grid)
    # print(grid.size())   #16*2*16*16
    grid = grid.transpose(1, 2)
    grid = grid.transpose(3, 2)
    print("*********************************************************************")
    # print(grid.dtype)

    output = F.grid_sample(image, grid, padding_mode='border',align_corners=False)  #16*1*32*32
    print(output.size())
    # # return output


a = [[[[0,1,2,3,1],
       [4,5,6,7,1],
       [8,9,10,11,1],
       [13,15,17,15,19]],

      [[12,13,14,15,1],
       [16,17,18,19,1],
       [20,21,22,23,1],
       [118,15,18,15,19]]],

      [[[24,25,26,27,1],
      [28,29,30,31,1],
      [32,33,34,35,1],
        [13,13,13,15,19]],

        [[36,37,38,39,1],
         [40,41,42,43,1],
         [44,45,46,47,1],
         [32,15,17,15,19]]]]

b = [[[[1,1,2,3,1],
       [1,5,6,7,1],
       [1,9,10,11,1],
       [1,15,17,15,19]],

      [[1,13,14,15,1],
       [1,17,18,19,1],
       [1,21,22,23,1],
       [1,15,18,15,19]]],

      [[[1,25,26,27,1],
      [1,29,30,31,1],
      [1,33,34,35,1],
        [1,13,13,15,19]],

        [[1,37,38,39,1],
         [1,41,42,43,1],
         [1,45,46,47,1],
         [1,15,17,15,19]]]]

a =torch.tensor(a)
b = torch.tensor(b)

optical_flow_warp(a,b)

#
# list = [1,2,3,4]
# for i in enumerate(list):
#     print()
