import numpy as np
import scipy.ndimage
# from scipy.io import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
import cv2
from criterion.pytorch_ssim import SSIM
from data_utils import rgb2ycbcr
import math
import torch
import PIL.Image as Image
from skimage.metrics import structural_similarity as ssim

def psnr1(img1, img2):
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr1 = 20 * math.log10(255 / math.sqrt(mse))
    return psnr1


def psnr2(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1e-10:
        return 100
    psnr2 = 20 * math.log10(1 / math.sqrt(mse))
    return psnr2
#
# if __name__ == '__main__':
#
#     hr_dir = "D:\CODE\VSR\\data\\test1\\autumn\\hr"
#     sr_dir = "D:\CODE\VSR\\results"
#     metric_dir = "D:\CODE\VSR\\metrics"
#     video_name = 'autumn'
#     # video_name = 'calendar'
#     print(video_name)
#     psnr_video = []                     #存放每一帧的结果
#     ssim_video = []
#     compute_ssim = SSIM()
#     for idx_frame in range(2, 48):
#         print(idx_frame)
#         print(hr_dir + '/hr_' + str(idx_frame).rjust(2,'0') + '.png')
#         print(sr_dir + '/' + video_name +'\\'+ 'sr_'+ str(idx_frame).rjust(2, '0') + '.png')
#         #
#         # img_hr = cv2.imread(hr_dir + '/hr' + str(idx_frame) + '.png',flags=1)
#         # # img_hr = cv2.imread(hr_dir + '/hr_' + str(idx_frame).rjust(2,'0') + '.png')
#         # img_sr = cv2.imread(sr_dir +'/' + video_name +'\\'+ 'sr_'+ str(idx_frame).rjust(2, '0') + '.png',flags=1)
#         #
#         # img_hr = torch.tensor(img_hr)
#         # img_sr = torch.tensor(img_sr)
#
#         img_hr = np.array(Image.open(hr_dir + '/hr' + str(idx_frame) + '.png'))
#         img_sr = np.array(Image.open(sr_dir + '/' + video_name + '\\' + 'sr_' + str(idx_frame).rjust(2, '0') + '.png'))
#
#
#         ssim_video.append(ssim(img_hr,img_sr,multichannel=True))
#
#     psnr_video = np.array(psnr_video)
#     ssim_video = np.array(ssim_video)
#     psnr = np.mean(psnr_video)
#     ssim = np.mean(ssim_video)
#     #
#
#     with open(metric_dir + '/results.txt', 'a') as f:  # 设置文件对象
#         print('%s -------Mean PSNR:   %0.4f' % (video_name, psnr), file=f)
#         print('%s -------Mean SSIM:   %0.4f' % (video_name, ssim), file=f)





import cv2
import numpy as np
def ssim(img1, img2):
  C1 = (0.01 * 255)**2
  C2 = (0.03 * 255)**2
  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1**2
  mu2_sq = mu2**2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()
def calculate_ssim(img1, img2):
  '''calculate SSIM
  the same outputs as MATLAB's
  img1, img2: [0, 255]
  '''
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return ssim(img1, img2)
  elif img1.ndim == 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')

if __name__ == '__main__':

    hr_dir = "D:\CODE\VSR\\data\\test1\\autumn\\hr"
    sr_dir = "D:\CODE\VSR\\results"
    metric_dir = "D:\CODE\VSR\\metrics"
    video_name = 'autumn'
    # video_name = 'calendar'
    print(video_name)
    psnr_video = []                     #存放每一帧的结果
    ssim_video = []
    compute_ssim = SSIM()
    for idx_frame in range(2, 48):
        print(idx_frame)
        print(hr_dir + '/hr_' + str(idx_frame).rjust(2,'0') + '.png')
        print(sr_dir + '/' + video_name +'\\'+ 'sr_'+ str(idx_frame).rjust(2, '0') + '.png')

        img_hr = cv2.imread(hr_dir + '/hr' + str(idx_frame) + '.png',flags=0)
        # img_hr = cv2.imread(hr_dir + '/hr_' + str(idx_frame).rjust(2,'0') + '.png')
        img_sr = cv2.imread(sr_dir +'/' + video_name +'\\'+ 'sr_'+ str(idx_frame).rjust(2, '0') + '.png',flags=0)

        ssim_video.append(calculate_ssim(img_hr,img_sr))
    psnr_video = np.array(psnr_video)
    ssim_video = np.array(ssim_video)
    psnr = np.mean(psnr_video)
    ssim = np.mean(ssim_video)

    with open(metric_dir + '/results.txt', 'a') as f:  # 设置文件对象
        print('%s -------Mean PSNR:   %0.4f' % (video_name, psnr), file=f)
        print('%s -------Mean SSIM:   %0.4f' % (video_name, ssim), file=f)

