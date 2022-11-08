import cv2
import math
import numpy as np
from data_utils import toTensor, rgb2ycbcr
import PIL.Image as Image
import array
import os


# from skimage.metrics import structural_similarity as ssim

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


import numpy
import scipy.ndimage
# from scipy.io import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
import cv2

'''
The function to compute SSIM
@param param: img_mat_1 1st 2D matrix
@param param: img_mat_2 2nd 2D matrix
'''


def compute_ssim(img_mat_1, img_mat_2):
    # Variables for Gaussian kernel definition
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    # Convert image matrices to double precision (like in the Matlab version)
    img_mat_1 = img_mat_1.astype(numpy.float64)
    img_mat_2 = img_mat_2.astype(numpy.float64)

    # Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2

    # Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

    # Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

    # Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

    # Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

    # Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

    # c1/c2 constants
    # First use: manual fitting
    # c_1 = 6.5025
    # c_2 = 58.5225

    # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2

    # Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    # Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    # SSIM
    ssim_map = num_ssim / den_ssim
    index = numpy.average(ssim_map)

    return index


if __name__ == '__main__':
    dir = 'data/X4K1000FPS'
    video_list = sorted(os.listdir(dir))
    
    psnr_list = []
    ssim_list = []
    for video_name in video_list:
        hr_dir = dir +'/' + video_name + "/hr"
        sr_dir = "results_log2_600K_X4K1000FPS"
        metric_dir = "metrics"
    # video_name = 'calendar'
        print(video_name)
        psnr_video = []                     
        ssim_video = []
        for idx_frame in range(2, 30):
            print(idx_frame)
            img_hr = cv2.imread(hr_dir + '/' + str(idx_frame).rjust(4,'0') + '.png')
        # img_hr = cv2.imread(hr_dir + '/hr_' + str(idx_frame).rjust(2,'0') + '.png')
            img_sr = cv2.imread(sr_dir +'/' + video_name +'/' + 'sr_'+ str(idx_frame).rjust(2, '0') + '.png')

            imag1, _, _ = rgb2ycbcr(img_hr)
            imag2, _, _ = rgb2ycbcr(img_sr)
            psnr_video.append(psnr1(imag1, imag2))
            ssim_video.append(compute_ssim(imag1, imag2))

        psnr_video = np.array(psnr_video)
        ssim_video = np.array(ssim_video)
        psnr = np.mean(psnr_video)
        psnr_list.append(psnr)
        
        ssim = np.mean(ssim_video)
        ssim_list.append(ssim)

        print('%s -------Mean PSNR:   %0.4f' % (video_name, psnr))
        print('%s -------Mean SSIM:   %0.4f' % (video_name, ssim))


        
        with open(metric_dir + '/600K_X4K1000FPS.txt', 'a') as f:  
            print('%s -------Mean PSNR:   %0.4f' % (video_name, psnr), file=f)
            print('%s -------Mean SSIM:   %0.4f' % (video_name, ssim), file=f)
    psnr_average = np.mean(np.array(psnr_list))
    ssim_average = np.mean(np.array(ssim_list))

    with open(metric_dir +'/600K_X4K1000FPS.txt','a') as f:
        print('Average PSNR:    %0.4f' % psnr_average , file = f)
        print('Average SSIM:    %0.4f' % ssim_average, file =f)



