# %%

import os
import glob
import scipy.misc as misc

# dir_path == directory path of HR images
dir_path = "data/data/test2/000003/hr"   # <<== input your directory
dir_path += "/*"

save_format = "png"
path = dir_path

file_list = glob.glob(path)
print(file_list)
# a = os.path.splitext(file_list[0])[-2]
# print(os.path.split(a))


'''
Generating BD dataset
'''

import numpy as np
#
# path = dir_path
#
# file_list = glob.glob(path)
#
# print(os.path.split(file_list[0])[-1])

# %%

# try:
#     os.mkdir("BD_original")
# except:
#     print("directory was already made!")
#
# # %%
#
# import cv2
# import os
#
# for file in file_list:
#     file_name = os.path.split(file)[-1]
#     img = cv2.imread(file)
#     blured_img = cv2.GaussianBlur(img, (7, 7), 1.6)
#     cv2.imwrite("BD_original/" + file_name, blured_img)
#
# # %%
#
# path = "./BD_original/*"
#
# file_list = glob.glob(path)
# print(file_list)

# %%

try:
    os.mkdir("BDx2")
    os.mkdir("BDx4")

except:
    print("directory was already made!")

# %%

import scipy.misc as misc

for file in file_list:
    file_name = os.path.splitext(file)[-2]
    file_name = os.path.split(file_name)[-1]
    img = misc.imread(file)
    lr = misc.imresize(img, size=0.5, interp="bicubic")
    misc.imsave("BDx2/" + file_name + ".{}".format(save_format), lr, format=save_format)

    lr = misc.imresize(img, size=0.25, interp="bicubic")
    misc.imsave("BDx4/" + file_name + ".{}".format(save_format), lr, format=save_format)
