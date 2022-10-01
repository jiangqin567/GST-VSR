# %%

import os
import glob
import scipy.misc as misc

# dir_path == directory path of HR images
# dir_path = "test2/000003"   # <<== input your directory
# dir_path += "/*"
#
# save_format = "png"
# path = dir_path
#
# file_list = glob.glob(path)
# print(file_list)
# a = os.path.splitext(file_list[0])[-2]

# print(os.path.split(a))
import numpy as np
from PIL import Image
'''
spcipy.misc has no attribution imresize
replaced by scipy_misc_imresize
'''


def scipy_misc_imresize(arr, size, interp='bilinear', mode=None):
	im = Image.fromarray(arr, mode=mode)
	ts = type(size)
	if np.issubdtype(ts, np.signedinteger):
		percent = size / 100.0
		size = tuple((np.array(im.size)*percent).astype(int))
	elif np.issubdtype(type(size), np.floating):
		size = tuple((np.array(im.size)*size).astype(int))
	else:
		size = (size[1], size[0])
	func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
	imnew = im.resize(size, resample=func[interp]) # 调用PIL库中的resize函数
	return np.array(imnew)


'''
Generating BD dataset
'''

import numpy as np
import cv2
# import scipy.misc as misc
import re
dir_path = 'train2'

file_list = os.listdir(dir_path)
for file in file_list:
    print(file)

    path = os.path.join(dir_path,file)
    # print(path)
    save_BD_original = path + '/' + "BD_original"
    save_BD_X4 = path + '/' + "BD_X4"
    # print(save_BD_original)
    if not os.path.exists(save_BD_original):
        os.mkdir(save_BD_original)
    if not os.path.exists(save_BD_X4):
        os.mkdir(save_BD_X4)
    hr_file_path = path + '/' + 'hr'
    hr_list = os.listdir(hr_file_path)
    for hr_img in hr_list:
        # print(hr_img)
        hr_img_path = os.path.join(hr_file_path, hr_img)

        #####rename file name
        # hr_new = "hr_"+ str(re.findall("\d+",hr_img)[0]) + '.png'
        # hr_new_path = os.path.join(hr_file_path,hr_new)
        # print(hr_img_path)
        # print(hr_new_path)
        # os.rename(hr_img_path,hr_new_path)



        img = cv2.imread(hr_img_path)
        blured_img = cv2.GaussianBlur(img, (7, 7), 1.6)
        blured_img_name ='blur_' + hr_img.split('_')[1]
        # print(blured_img_name)
        blured_img_path = save_BD_original + '/' + blured_img_name
        print(blured_img_path)
        cv2.imwrite( blured_img_path, blured_img)
        #
        # lr = misc.imresize(blured_img, size=0.25, interp="bicubic")
        lr = scipy_misc_imresize(blured_img, size=0.25, interp="bicubic")
        lr_name = blured_img_name.split('.')[0] + '_' + 'bicubic.png'
        # print(lr_name)
        lr_path = save_BD_X4 + '/' + lr_name
        print(lr_path)
        cv2.imwrite(lr_path, lr)



# for file in file_list:
#     file_name = os.path.split(file)[-1]
#     img = cv2.imread(file)
#     blured_img = cv2.GaussianBlur(img, (7, 7), 1.6)
#     cv2.imwrite("BD_original/" + file_name, blured_img)

# %%
#
# path = "./BD_original/*"
#
# file_list = glob.glob(path)
# print(file_list)

# %%

# try:
#     os.mkdir("BDx2")
#     os.mkdir("BDx4")
#
# except:
#     print("directory was already made!")
#
# # %%
#
# import scipy.misc as misc
#
# for file in file_list:
#     file_name = os.path.splitext(file)[-2]
#     file_name = os.path.split(file_name)[-1]
#     img = misc.imread(file)
#     lr = misc.imresize(img, size=0.5, interp="bicubic")
#     misc.imsave("BDx2/" + file_name + ".{}".format(save_format), lr, format=save_format)
#
#     lr = misc.imresize(img, size=0.25, interp="bicubic")
#     misc.imsave("BDx4/" + file_name + ".{}".format(save_format), lr, format=save_format)
