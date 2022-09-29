import os
import sys
import cv2
import numpy as np
import re

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_util import imresize_np
except ImportError:
    pass

def generate_mod_LR_bic(up_scale, sourcedir, savedir, format='.png', crop=[0,0,0,0]): # crop l r t b
    saveLRpath = savedir
    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    filepaths = [f for f in os.listdir(sourcedir) if f.endswith(format)]

    num_files = len(filepaths)
    # print(num_files)
    # prepare data with augementation
    for i in range(num_files):
        filename = filepaths[i]
        image = cv2.imread(os.path.join(sourcedir, filename))
        image = image[0 + crop[2]: image.shape[0] - crop[3], 0 + crop[0]: image.shape[1] - crop[1], :]

        width = int(np.floor(image.shape[1] / up_scale))
        height = int(np.floor(image.shape[0] / up_scale))
        # # modcrop
        if len(image.shape) == 3:
            image_HR = image[0:up_scale * height, 0:up_scale * width, :]
        else:
            image_HR = image[0:up_scale * height, 0:up_scale * width]
        # LR
        image_LR = imresize_np(image_HR, 1 / up_scale, True)
        lr_name = re.findall(r"\d*",filename)[2]
        lr_name = ''.join(lr_name)
        # cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        saveLRname = saveLRpath + '/' +  'lr' + lr_name + '.png'
        # saveLRname = saveLRpath + '/' + filename
        print(saveLRname)
        cv2.imwrite(saveLRname, image_LR)

