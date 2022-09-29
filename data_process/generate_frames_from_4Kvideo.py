import cv2
import numpy as np
from PIL import Image
import glob
import cv2
import os
import shutil

# Configuration
# ============================

videoFolder = '/home/fz/COPY/Video_4K'
frameFolder = '/home/fz/VSR/data/test1'


# train_txt = '/home/fz/Downloads/train.txt'
# # valid_txt = 'E:/Video_4K/valid.txt'
test_txt = '/home/fz/Downloads/test.txt'
#
# # Run
# # ============================
# #
# with open(train_txt) as f:
#     temp = f.readlines()
#     train_list = [v.strip() for v in temp]
#
#
# #
# #
# #
# # #
# # # with open(valid_txt) as f:
# # #     temp = f.readlines()
# # #     valid_list = [v.strip() for v in temp]
# # #
with open(test_txt) as f:
    temp = f.readlines()
    train_list = [v.strip() for v in temp]
# #
# # # mov_files = glob.glob(os.path.join(videoFolder, '*'))
#
mov_list = os.listdir(videoFolder)
# print(len(mov_list))

for i, mov in enumerate(mov_list):
    print(i)
    if mov.split('.')[0] in train_list:
        print(mov)
        mov_folder = frameFolder
        image_index = 0
        mov_path = os.path.join(videoFolder,mov)
        video = cv2.VideoCapture(mov_path)
        success, frame = video.read()
        frame = np.transpose(frame, (0, 1, 2))
        while success:
            # save_folder = os.path.join(mov_folder, mov.split('.')[0], "{:06d}".format(folder_index))
            # print(save_folder)
            save_folder = os.path.join(mov_folder, mov.split('.')[0])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            cv2.imwrite(os.path.join(save_folder, str(image_index).rjust(8,'0') + '.png'), frame)
            image_index += 1
            success, frame = video.read()
            frame = np.transpose(frame, (0, 1, 2))
            if image_index == 100:
                image_index = 0
                break





    #     if len(glob.glob(os.path.join(save_folder, '*.png'))) != 50:
    #         shutil.rmtree(save_folder)
    #     continue
    #valid,test is the all frames in videos
    # if mov_path.split('\\')[-1].split('.')[0] in valid_list:
    #     mov_folder = os.path.join(frameFolder, 'valid')
    # elif mov_path.split('\\')[-1].split('.')[0] in test_list:
    #     mov_folder = os.path.join(frameFolder, 'test')
    #
    # image_index = 0
    # video = cv2.VideoCapture(mov_path)
    # success, frame = video.read()
    # frame = np.transpose(frame, (0, 1, 2))
    # while success:
    #     # save_folder = os.path.join(mov_folder, mov_path.split('\\')[-1].split('.')[0], "{:04d}".format(folder_index))
    #     save_folder = os.path.join(mov_folder, mov_path.split('\\')[-1].split('.')[0])
    #     check_if_folder_exist(save_folder)
    #     cv2.imwrite(os.path.join(save_folder, str(image_index) + '.png'), frame)
    #     print(str(i) + ' ' + str(image_index))
    #     image_index += 1
    #     success, frame = video.read()
    #     if not success:
    #         break
    #     frame = np.transpose(frame, (0, 1, 2))
