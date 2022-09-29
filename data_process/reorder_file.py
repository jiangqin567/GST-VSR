import os
import shutil
import glob

# dir = 'data\\test'
#
#
# HR_dir = 'dataset\REDS\\val_sharp'
# LR_dir = 'dataset\REDS\\val_sharp_bicubic\X4'
#
#
# folder_HR = glob.glob(os.path.join(HR_dir, '*'))   #000,001
# # print(folder_HR)
# folder_LR = glob.glob(os.path.join(LR_dir, '*'))   #000,001
# # print(folder_LR)
#
# for folder_hr in folder_HR:
#     print(folder_hr)
#     folder_name = folder_hr.split('\\')[-1].split('.')[0]
#     HR_dir_out = 'data\\test' + '/' + folder_name + '/' + 'hr'
#     if not os.path.exists(HR_dir_out):
#         os.makedirs(HR_dir_out)
#     # print(folder_name)
#     folder_path = HR_dir +'/'+ folder_name
#     file_list = os.listdir(folder_path)
#     for file in file_list:
#         file_path = folder_hr +'/' + file
#         print(file_path)
#         print(HR_dir_out)
#         shutil.move(file_path, HR_dir_out)
#
# for folder_lr in folder_LR:
#     folder_name = folder_lr.split('\\')[-1].split('.')[0]
#
#     LR_dir_out = 'data\\test' + '/' + folder_name + '/' + 'lr_x4'
#     if not os.path.exists(LR_dir_out):
#         os.makedirs(LR_dir_out)
#
#     print(folder_name)
#     folder_path = LR_dir +'/'+ folder_name
#     file_list = os.listdir(folder_path)
#     for file in file_list:
#         file_path = folder_lr +'/' + file
#         print(file_path)
#         print(LR_dir_out)
#         shutil.move(file_path, LR_dir_out)

#
# #
dir  = '..data\X4K1000FPS'
dir_list = os.listdir(dir)    #
target_dir = '..\data\X4K1000FPS'

for type in dir_list:
    type_path = os.path.join(dir,type)
    type_list = os.listdir(type_path)
    for type in type_list:
        test_dir = os.path.join(type_path,type)
        target_path = os.path.join(type_path,test_dir) + '/' + 'hr'
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        image_list = os.listdir(test_dir)

        for image in image_list:
            if image.__contains__('.png'):
                image_path = os.path.join(test_dir,image)
                shutil.move(image_path, target_path)






    #
    # file_path = dir + '/'+ dir_name
    # file_list = os.listdir(file_path)   #0.png
    # target_path = os.path.join(target_dir,dir_name) +'/' + 'hr'
    # # print(target_path)
    # if not os.path.isdir(target_path):
    #     os.mkdir(target_path)
    # # print(target_path)
    # #     shutil.rmtree(file_path, True)
    #
    # # file_list = os.listdir(file_path)
    #
    # for file in file_list:
    #     if file.__contains__('.jpg'):
    #         file_name = file_path +'/' + file
    #         print(file_name)
    #         print(target_path)
    #         shutil.move(file_name, target_path)

        # if os.path.isdir(file):
        #     print(file)
        #     shutil.rmtree(file, True)

    #

    # # print(file_list)
    # for file in file_list:
    #     image_path = os.path.join(file_path,file)
    #     print(image_path)
    #     image = os.listdir(image_path)
    #
    #
    #     for image_name in image:
    #         image_name = image_path + '/' + image_name
    #         print(image_name)
    #         shutil.move(image_name, file_path)
            # s
    #
    # if os.path.isdir(filepath):
    #         shutil.rmtree(filepath, True)
    #     print(image)
    # #
