from generate_mod_LR_bic import generate_mod_LR_bic
import os
import glob

folder_dir= '/home/fz/VSR/data/test'

scale = 4
folder_in = os.listdir(folder_dir)
# print(len(folder_in))
# folder_in = glob.glob(os.path.join(folder, '*'))
for folder in folder_in:
    # print(folder)
    folder_path = os.path.join(folder_dir,folder)
    # print(folder_path)
    HR_dir = folder_path + '/' + 'hr'
    LR_dir =folder_path + '/' + 'lr' + '_x' + str(scale)

    if not os.path.exists(LR_dir):
            os.makedirs(LR_dir)
    # for test in test_list:
    #     test_path = os.path.join(folder_path,test)
    #     print(test_path)
    #     HR_dir = test_path + '/' + 'hr'
    #     LR_dir = test_path + '/' + 'lr' + '_x' + str(scale)
    #     if not os.path.exists(LR_dir):
    #             os.makedirs(LR_dir)

    path_in = HR_dir
    path_out = LR_dir

    generate_mod_LR_bic(scale, path_in, path_out)


