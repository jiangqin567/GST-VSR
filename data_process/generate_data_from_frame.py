from generate_mod_LR_bic import generate_mod_LR_bic
import os
import glob

# Configuration
# ============================

# args = [
#         {
#             'folder_in': '../datasets/adobe240fps/frame',
#             'folder_out': '../datasets/adobe240fps/frame_LRx4',
#             'up_scale': 8,
#         },
#         {
#             'folder_in': '../datasets/adobe240fps/frame',
#             'folder_out': '../datasets/adobe240fps/frame_HR',
#             'up_scale': 2,
#         },
#     ]

#
# args = [
#         {
#             'folder_in': 'data/test2',
#             'folder_out': 'data/test2',
#             'up_scale': 4,
#         },
        # {
        #     'folder_in': '../datasets/adobe240fps/frame',
        #     'folder_out': '../datasets/adobe240fps/frame_HR',
        #     'up_scale': 2,
    #     # },
    # ]

# Run
# =========================================================================================

# for arg in args:
#     folder_in = arg['folder_in']
#     folder_out = arg['folder_out']
#     up_scale = arg['up_scale']
#
#     folder_root = folder_in
#     folder_leaf = []
#     folder_branch = []
#     file_leaf = []
#     index = 0
#
#     for dirpath, subdirnames, filenames in os.walk(folder_root):
#         print('Processing ' + str(index) + ' Item')
#         index += 1
#
#         if len(subdirnames) == 0:
#             folder_leaf.append(dirpath)
#         else:
#             folder_branch.append(dirpath)
#
#         for i in range(len(filenames)):
#             file_leaf.append(os.path.join(dirpath, filenames[i]))
#
#     for i in range(len(folder_leaf)):
#         print('Processing ' + str(i) + ' to Get LR image')
#         path_in = folder_leaf[i]
#         # print(path_in)
#         path_out = os.path.join(folder_out, folder_leaf[i][len(folder_in) + 1:])
#         # print(path_out)
#         if not os.path.exists(path_out):
#             os.makedirs(path_out)
#
#         generate_mod_LR_bic(up_scale, path_in, path_out)
# =========================================================================================

folder_dir= '/home/fz/VSR/data/train1'

scale = 16
folder_in = os.listdir(folder_dir)
# print(len(folder_in))
# folder_in = glob.glob(os.path.join(folder, '*'))
for folder in folder_in:
    # print(folder)
    folder_path = os.path.join(folder_dir,folder)
    # print(folder_path)
    HR_dir = folder_path + '/' + 'hr'
    # HR_dir = folder_path
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
    # print(path_out)
    # print(path_in)

    generate_mod_LR_bic(scale, path_in, path_out)

#
# for i, folder_path in enumerate(folder_in):
#
#     print(folder_path)
#     HR_dir = folder_path + '/' + 'hr'
#     LR_dir = folder_path + '/' + 'lr' + '_x' + str(scale)
#
#
#     if not os.path.exists(LR_dir):
#             os.makedirs(LR_dir)
#
#     path_in = HR_dir
#     path_out = LR_dir
    # generate_mod_LR_bic(scale, path_in, path_out)

