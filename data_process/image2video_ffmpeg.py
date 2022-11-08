## You need ffmpeg version 4 to support the option '-pred mixed' which is new in version 4.
## The option '-pred mixed' gives smaller .png file size (lossless compression).
## The older version of ffmpeg also can be used without the option '-pred mixed'
## To install the ffmpeg version 4 in ubuntu, please run the below lines through terminal.

# conda install -c conda-forge ffmpeg

## Please modify the below code lines if needed.

import os, glob, sys

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
#         print(log_dir, " created")
    return log_dir

test_dir = "../data/windflow/"
new_dir = "../results/"
outfile = new_dir + "windflow.avi"
image_list = sorted(os.listdir(test_dir))
cmd = "ffmpeg -r 5 -f image2 -i {}%d.bmp  {} ".format(test_dir, outfile)
print(cmd)
if os.system(cmd):
    raise KeyboardInterrupt



#
# for test_type in test_types:
#     test_type_path = test_dir + test_type
#     print(test_type_path)
#     image_path = test_type_path
#
#     test_video_list = os.listdir(test_type_path)
#     #
#     for test_video in test_video_list:
#         image_path = test_type_path + test_video
#         print(image_path)
#         samples = sorted(glob.glob(image_path))
        # # print(samples)
        # for sample in samples:
        #     new_dir = sample.replace('test', 'test_video')
        #     check_folder(new_dir)
        #     outfile = new_dir + new_dir.split('/')[-2] + '.mp4'
        #     cmd = "ffmpeg -f image2 -i {}%04d.png  {} ".format(sample, outfile)
        #     print(cmd)
        #     if os.system(cmd):
        #         raise KeyboardInterrupt
        #




# for test_type in test_types:
#     test_type_path = test_dir + test_type + '/'
#     print(test_type_path)
#     test_video_list = os.listdir(test_type_path)
#     #
#     for test_video in test_video_list:
#         image_path = test_type_path + test_video +'/'
#         # print(image_path)
#         samples = sorted(glob.glob(image_path))
#         # print(samples)
#         for sample in samples:
#             new_dir = sample.replace('test', 'test_video')
#             check_folder(new_dir)
#             outfile = new_dir + new_dir.split('/')[-2] + '.mp4'
#             cmd = "ffmpeg -f image2 -i {}%04d.png  {} ".format(sample, outfile)
#             print(cmd)
#             if os.system(cmd):
#                 raise KeyboardInterrupt
