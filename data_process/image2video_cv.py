import cv2
import numpy as np
import glob
import os

# 其它格式的图片也可以
img_array = []
for filename in glob.glob('../data/windflow/*.bmp'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# avi：视频类型，mp4也可以
# cv2.VideoWriter_fourcc(*'DIVX')：编码格式
# 5：视频帧率
# size:视频中图片大小
# out = cv2.VideoWriter('results/windflow.avi',
#                       cv2.VideoWriter_fourcc(*'DIVX'),
#                       5, size)
out = cv2.VideoWriter('../results/windflow.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'),
                      5, (1024,910))

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
