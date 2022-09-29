import cv2
import os
from datetime import datetime
import glob

def video_to_frames(video_path, video_name, dir):

    dir_train = dir + '/'
    target_path = dir_train + '/'+ video_name + '/' + 'hr'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    videoCapture = cv2.VideoCapture()
    video = video_path
    print(video)
    videoCapture.open(video)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frames)
    frames = int(frames)

    if dir.split('/')[-1] == "train2":
        frames = 50

    for i in range(frames):
        print(i)
        ret, frame = videoCapture.read()
        cv2.imwrite(target_path + "\hr%d.png" % (i), frame)


if __name__ == '__main__':
    video_path = "E:\Video_4K"

    train_dir ="../data/train2"
    test_dir = "../data/test2"

    train_txt = 'E:/Video_4K/train.txt'
    test_txt = 'E:/Video_4K/test.txt'


    with open(train_txt) as f:
        temp = f.readlines()
        train_list = [v.strip() for v in temp]

    with open(test_txt) as f:
        temp = f.readlines()
        test_list = [v.strip() for v in temp]


    mov_files = glob.glob(os.path.join(video_path, '*'))

    for i, mov_path in enumerate(mov_files):
        print(mov_path)

        if mov_path.split('\\')[-1].split('.')[0] in train_list:
            video_name = mov_path.split('\\')[-1].split('.')[0]
            video_to_frames(mov_path, video_name, train_dir)

        if mov_path.split('\\')[-1].split('.')[0] in test_list:
            video_name = mov_path.split('\\')[-1].split('.')[0]
            video_to_frames(mov_path, video_name, test_dir)





