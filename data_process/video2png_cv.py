import cv2
import os
from datetime import datetime

def video_to_frames(video_path, dir):   #原视频文件夹和目标文件夹
    """
    输入：path(视频文件的路径)
    """
    # VideoCapture视频读取类
    video_list_dir = video_path
    video_list = os.listdir(video_list_dir)

    print(video_list)
    length = len(video_list)

    for video_list_name in video_list:

        dir_train = dir + '/' + video_list_name.split('_')[0]

        if not os.path.exists(dir_train):
            os.makedirs(dir_train)

        target_path = dir_train + '/'+ 'hr'
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        videoCapture = cv2.VideoCapture()
        video = video_path + '/' + video_list_name
        print(video)
        videoCapture.open(video)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print(frames)
        frames = int(frames)

        for i in range(50):
            print(i)
            ret, frame = videoCapture.read()
            cv2.imwrite(target_path + "\hr%d.png" % (i), frame)

    return

if __name__ == '__main__':
    t1 = datetime.now()
    video_path = "../data/video_test"
    dir = "../data/test1"
    video_to_frames(video_path,dir)
    t2 = datetime.now()
    print("Time cost = ", (t2 - t1))
    print("SUCCEED !!!")

