import glob
import os
import shutil

def slipt_4Kdataset(path,target):

    image_list = os.listdir(path)
    for image in image_list:
        print(image)
        image_path = os.path.join(path,image)
        if  image.__contains__('_'):
            clip_id = image.split('_')[0]   #0,1,2
        else:
            clip_id = '100'

        clip_id_dir = os.path.join(target,clip_id)

        if not os.path.exists(clip_id_dir):
            os.makedirs(clip_id_dir)
        target_path = clip_id_dir
        print(image_path)
        print(target_path)
        shutil.move(image_path, target_path)
        # break

if __name__ == '__main__':
    path = "..data\groudtruth"
    target = "..data\\4Kdata"
    slipt_4Kdataset(path,target)






