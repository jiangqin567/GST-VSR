import os


def re_name(path):

    id = 0

    for file in sorted(os.listdir(path)):

        print(file)
        # file_path = os.path.join(path, file)
        #
        # file_new = file.replace(file, str(id).rjust(6,'0') + ".mp4")
        #
        # id = id + 1
        #
        # print(file_new)
        #
        # file_new_path = os.path.join(path, file_new)
        #
        # os.rename(file_path, file_new_path)




if __name__ == '__main__':

    path = r"/home/fz/VSR/data/train/000000/hr"

    re_name(path)