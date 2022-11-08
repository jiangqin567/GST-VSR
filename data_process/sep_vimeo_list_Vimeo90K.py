import os, shutil

if __name__ ==  "__main__":
    mode = 'test'
    # inPath = '/home/ubuntu/Disk/dataset/Vimeo-90k/vimeo_septuplet/sequences/'
    # outPath = '/home/ubuntu/Disk/dataset/Vimeo-90k/vimeo_septuplet/sequences/' + mode + '/'
    # guide = '/home/ubuntu/Disk/dataset/Vimeo-90k/vimeo_septuplet/sep_' + mode + 'list.txt'


    inPath = 'E:/Video_4K/'
    outPath = 'data/frame/' + mode + '/'
    guide = 'E:/Video_4K/' + mode + '.txt'

    f = open(guide, "r")
    lines = f.readlines()
    
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    for l in lines:
        line = l.replace('\n','')
        this_folder = os.path.join(inPath, line)
        dest_folder = os.path.join(outPath, line)
        print(this_folder)
        shutil.copytree(this_folder, dest_folder)
    print('Done')
