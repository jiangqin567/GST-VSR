import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from motion_compensation_revise import VSR
import argparse
from data_utils_480P import TrainsetLoader
import numpy as np
import torch.nn as nn
import scipy.ndimage
from alignment import  optical_flow_warp
from loss import MSE_and_SSIM_loss
from torch.utils.tensorboard import SummaryWriter
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=64)  #一张图片分成很多个patch,送入神经网络进行卷积，patch32表示裁剪成32*32的块
    parser.add_argument('--batch_size', type=int, default=16)   #batch表示一次处理16张图片
    parser.add_argument('--epoch', type=int, default=200000, help='number of iterations to train')
    parser.add_argument('--trainset_dir', type=str, default='data/train1')
    parser.add_argument('--frame_num', type=int, default=5)

    return parser.parse_args()

def main(cfg):
    use_gpu = cfg.gpu_mode
    frame_num = cfg.frame_num
    net = VSR(cfg.upscale_factor,cfg.frame_num)
    # net = nn.DataParallel(net,device_ids=[0,1,2,3,4,5,6,7])
    if use_gpu:
        net.cuda()
        net = nn.DataParallel(net)

    cudnn.benchmark = True

    print("loading data********************")
    train_set = TrainsetLoader(cfg.trainset_dir, cfg.upscale_factor, cfg.patch_size, cfg.epoch*cfg.batch_size,cfg.frame_num) #
    train_loader = DataLoader(train_set, num_workers=4, batch_size=cfg.batch_size, pin_memory=True,
                               shuffle=True)
    print("data loaded*********************")


    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion_ME = torch.nn.MSELoss()
    criterion = MSE_and_SSIM_loss()

    if use_gpu:
        criterion = criterion.cuda()
    milestones = [50000, 100000, 150000, 200000, 250000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)#当epoch满足milestones时，调整学习率，gamma学习率调整倍数
    loss_list = []
    best_loss = 1
    writer_iter_dir = r'runs'
    if not os.path.join(writer_iter_dir): os.makedirs(writer_iter_dir)
    writer_iter = SummaryWriter(writer_iter_dir)
    # loss_ME_list=[]
    ME_res_list = []
    Ali_res_list = []

    for idx_iter, (LR, HR) in enumerate(train_loader):

        print("running*********idx_iter:", idx_iter)
        LR, HR = Variable(LR), Variable(HR)
        if use_gpu:
            LR = LR.cuda()
            HR = HR.cuda()

        SR, ME_res_list, Ali_res_list= net(LR)  #torch.Size([16, 1, 256, 256])
        # lpss_ME = np.array(loss_ME_list).mean()
        loss_ME = 0
        loss_Ali = 0

        for ME_res in ME_res_list:
            loss_ME += L1_regularization(ME_res)

        for Ali_res in Ali_res_list:
            loss_Ali += L1_regularization(Ali_res)

        loss_SR = criterion(SR, torch.unsqueeze(HR[:, 1, :, :], 1)) + 0.01 * loss_ME + 0.25 * loss_Ali

        # loss_SR = criterion(SR, torch.unsqueeze(HR[:, 1, :, :], 1))
        loss = loss_SR
        loss_list.append(loss.data.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        info_dir = 'info_480P'
        with open(info_dir + '/info.txt', 'a') as f:  # 设置文件对象
            print('Iteration---%6d,   criterion---%f' % (idx_iter + 1, np.array(loss_list).mean()), file=f)

        # print('Iteration---%6d,   criterion---%f' % (idx_iter + 1, np.array(loss_list).mean()))
        # torch.save(net.state_dict(), 'log/BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
        # loss_list = []

        # save checkpoint
        # if idx_iter % 10 == 0:   #第0、5000、10000
        #
        #     print('Iteration---%6d,   criterion---%f' % (idx_iter + 1, np.array(loss_list).mean()))
        #     torch.save(net.state_dict(), 'log/BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
        #     loss_list = []
        if idx_iter % 100 == 0:  # 第0、5000、10000
            loss_mean = np.array(loss_list).mean()

            # writer_iter.add_scalar('Loss/Train_iter', loss_mean.item(), idx_iter)
            #
            # for name, param in net.named_parameters():
            #     writer_iter.add_histogram('batch_' + name + '_param', param, idx_iter)
            #     writer_iter.add_histogram('batch_' + name + '_grad', param.grad, idx_iter)

            if best_loss > loss_mean:
                best_loss = loss_mean
                print('Iteration---%6d,   best_loss：criterion---%f' % (idx_iter, np.array(loss_list).mean()))
                torch.save(net.state_dict(), 'log_480P/BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
                print(' Best average loss: %.3f' % (best_loss))

        # scheduler.step()

def L1_regularization(image):
    b, _, h, w = image.size()
    reg_x_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 1:, 0:w-1]
    reg_y_1 = image[:, :, 0:h-1, 0:w-1] - image[:, :, 0:h-1, 1:]
    reg_L1 = torch.abs(reg_x_1) + torch.abs(reg_y_1)
    return torch.sum(reg_L1) / (b*(h-1)*(w-1))

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)







