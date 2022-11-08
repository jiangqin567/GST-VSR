import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from motion_compensation_revise import VSR
import argparse
from data_utils import TrainsetLoader
import numpy as np
import torch.nn as nn
import scipy.ndimage
from alignment import optical_flow_warp
from loss import MSE_and_SSIM_loss
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upscale_factor", type=int, default=4)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=600001, help='number of iterations to train')
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
        net = nn.DataParallel(net,device_ids=[0,1,2,3,4,5,6,7])

    cudnn.benchmark = True

    print("loading data********************")
    train_set = TrainsetLoader(cfg.trainset_dir, cfg.upscale_factor, cfg.patch_size, cfg.epoch*cfg.batch_size,cfg.frame_num) #
    train_loader = DataLoader(train_set, num_workers=32, batch_size=cfg.batch_size, pin_memory=True,
                               shuffle=True)
    print("data loaded*********************")


    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    criterion_ME = torch.nn.MSELoss()
    criterion = MSE_and_SSIM_loss()

    if use_gpu:
        criterion = criterion.cuda()
    milestones = [50000, 100000, 150000, 200000, 250000,300000,350000,400000,450000,550000,600000,700000,800000,900000,]
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    loss_list = []
    # loss_ME_list=[]


    for idx_iter, (LR, HR) in enumerate(train_loader):

        print("running*********idx_iter:", idx_iter)
        LR, HR = Variable(LR), Variable(HR)
        if use_gpu:
            LR = LR.cuda()
            HR = HR.cuda()

        SR, ME_res_list,Ali_res_list = net(LR)  #torch.Size([16, 1, 256, 256])
        # lpss_ME = np.array(loss_ME_list).mean()
        loss_ME = 0
        loss_Ali = 0
        for ME_res in ME_res_list:
            loss_ME += L1_regularization(ME_res)

        for Ali_res in Ali_res_list:
            loss_Ali += L1_regularization(Ali_res)

        loss_SR = criterion(SR, torch.unsqueeze(HR[:, 2, :, :], 1)) + 0.01 * loss_ME + 0.25* loss_Ali 


        loss = loss_SR
        loss_list.append(loss.data.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        
        info_dir = 'info3'
        with open(info_dir + '/info.txt', 'a') as f:  
            print('Iteration---%6d,   criterion---%f' % (idx_iter + 1, np.array(loss_list).mean()), file=f)

        # print('Iteration---%6d,   criterion---%f' % (idx_iter + 1, np.array(loss_list).mean()))
        # torch.save(net.state_dict(), 'log/BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
        # loss_list = []

        # save checkpoint
        
        if idx_iter % 100000 == 0:   
   #         loss_mean = np.array(loss_list).mean()
    #        writer_iter.add_scaler('Loss/Train_iter',loss.item(),idx_iter)

   
            
            print('Iteration---%6d,   criterion---%f' % (idx_iter + 1, np.array(loss_list).mean()))
            torch.save(net.state_dict(), 'log3/BI_x' + str(cfg.upscale_factor) + '_iter' + str(idx_iter) + '.pth')
            loss_list = []

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







