import argparse
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from vis_tools import *
from data_utils import *
from loss import GeneratorLoss
from model import Generator, Discriminator
import architecture as arch
import pdb 
import torch.nn.functional as F
# import pytorch_ssim

gpu_id = 0
NUM_EPOCHS = 40
display = visualizer(port=8094)
report_feq = 10

train_set = MyDataLoader(hr='../data/train_face/HR/', lr='../data/train_face/HR_JPEG/10/')
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=2, shuffle=True)

netG = arch.RRDB_Net(3, 3, 64, 6, gc=32, upscale=1, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
netD = Discriminator()

generator_criterion = GeneratorLoss()
if torch.cuda.is_available():
    netG.to(gpu_id)
    netD.to(gpu_id)
    generator_criterion.to(gpu_id)

optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)

step = 0
for epoch in range(1, NUM_EPOCHS):
    netG.train(); netD.train()

    for idx, (lr, hr) in enumerate(train_loader):
        lr, hr = lr.to(gpu_id), hr.to(gpu_id)

        ############# Forward ###############
        hr_hat = netG(lr)
        hr_hat = (F.tanh(hr_hat) + 1) / 2

        ############# Update D ###############
        netD.zero_grad()
        real_out = netD(hr).mean()
        fake_out = netD(hr_hat).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        
        ############# Update G ###############
        netG.zero_grad()
        g_loss = generator_criterion(fake_out, hr_hat, hr)
        g_loss.backward()
        optimizerG.step()
        hr_hat = netG(lr)
        hr_hat = (F.tanh(hr_hat) + 1) / 2
        
        if step % report_feq == 0:
            # ssim = pytorch_ssim.ssim(hr, hr_hat)
            # ssim_np = ssim.cpu().data.numpy()
            # err_dict = {'ssim:': ssim_np}   
            # display.plot_error(err_dict)
            
            vis_high = (hr*255)[0].detach().cpu().data.numpy()
            vis_low = (lr*255)[0].detach().cpu().data.numpy()
            vis_recon = (hr_hat*255)[0].detach().cpu().data.numpy()

            display.plot_img_255(vis_high, win=1, caption='high')
            display.plot_img_255(vis_low,  win=2, caption='low')
            display.plot_img_255(vis_recon,  win=3, caption='sr')

        print(epoch, step)
        step += 1
        
        ########## Save Models ##########
        if step % 5000 == 0:
            if not os.path.exists('models'): os.mkdir('models')
            torch.save(netG, 'models/G_'+str(step)+'.pt')
            torch.save(netD, 'models/D_'+str(step)+'.pt')


