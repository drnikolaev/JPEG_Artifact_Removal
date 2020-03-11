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
import torch.nn as nn
import torch.nn.functional as F
# import torch.device as device
# import pytorch_ssim

gpu_id = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 100
display = visualizer(port=8097)
report_feq = 10

train_set = MyDataLoader(hr_dir='/home/snikolaev/Artifacts/MAXI/', lr_dir='/home/snikolaev/Artifacts/MIDI/')
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)

#netG = torch.load('models/G_12000.pt')
#netD = torch.load('models/D_12000.pt')
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
    netG.train()
    netD.train()

    for idx, (lr, hr) in enumerate(train_loader):
        lr, hr = lr.to(gpu_id), hr.to(gpu_id)

        ############# Forward ###############
        # hr_hat = netG(lr)
        hr_hat = nn.parallel.data_parallel(netG, lr, device_ids=[0, 1, 2, 3])
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
        g_loss = generator_criterion(fake_out, hr_hat, hr).cuda()
        g_loss.backward()
        optimizerG.step()
        # hr_hat = netG(lr)

        # hr_hat = nn.parallel.data_parallel(netG, lr, device_ids=[0, 1, 2, 3])
        # hr_hat = (F.tanh(hr_hat) + 1) / 2
        
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
            display.plot_img_255(vis_recon,  win=3, caption='recovered')

        print(epoch, step)
        step += 1
        
        ########## Save Models ##########
        if step % 1000 == 0:
            if not os.path.exists('models'): os.mkdir('models')
            torch.save(netG, 'models/G_'+str(step)+'.pt')
            torch.save(netD, 'models/D_'+str(step)+'.pt')


