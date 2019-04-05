from skimage.io import imsave
import argparse
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils import data
from model import Generator
import pdb
from torch.nn import functional as F
from data_utils import *
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data.dataset import Dataset
import os

q = 10

if not os.path.exists('HR'+str(q)+'_HR_results'): os.mkdir('HR'+str(q)+'_HR_results')
if not os.path.exists('HR'+str(q)+'_HR_results/hr'): os.mkdir('HR'+str(q)+'_HR_results/hr')
if not os.path.exists('HR'+str(q)+'_HR_results/lr'): os.mkdir('HR'+str(q)+'_HR_results/lr')
if not os.path.exists('HR'+str(q)+'_HR_results/sr'): os.mkdir('HR'+str(q)+'_HR_results/sr')

gpu_id = 3

netG = torch.load('models/G_280000.pt').to(gpu_id)

train_set = MyDataLoader(lap_hr_dir='../data/test_face_patch/HR_PNG/', lap_lp_dir='../data/test_face_patch/HR_JPEG/10/')
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)

cnt = 0

for idx, (lr, hr) in enumerate(train_loader):
    lr, hr = lr.to(gpu_id), hr.to(gpu_id)
    
    hr_hat = netG(lr)
    hr_hat = (F.tanh(hr_hat) + 1) / 2
    
    save_image(hr, 'HR'+str(q)+'_HR_results/hr/'+str(cnt)+'.jpg', nrow=1, padding=0)
    save_image(lr, 'HR'+str(q)+'_HR_results/lr/'+str(cnt)+'.jpg', nrow=1, padding=0)
    save_image(hr_hat, 'HR'+str(q)+'_HR_results/sr/'+str(cnt)+'.jpg', nrow=1, padding=0)
    
    cnt += 1

    print(cnt)



