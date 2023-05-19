#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:20:19 2019

@author: owen
"""
# import pdb
from skimage.io import imsave
# import argparse
# import time
import torch
# from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from torchvision.transforms.functional import to_pil_image

#from torch.utils import data
from model import Generator
import pdb
from torch.nn import functional as F


# from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from torchvision.utils import save_image
import os
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import glob
# import pdb
from PIL import Image
import numpy as np
# import matplotlib.pyplot as plt
# from torchvision.utils import save_image

gpu_id = 0
CROP_SIZE = 350
UPSCALE_FACTOR = 2
TEST_MODE = True 
MODEL_NAME = 'G_24000.pt'
# model = Generator(UPSCALE_FACTOR).eval()
# model.to(gpu_id)
model = torch.load('models/' + MODEL_NAME)
model.to(gpu_id)

# def train_hr_transform(crop_size):
#     return Compose([
#         RandomCrop(crop_size),
#         ToTensor(),
#     ])
#
# def train_lr_transform(crop_size, upscale_factor):
#     return Compose([
#         ToPILImage(),
#         Resize(crop_size * upscale_factor, interpolation=Image.LANCZOS),
#         ToTensor()
#     ])
#
# def down_transform(crop_size):
#     return Compose([
#         ToPILImage(),
#         Resize(crop_size, interpolation=Image.LANCZOS),
#         ToTensor()
#     ])

def pil_transform():
    return Compose([
        ToPILImage()
    ])

def tt_transform():
    return Compose([
        ToTensor()
    ])


high_file = 'results_high_train'
low_file = 'results_low_train'
SR_file = 'results_SR_train'
if not os.path.exists(high_file): os.mkdir(high_file)
if not os.path.exists(low_file): os.mkdir(low_file)
if not os.path.exists(SR_file): os.mkdir(SR_file)

cnt = 0
for img_file in glob.glob('/home/snikolaev/JPEG_Artifact_Removal/verification/MIDI/*.jpg'):
    # img_file = 'high_big_picture/All_Hail_King_Julien_80018987_boxshot_USA_en_1_571x800_50_100.jpg'
    save_name = img_file.split('/')[-1]
    # pdb.set_trace()
    img = Image.open(img_file).convert('RGB')
    img = img.resize((img.width * UPSCALE_FACTOR, img.height * UPSCALE_FACTOR), resample=Image.LANCZOS)
    # img.show()
    img = np.array(img)

    s = CROP_SIZE #128
    h, w = img.shape[0], img.shape[1]
    nh, nw = int(np.ceil(h/s)), int(np.ceil(w/s))
    delta_h = s*nh - h; delta_w = s*nw - w
    img_pad = np.zeros((img.shape[0]+delta_h, img.shape[1]+delta_w, 3))
    img_pad[:h, :w] = img

    # hr_transform = train_hr_transform(s)
    # lr_transform = train_lr_transform(s, UPSCALE_FACTOR)
    # dn_transform = down_transform(s)
    ttt = tt_transform()
    pil = pil_transform()

    ########## save image patches ##########
    c = 0
    img_pad_2x = np.zeros_like(img_pad)
    for i in range(nh): 
        for j in range(nw): 
            patch = img_pad[i*s:(i+1)*s, j*s:(j+1)*s, :]
            
            # super resolve
            # data = hr_transform(Image.fromarray(patch.astype(np.uint8)))
            # data = lr_transform(data).unsqueeze(0)

            data = ttt(Image.fromarray(patch.astype(np.uint8))).unsqueeze(0)

            fake_img = model(Variable(data).to(gpu_id))
            fake_img = (F.tanh(fake_img) + 1) / 2
            img_pad_2x[i * s:(i + 1) * s, j * s:(j + 1) * s, :] = fake_img.detach().cpu().data.transpose(1,3).transpose(1,2).numpy()[0]#.astype(np.uint8)
            # img_pad_2x[i * s:(i + 1) * s, j * s:(j + 1) * s] = dn_transform(fake_img.cpu().data.transpose(1,3).transpose(1,2).numpy()[0].astype(np.uint8))

            # stitch upsampled patches into canvas
            # img_pad_2x[i*s:(i+1)*s, j*s:(j+1)*s, 0:2] = patch_2x[0:s, 0:s, 0:2] #.astype(object)

            c += 1

    img_2x = np.zeros((img.shape[0], img.shape[1],3))
    img_2x= img_pad_2x[:img_2x.shape[0], :img_2x.shape[1]]

    # img_2x=pil(img_2x)
    # img_2x=pil(ttt(img_2x))
    img_2x=Image.fromarray((img_2x * 255.).astype(np.uint8))
    # img_2x.show()
    img_1x=img_2x.resize((img_2x.width // UPSCALE_FACTOR, img_2x.height // UPSCALE_FACTOR), resample=Image.LANCZOS)
    # img_1x=Image.fromarray(img_2x.resize((img_2x.shape[1] // UPSCALE_FACTOR, img_2x.shape[0] // UPSCALE_FACTOR), resample=Image.LANCZOS).astype(np.uint8))

    img_1x.save(str(SR_file)+'/'+str(save_name)+'.png', 'PNG')
    # imsave(str(low_file)+'/'+str(save_name), low)
    # imsave(str(high_file)+'/'+str(save_name), img)
    
    cnt += 1
    print(cnt, save_name)



'''
if not os.path.exists('test_results'): os.mkdir('test_results')
if not os.path.exists('test_results/SR'): os.mkdir('test_results/SR')
if not os.path.exists('test_results/low'): os.mkdir('test_results/low')
if not os.path.exists('test_results/high'): os.mkdir('test_results/high')

cnt = 0
for img_file in glob.glob('../test_icon_imgs/*.jpg'):
    # img_file = 'high_big_picture/All_Hail_King_Julien_80018987_boxshot_USA_en_1_571x800_50_100.jpg'

    img = np.array(Image.open(img_file))
    low = np.array(Image.open(img_file).resize((int(img.shape[1]/2), int(img.shape[0]/2))))

    s = 128
    h, w = img.shape[0], img.shape[1]
    nh, nw = int(np.ceil(h/s)), int(np.ceil(w/s))
    delta_h = s*nh - h; delta_w = s*nw - w
    img_pad = np.zeros((img.shape[0]+delta_h, img.shape[1]+delta_w, 3))
    img_pad[:h, :w] = img

    hr_transform = train_hr_transform(128)
    lr_transform = train_lr_transform(128, 2)

    ########## save image patches ##########
    c = 0
    img_pad_2x = np.zeros((img_pad.shape[0], img_pad.shape[1], 3))
    for i in range(nh): 
        for j in range(nw): 
            patch = img_pad[i*s:(i+1)*s, j*s:(j+1)*s]
            
            # super resolve
            data = hr_transform(Image.fromarray(patch.astype(np.uint8)))
            data = lr_transform(data).unsqueeze(0)
            z = Variable(data).cuda()

            pdb.set_trace()
            fake_img = model(z)
            fake_img = (F.tanh(fake_img) + 1) / 2        
            patch_2x = fake_img.cpu().data.transpose(1,3).transpose(1,2).numpy()[0]
            
            # stitch upsampled patches into canvas
            img_pad_2x[i*s:(i+1)*s, j*s:(j+1)*s] = patch_2x

            c += 1
            # print(c)
    
    img_2x = np.zeros((img.shape[0], img.shape[1],3))
    img_2x= img_pad_2x[:img_2x.shape[0], :img_2x.shape[1]]

    imsave('test_results/SR/'+str(cnt)+'.jpg', img_2x)
    imsave('test_results/low/'+str(cnt)+'.jpg', low)
    imsave('test_results/high/'+str(cnt)+'.jpg', img)
    
    cnt += 1
    print(cnt)
'''


