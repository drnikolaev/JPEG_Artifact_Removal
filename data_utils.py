import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import glob
import pdb
import re

# def toTensor_transform():
#     return Compose([
#         CenterCrop((200, 200)),
#         ToTensor()
#     ])
#
# def toTensor_transform2():
#     return Compose([
#         CenterCrop((400, 400)),
#         ToTensor()
#     ])

class MyDataLoader(Dataset):
    def __init__(self, hr_dir, lr_dir, infer=False):
        super(MyDataLoader, self).__init__()

        lr_list = []
        hr_list = []
        self.infer = infer
        # if infer:
        #     for file in glob.glob(str(lr_dir) + '*.jpg'):
        #         lr_list.append(file)
        #     self.lr_list = lr_list
        #     self.hr_list = hr_list
        #     return

        for file in glob.glob(str(lr_dir) + '*.jpg'):
            hr_file = re.sub(r'MIDI', 'MAXI', file)
            hr_file = re.sub(r'\(1\)\.jpg', r'(2).jpg', hr_file)

            if os.path.exists(hr_file):
                hr_list.append(hr_file)
                lr_list.append(file)

        self.hr_list = hr_list
        self.lr_list = lr_list
        
    def __getitem__(self, idx):
        scale = 2
        lsize = 400 if self.infer else 350

        tt = ToTensor()
        # if self.infer:
        #     hr = None
        #     lr = Image.open(self.lr_list[idx]).convert(mode='RGB')
        #     rsz = Resize((lr.height * scale, lr.width * scale), interpolation=Image.LANCZOS)
        #     lr = tt(rsz(lr))
        #     return lr, self.lr_list[idx]

        hr = Image.open(self.hr_list[idx]).convert(mode='RGB')
        lr = Image.open(self.lr_list[idx]).convert(mode='RGB')

        lr = lr.resize(size = (lr.width * scale, lr.height * scale), resample=Image.LANCZOS)

        left = np.random.randint(low=0, high=lr.width-lsize if lr.width > lsize else 1)
        top = np.random.randint(low=0, high=lr.height-lsize if lr.height > lsize else 1)

        hr = hr.resize(size = (lr.width, lr.height), resample=Image.LANCZOS)
        hr = hr.crop((left, top, left + lsize, top + lsize))
        lr = lr.crop((left, top, left + lsize, top + lsize))
        return tt(lr), tt(hr)

    def __len__(self):
        return len(self.lr_list)

'''
train_set = MyDataLoader()
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=100, shuffle=True)

for idx, (lr, hr) in enumerate(train_loader):
print(idx, lr.shape, hr.shape)
'''
