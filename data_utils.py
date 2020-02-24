import os
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
    def __init__(self, hr_dir, lr_dir):
        super(MyDataLoader, self).__init__()

        # n_imgs = 0
        # for file in glob.glob(str(hr_dir)+'*.png'):
        #     n_imgs += 1
        #
        # hr_list = []; lr_list = []
        # for i in range(n_imgs):
        #     hr_list.append(str(hr_dir)+str(i)+'.png')
        #     lr_list.append(str(lr_dir)+str(i)+'.jpg')

        hr_list = []
        lr_list = []
        for file in glob.glob(str(lr_dir) + '*.jpg'):
            hr_file = re.sub(r'MIDI', 'MAXI', file)
            hr_file = re.sub(r'\(1\)\.jpg', r'(2).jpg', hr_file)

            if os.path.exists(hr_file):
                hr_list.append(hr_file)
                lr_list.append(file)

        # self.transform = toTensor_transform()
        # self.transform2 = toTensor_transform2()
        self.hr_list = hr_list
        self.lr_list = lr_list
        
    def __getitem__(self, idx):
        # hr = self.transform2(Image.open(self.hr_list[idx]).convert(mode='RGB'))
        # lr = self.transform(Image.open(self.lr_list[idx]).convert(mode='RGB'))
        hr = Image.open(self.hr_list[idx]).convert(mode='RGB')
        lr = Image.open(self.lr_list[idx]).convert(mode='RGB')

        scale = 2
        lsize = 300
        lcc = CenterCrop((lsize, lsize))
        tt = ToTensor()
        ratio = float(hr.width)/float(lr.width)
        hsize = lsize * scale
        hccr = CenterCrop((hsize * ratio, hsize * ratio))
        rsz = Resize((hsize))


        # print(hr.width, float(hr.width)/float(lr.width), float(hr.height)/float(lr.height), lr.width)

        # hr = self.transform2(hr)
        # lr = self.transform(lr)

        lr = tt(lcc(lr))
        hr = tt(rsz(hccr(hr)))

        return lr, hr

    def __len__(self):
        return len(self.hr_list)

'''
train_set = MyDataLoader()
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=100, shuffle=True)

for idx, (lr, hr) in enumerate(train_loader):
    print(idx, lr.shape, hr.shape)
'''




