from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import glob
import pdb

def toTensor_transform():
    return Compose([
        ToTensor()
    ])

class MyDataLoader(Dataset):
    def __init__(self, hr_dir, lr_dir):
        super(MyDataLoader, self).__init__()

        n_imgs = 0
        for file in glob.glob(str(hr_dir)+'*.png'):
            n_imgs += 1
        
        hr_list = []; lr_list = []
        for i in range(n_imgs):
            hr_list.append(str(hr_dir)+str(i)+'.png')
            lr_list.append(str(lr_dir)+str(i)+'.jpg')
            
        self.transform = toTensor_transform()
        self.hr_list = hr_list
        self.lr_list = lr_list
        
    def __getitem__(self, idx):
        hr = self.transform(Image.open(self.hr_list[idx]))
        lr = self.transform(Image.open(self.lr_list[idx]))
        return lr, hr

    def __len__(self):
        return len(self.hr_list)

'''
train_set = MyDataLoader()
train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=100, shuffle=True)

for idx, (lr, hr) in enumerate(train_loader):
    print(idx, lr.shape, hr.shape)
'''




