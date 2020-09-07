import os
import glob
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import random

def make_datapath_list(iorm='img', path='img', phase="train", rate=0.8):
    """
    make filepath list for train and validation image and mask.
    """
    rootpath = "./dataset/"+path
    target_path = os.path.join(rootpath+'/*.jpg')
    #print(target_path)

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    if phase=='train' and iorm=='img':
        num = len(path_list)
        random.shuffle(path_list)
        return path_list[:int(num*rate)], path_list[int(num*rate):]

    elif phase=='test' or iorm=='mask':
        return path_list


class ImageTransform():
    """
    preprocessing images
    """
    def __init__(self, size, mean, std):
        self.data_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class MaskTransform():
    """
    preprocessing images
    """
    def __init__(self, size):
        self.data_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor()])

    def __call__(self, img):
        return self.data_transform(img)


class ImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """

    def __init__(self, img_list, mask_list, img_transform, mask_transform):
        self.img_list = img_list
        self.mask_list = mask_list
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image
        '''
        gt = Image.open(self.img_list[index])
        gt = self.img_transform(gt.convert('RGB'))

        mask = Image.open(self.mask_list[random.randint(0, len(self.mask_list) - 1)])
        mask = self.mask_transform(mask.convert('RGB'))

        return gt * mask, mask, gt
