from collections import OrderedDict
from models.UNet_with_PConv import PConvUNet
from models.loss import Losses
from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform, MaskTransform
from torchvision import models
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch
import os
from torchvision.utils import make_grid
from torchvision.utils import save_image

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def plot_log(data, save_model_name='model'):
    plt.cla()
    for key in data.keys():
        plt.plot(data[key], label='key'+'_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

def train_model(pconv_unet, dataloader, val_dataset, num_epochs, save_model_name='model'):

    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pconv_unet.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        pconv_unet = torch.nn.DataParallel(pconv_unet)
        print("parallel mode")

    print("device:{}".format(device))

    #TODO: lr
    if False:#finetune
        lr = 5e-5
        pconv_unet.fine_tune = True
    else:
        lr = 2e-4

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pconv_unet.parameters()), lr=lr)
    criterion = Losses().to(device)

    z_dim = 20
    mini_batch_size = 16


    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    hole_losses = []
    valid_losses = []
    perceptual_losses = []
    style_losses = []
    total_variation_losses = []
    losses = {'hole':hole_losses, 'valid':valid_losses, 'perceptual':perceptual_losses, 'style':style_losses, 'tv':total_variation_losses}

    for epoch in range(num_epochs+1):
        pconv_unet.train()

        t_epoch_start = time.time()
        epoch_loss = {}
        epoch_loss['hole'] = 0.0
        epoch_loss['valid'] = 0.0
        epoch_loss['perceptual'] = 0.0
        epoch_loss['style'] = 0.0
        epoch_loss['tv'] = 0.0


        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')

        for images, mask, gt in tqdm(dataloader):

            # Train Discriminator
            # if size of minibatch is 1, an error would be occured.
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            mask = mask.to(device)
            gt = gt.to(device)

            mini_batch_size = images.size()[0]

            output, _ = pconv_unet(images, mask)
            loss_dict = criterion(images, mask, output, gt)

            loss = 0.0
            lambda_dict = {'valid':1.0, 'hole':6.0, 'perceptual':0.05, 'style':120, 'tv':0.1}
            for key, _lambda in lambda_dict.items():
                loss += _lambda * loss_dict[key]
                epoch_loss[key] += _lambda * loss_dict[key]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration += 1

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {}'.format(epoch))
        for key in lambda_dict.keys():
            print('{:s}_loss:{:.4f}'.format(key, epoch_loss[key]/batch_size))
            losses[key].append(epoch_loss[key]/batch_size)
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        plot_log(losses, save_model_name)

        if(epoch%10 == 0):
            torch.save(pconv_unet.state_dict(), 'checkpoints/PConvUNet_'+save_model_name+'_'+str(epoch)+'.pth')
            pconv_unet.eval()
            evaluate(pconv_unet, val_dataset, device, '{:s}/test_{:d}.jpg'.format('result', i + 1))
    return pconv_unet

def main():
    pconv_unet = PConvUNet()

    train_img_list, val_img_list = make_datapath_list(iorm='img', phase='train')
    mask_list = make_datapath_list(iorm='mask')
    mean = (0.5,)
    std = (0.5,)
    train_dataset = ImageDataset(img_list=train_img_list, mask_list=mask_list,
                                img_transform=ImageTransform(size=(256, 256), mean=mean, std=std),
                                mask_transform=MaskTransform(size=(256, 256)))
    val_dataset = ImageDataset(img_list=val_img_list, mask_list=mask_list,
                                img_transform=ImageTransform(size=(256, 256), mean=mean, std=std),
                                mask_transform=MaskTransform(size=(256, 256)))

    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#num_workers=4)

    num_epochs = 10000
    pconv_unet_update = train_model(pconv_unet, dataloader=train_dataloader,
                                    val_dataset=val_dataset, num_epochs=num_epochs,
                                    save_model_name='PConvUNet')


if __name__ == "__main__":
    main()