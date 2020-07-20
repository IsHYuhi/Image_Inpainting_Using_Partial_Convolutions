import torch
import torch.nn as nn
from torchvision import models

class VGG16map(nn.Module):
    def __init__(self):
        super(VGG16map, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.layer1 = nn.Sequential(*vgg16.features[:5])
        self.layer2 = nn.Sequential(*vgg16.features[5:10])
        self.layer3 = nn.Sequential(*vgg16.features[10:17])

        #fix
        for i in range(3):
            for param in getattr(self, 'layer'+str(i+1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        pool1 = self.layer1(input)
        pool2 = self.layer2(pool1)
        pool3 = self.layer3(pool2)

        return [pool1, pool2, pool3]

class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.vgg_map = VGG16map()

    def gram_matrix(self, input):
        (b, ch, h, w) = input.size()
        features = input.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        input = torch.zeros(b, ch, ch).type(features.type())
        gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1./(ch * h * w), out=None)
        return gram

    def forward(self, input, mask, out, gt):
        loss_dict = {}
        comp = mask * input + (1 - mask) * out

        # L_hole
        loss_dict['hole'] = self.l1_loss(mask * out, mask * gt)
        # L_valid
        loss_dict['valid'] = self.l1_loss((1 - mask) * out, (1 - mask) * gt)

        #L_perceptual
        out_feature_map = self.vgg_map(out)
        comp_feature_map = self.vgg_map(comp)
        gt_feature_map = self.vgg_map(gt)

        loss_dict['perceptual'] = 0.0
        for i in range(3):
            loss_dict['perceptual'] += self.l1_loss(out_feature_map[i], gt_feature_map[i]) + self.l1_loss(comp_feature_map[i], gt_feature_map[i])

        # L_style_out, L_style_comp
        loss_dict['style'] = 0.0
        for i in range(3):
            print(out_feature_map[i].shape)
            loss_dict['style'] += self.l1_loss(self.gram_matrix(out_feature_map[i]), self.gram_matrix(gt_feature_map[i]))
            loss_dict['style'] += self.l1_loss(self.gram_matrix(comp_feature_map[i]), self.gram_matrix(gt_feature_map[i]))

        # L_tv (total variation)
        loss_dict['tv'] = self.l1_loss(comp[:, :, :, :-1], comp[:, :, :, 1:]) + self.l1_loss(comp[:, :, :-1, :], comp[:, :, 1:, :])

        return loss_dict




if __name__ == '__main__':
    net = VGG16map()
    size = (3, 3, 256, 256)
    input = torch.randn(size)
    mask = torch.ones(size)
    ground_truth = torch.randn(size)
    out = torch.randn(size)
    out1, out2, out3 = net(input)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
    losses = Losses()
    print(losses(input, mask, out, ground_truth))