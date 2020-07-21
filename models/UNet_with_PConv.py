import torch.nn.functional as F
import torch.nn as nn
import torch

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, batch_norm=False, non_linearity=None):
        super(PartialConv, self).__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        self.input_conv.apply(self.weights_init)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if non_linearity == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif non_linearity == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)


        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # fix
        for param in self.mask_conv.parameters():
            param.requires_grad = False


    def forward(self, input, mask):
        output = self.input_conv(input*mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        #0:True
        mask_0 = output_mask == 0

        #caluculate sum(M)
        mask_sum = output_mask.masked_fill_(mask_0, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(mask_0, 0.0)

        # x' 0 (otherwize)
        new_m = torch.ones_like(output)
        new_m = new_m.masked_fill_(mask_0, 0.0)


        if hasattr(self, 'bn'):
            output = self.batch_norm(output)
        if hasattr(self, 'activation'):
            output = self.activation(output)

        return output, new_m

    def weights_init(self, m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

class PConvUNet(nn.Module):
    def __init__(self, input_channels=3):
        super(PConvUNet, self).__init__()

        self.fine_tune = False

        self.pconv1 = PartialConv(input_channels, 64, kernel_size=7, stride=2, padding=3, non_linearity='relu')

        self.pconv2 = PartialConv(64, 128, kernel_size=5, stride=2, padding=2, batch_norm=True, non_linearity='relu')

        self.pconv3 = PartialConv(128, 256, kernel_size=5, stride=2, padding=2, batch_norm=True, non_linearity='relu')

        self.pconv4 = PartialConv(256, 512, kernel_size=3, stride=2, padding=1, batch_norm=True, non_linearity='relu')

        #self.pconv5~8
        for i in range(4, 8):
            setattr(
                self, 'pconv'+str((i+1)),
                PartialConv(512, 512, kernel_size=3, stride=2, padding=1, batch_norm=True, non_linearity='relu'))

        #self.pconv9~12
        for i in range(8, 12):
            setattr(
                self, 'pconv'+str((i+1)),
                PartialConv(512+512, 512, kernel_size=3, stride=1, padding=1, batch_norm=True, non_linearity='leakyrelu'))

        self.pconv13 = PartialConv(512+256, 256, kernel_size=3, stride=1, padding=1, batch_norm=True, non_linearity='leakyrelu')

        self.pconv14 = PartialConv(256+128, 128, kernel_size=3, stride=1, padding=1, batch_norm=True, non_linearity='leakyrelu')

        self.pconv15 = PartialConv(128+64, 64, kernel_size=3, stride=1, padding=1, batch_norm=True, non_linearity='leakyrelu')

        self.pconv16 = PartialConv(64+3, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, input, mask):
        x1, m1 = self.pconv1(input, mask)
        x2, m2 = self.pconv2(x1, m1)
        x3, m3 = self.pconv3(x2, m2)
        x4, m4 = self.pconv4(x3, m3)
        x5, m5 = self.pconv5(x4, m4)
        x6, m6 = self.pconv6(x5, m5)
        x7, m7 = self.pconv7(x6, m6)
        x8, m8 = self.pconv8(x7, m7)

        x8, m8 = F.interpolate(x8, scale_factor=2, mode='nearest'), F.interpolate(m8, scale_factor=2, mode='nearest')
        concat1, m_concat1 = torch.cat([x8, x7], dim=1), torch.cat([m8, m7], dim=1)
        x9, m9 = self.pconv9(concat1, m_concat1)

        x9, m9 = F.interpolate(x9, scale_factor=2, mode='nearest'), F.interpolate(m9, scale_factor=2, mode='nearest')
        concat2, m_concat2 = torch.cat([x9, x6], dim=1), torch.cat([m9, m6], dim=1)
        x10, m10 = self.pconv10(concat2, m_concat2)

        x10, m10 = F.interpolate(x10, scale_factor=2, mode='nearest'), F.interpolate(m10, scale_factor=2, mode='nearest')
        concat3, m_concat3 = torch.cat([x10, x5], dim=1), torch.cat([m10, m5], dim=1)
        x11, m11 = self.pconv11(concat3, m_concat3)

        x11, m11 = F.interpolate(x11, scale_factor=2, mode='nearest'), F.interpolate(m11, scale_factor=2, mode='nearest')
        concat4, m_concat4 = torch.cat([x11, x4], dim=1), torch.cat([m11, m4], dim=1)
        x12, m12 = self.pconv12(concat4, m_concat4)

        x12, m12 = F.interpolate(x12, scale_factor=2, mode='nearest'), F.interpolate(m12, scale_factor=2, mode='nearest')
        concat5, m_concat5 = torch.cat([x12, x3], dim=1), torch.cat([m12, m3], dim=1)
        x13, m13 = self.pconv13(concat5, m_concat5)

        x13, m13 = F.interpolate(x13, scale_factor=2, mode='nearest'), F.interpolate(m13, scale_factor=2, mode='nearest')
        concat6, m_concat6 = torch.cat([x13, x2], dim=1), torch.cat([m13, m2], dim=1)
        x14, m14 = self.pconv14(concat6, m_concat6)

        x14, m14 = F.interpolate(x14, scale_factor=2, mode='nearest'), F.interpolate(m14, scale_factor=2, mode='nearest')
        concat7, m_concat7 = torch.cat([x14, x1], dim=1), torch.cat([m14, m1], dim=1)
        x15, m15 = self.pconv15(concat7, m_concat7)

        x15, m15 = F.interpolate(x15, scale_factor=2, mode='nearest'), F.interpolate(m15, scale_factor=2, mode='nearest')
        concat8, m_concat8 = torch.cat([x15, input], dim=1), torch.cat([m15, mask], dim=1)
        out, out_mask = self.pconv16(concat8, m_concat8)

        return out, out_mask


    def train(self, mode=True):
        super().train(mode)
        if self.fine_tune:
            for name, module in self.named_modules():
                if name:
                    if isinstance(module, nn.BatchNorm2d) and 1<=int(name.split('.')[0][5:])<=8:
                        module.eval()


if __name__ == '__main__':
    size = (3, 3, 256, 256)
    input = torch.ones(size)
    input_mask = torch.ones(size)
    l1 = nn.L1Loss()
    input.requires_grad = True

    conv = PartialConv(3, 3, 3, 1, 1)
    output, output_mask = conv(input, input_mask)
    print(output.shape)
    print(output_mask.shape)
    loss = l1(output, torch.randn(3, 3, 256, 256))
    loss.backward()
    print(loss.item())

    model = PConvUNet()
    output, output_mask = model(input, input_mask)
    print(output.shape)
    print(output_mask.shape)
    loss = l1(output, torch.randn(3, 3, 256, 256))
    loss.backward()
    print(loss.item())

