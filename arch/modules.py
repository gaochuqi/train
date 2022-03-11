import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import arch.block as B
import config as cfg

device = cfg.device

def save_feature(feat,name):
    import os
    import cv2
    PROJ = '/home/wen/Documents/project/video/denoising/emvd'
    a100_model = 'results/a100'
    sub_dir = a100_model
    sub_folder = 'model'
    output_dir = os.path.join(PROJ, sub_dir, sub_folder, 'outputs')
    b,c,h,w = feat.shape
    feat = feat.data.cpu().numpy()
    for i in range(c):
        f = feat[0,i,:,:]
        v_max = np.max(f)
        v_min = np.min(f)
        tmp = (f - v_min) / (v_max - v_min)
        cv2.imwrite(output_dir + '/{}_{}.png'.format(name, str(i)), np.uint8(tmp * 255))
'''
content reconstruction module, 
structural reconstruction module, 
photo-realism reconstruction module,
'''

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)

def activation(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class _ResBlock_32(nn.Module):
    def __init__(self, nc=64):
        super(_ResBlock_32, self).__init__()
        self.c1 = conv_layer(nc, nc, 3, 1, 1)
        self.d1 = conv_layer(nc, nc//2, 3, 1, 1)  # rate=1
        self.d2 = conv_layer(nc, nc//2, 3, 1, 2)  # rate=2
        self.d3 = conv_layer(nc, nc//2, 3, 1, 3)  # rate=3
        self.d4 = conv_layer(nc, nc//2, 3, 1, 4)  # rate=4
        self.d5 = conv_layer(nc, nc//2, 3, 1, 5)  # rate=5
        self.d6 = conv_layer(nc, nc//2, 3, 1, 6)  # rate=6
        self.d7 = conv_layer(nc, nc//2, 3, 1, 7)  # rate=7
        self.d8 = conv_layer(nc, nc//2, 3, 1, 8)  # rate=8
        self.act = activation('lrelu')
        self.c2 = conv_layer(nc * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)
        d5 = self.d5(output1)
        d6 = self.d6(output1)
        d7 = self.d7(output1)
        d8 = self.d8(output1)

        add1 = d1 + d2
        add2 = add1 + d3
        add3 = add2 + d4
        add4 = add3 + d5
        add5 = add4 + d6
        add6 = add5 + d7
        add7 = add6 + d8

        combine = torch.cat([d1, add1, add2, add3, add4, add5, add6, add7], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

        return output

class ResBlock(nn.Module):
    def __init__(self, nc=32, r1=1,r2=2,r3=3,r4=4):
        super(ResBlock, self).__init__()
        self.c1 = conv_layer(nc, nc, 3, 1, 1)
        self.d1 = conv_layer(nc, nc//2, 3, 1, r1)  # rate=1
        self.d2 = conv_layer(nc, nc//2, 3, 1, r2)  # rate=3
        self.d3 = conv_layer(nc, nc//2, 3, 1, r3)  # rate=6
        self.d4 = conv_layer(nc, nc//2, 3, 1, r4)  # rate=9
        self.act = activation('relu')
        self.c2 = conv_layer(nc // 2 * 4, nc, 1, 1, 1)  # 256-->64

    def forward(self, input):
        output1 = self.act(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        add1 = d1
        add2 = add1 + d2
        add3 = add2 + d3
        add4 = add3 + d4

        combine = torch.cat([add1, add2, add3, add4], 1)
        output2 = self.c2(self.act(combine))
        output = input + output2.mul(0.2)

        return output

class Realism(nn.Module):
    def __init__(self, nc=32):
        super(Realism, self).__init__()
        self.rb1 = _ResBlock_32(nc) # ResBlock(nc,1,2,3,4) # _ResBlock_32(nc) #
        self.rb2 = _ResBlock_32(nc) # ResBlock(nc,5,6,7,8) # _ResBlock_32(nc) #
        self.UpC1 = nn.ConvTranspose2d(nc, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.UpC2 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.UpC3 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.OutC = nn.Conv2d(16, 4, 3, 1, padding=1, bias=False, dilation=1, groups=1)
        self.act = activation('relu')

    def forward(self, input, refine=None):
        t = self.rb1(input)
        t = self.rb2(t)
        b,c,h,w = input.size()
        y = self.act(self.UpC1(t, output_size=[b,c,h*2,w*2]))
        y = self.act(self.UpC2(y, output_size=[b,c,h*4,w*4]))
        y = self.act(self.UpC3(y, output_size=[b,c,h*8,w*8]))
        output = self.OutC(y)
        if refine is not None:
            output += refine
        return output


#########################
# Discriminator
#########################
class Discriminator(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='lrelu'):
        super(Discriminator, self).__init__()

        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type)  # 3-->64
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type,  # 64-->64, 96*96
                             act_type=act_type)

        conv2 = B.conv_block(base_nf, base_nf * 2, kernel_size=3, stride=1, norm_type=norm_type,  # 64-->128
                             act_type=act_type)
        conv3 = B.conv_block(base_nf * 2, base_nf * 2, kernel_size=4, stride=2, norm_type=norm_type,  # 128-->128, 48*48
                             act_type=act_type)

        conv4 = B.conv_block(base_nf * 2, base_nf * 4, kernel_size=3, stride=1, norm_type=norm_type,  # 128-->256
                             act_type=act_type)
        conv5 = B.conv_block(base_nf * 4, base_nf * 4, kernel_size=4, stride=2, norm_type=norm_type,  # 256-->256, 24*24
                             act_type=act_type)

        conv6 = B.conv_block(base_nf * 4, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type,  # 256-->512
                             act_type=act_type)
        conv7 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type,  # 512-->512 12*12
                             act_type=act_type)

        conv8 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type,  # 512-->512
                             act_type=act_type)
        conv9 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type,  # 512-->512 6*6
                             act_type=act_type)
        conv10 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, norm_type=norm_type,
                              act_type=act_type)
        conv11 = B.conv_block(base_nf * 8, base_nf * 8, kernel_size=4, stride=2, norm_type=norm_type,  # 3*3
                              act_type=act_type)

        self.features = B.sequential(conv0, conv1, conv2,
                                     conv3, conv4, conv5,
                                     conv6, conv7, conv8,
                                     conv9, conv10, conv11)

        self.classifier2 = nn.Sequential(
            nn.Linear(base_nf * 8 * 3 * 3, 128), # 512
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1))

    def forward(self, x):
        t = self.features(x)
        y = t.contiguous().view(t.size(0), -1)
        x = self.classifier2(y)
        return x

#######################################################
# PixelShuffle & PixelUnShuffle         channel first
#######################################################
def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)

class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)

#######################################################
# PixelShuffle & PixelUnShuffle         channel last
#######################################################
def _depthtospace(self, tensor, scale_factor):
    """
    将图像像素从排列，与 self._spacetodepth 效果相反
    eg. scale=2 （1, 256, 256, 12） -> (1, 512, 512, 3)
    input:.
        tensor ： 输入tensor (N,H,W,C)
        scale_factor : 转换倍数
    output :
        tensor : 输入tensor (N,H,W,C)
    """
    tensor = tensor.transpose((0, 3, 1, 2))
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, scale_factor, scale_factor, new_ch, height, width])
    tensor = tensor.transpose([0, 3, 4, 1, 5, 2])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    tensor = tensor.transpose((0, 2, 3, 1))

    return tensor

def _spacetodepth(self, tensor, scale_factor):
    """
    将图像像素从排列，与 self._depthtospace 效果相反
    eg. scale=2 （1, 512, 512, 3） -> (1, 256, 256, 12)
    input:
        tensor ： 输入tensor (N,H,W,C)
        scale_factor : 转换倍数
    output :
        tensor : 输入tensor (N,H,W,C)
    """

    tensor = tensor.transpose((0, 3, 1, 2))
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    tensor = tensor.transpose([0, 3, 5, 1, 2, 4])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    tensor = tensor.transpose((0, 2, 3, 1))

    return tensor

class binning_CRVD(nn.Module): # 64 => 16
    def __init__(self):
        super(binning, self).__init__()
        tmp = np.ones((1, 4, 1, 1), dtype=np.float32) * (1 / 4)
        tmp = torch.from_numpy(tmp).float()
        # gb, b, r, gr
        self.bin_gb = torch.nn.Parameter(tmp, requires_grad=True)
        self.bin_b = torch.nn.Parameter(tmp, requires_grad=True)
        self.bin_r = torch.nn.Parameter(tmp, requires_grad=True)
        self.bin_gr = torch.nn.Parameter(tmp, requires_grad=True)
    # | noisy1 | noisy2 | noisy3 | noisy4 | => | gbrg | gbrg | gbrg | gbrg | =>
    # | ggggbbbbrrrrgggg | ggggbbbbrrrrgggg | ggggbbbbrrrrgggg | ggggbbbbrrrrgggg |
    # => | ggggbbbbrrrrgggg |
    def forward(self, x):
        n=16
        bin = torch.zeros((16, 64, 1, 1), device=self.bin_gb.device)
        for i in range(4):
            for j in range(4):
                bin[i, j * n + i, :, :] = self.bin_gb[:, j, :, :]
                bin[4 + i, j * n + i + 4, :, :] = self.bin_b[:, j, :, :]
                bin[8 + i, j * n + i + 8, :, :] = self.bin_r[:, j, :, :]
                bin[12 + i, j * n + i + 12, :, :] = self.bin_gr[:, j, :, :]
        out = F.conv2d(x, bin, stride=1, padding=0, bias=None)
        return out

class binning(nn.Module): # 64 => 16
    def __init__(self):
        super(binning, self).__init__()
        tmp = np.ones((1, 4, 1, 1), dtype=np.float32) * (1 / 4)
        tmp = torch.from_numpy(tmp).float()
        # gb, b, r, gr
        self.bin_gb = torch.nn.Parameter(tmp, requires_grad=True)
        self.bin_b = torch.nn.Parameter(tmp, requires_grad=True)
        self.bin_r = torch.nn.Parameter(tmp, requires_grad=True)
        self.bin_gr = torch.nn.Parameter(tmp, requires_grad=True)

    def forward(self, x):
        n=16
        bin = torch.zeros((16, 64, 1, 1), device=self.bin_gb.device)
        for i in range(4):
            for j in range(4):
                bin[i, j * n + i, :, :] = self.bin_r[:, j, :, :]
                bin[4 + i, j * n + i + 4, :, :] = self.bin_gr[:, j, :, :]
                bin[8 + i, j * n + i + 8, :, :] = self.bin_gb[:, j, :, :]
                bin[12 + i, j * n + i + 12, :, :] = self.bin_b[:, j, :, :]
        out = F.conv2d(x, bin, stride=1, padding=0, bias=None)
        return out

class binnings_CRVD(nn.Module): # 64 => 16
    def __init__(self):
        super(binnings_CRVD, self).__init__()
        c = 4 ** cfg.px_num
        n = 4
        tmp = np.ones((c // n, n, 1, 1), dtype=np.float32) * (1 / n)
        tmp = torch.from_numpy(tmp).float().to(device)
        self.groupn = torch.nn.Parameter(tmp, requires_grad=True)
    # CRVD
    # | noisy1 | noisy2 | noisy3 | noisy4 | => | gbrg | gbrg | gbrg | gbrg | =>
    # | ggggbbbbrrrrgggg | ggggbbbbrrrrgggg | ggggbbbbrrrrgggg | ggggbbbbrrrrgggg |
    # => | ggggbbbbrrrrgggg |

    def forward(self, x):
        c = 4 ** cfg.px_num
        n = 4
        g = c // n
        filters = torch.zeros((g, c, 1, 1)).to(x.device)
        for i in range(g):
            for j in range(n):
                filters[i, i + j * g, :, :] = self.groupn[i, j, :, :]
        out = F.conv2d(x, filters, stride=1, padding=0, bias=None)
        return out
