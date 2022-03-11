import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config as cfg
import arch.EdgeOrientedConvBlock as ecb
from arch.modules import Realism

# r,gr,b,gb
if cfg.bayer_pattern == 'RGBG':
    cfa = np.array(
        [[0.5, 0.5, 0.5, 0.5],
         [-0.5, 0.5, 0.5, -0.5],
         [0.65, 0.2784, -0.2784, -0.65],
         [-0.2784, 0.65, -0.65, 0.2764]])

# r,gr,b,gb => gb,b,r,gr
if cfg.bayer_pattern == 'GBRG':
    cfa = np.array(
        [[0.5, 0.5, 0.5, 0.5],
         [-0.5, 0.5, -0.5, 0.5],
         [-0.65, -0.2784, 0.65, 0.2784],
         [0.2764, -0.65, -0.2784, 0.65]])

if cfg.bayer_pattern == 'RGGB':
    cfa = np.array(
        [[0.5, 0.5, 0.5, 0.5],
         [-0.5, 0.5, -0.5, 0.5],
         [0.65, 0.2784, -0.65, -0.2784],
         [-0.2784, 0.65, 0.2764, -0.65]])

cfa = np.expand_dims(cfa, axis=2)
cfa = np.expand_dims(cfa, axis=3)
cfa = torch.tensor(cfa).float()  # .cuda()
cfa_inv = cfa.transpose(0, 1)
##############################################
# n = 4 ** cfg.px_num
# cfan = np.zeros((n,n,1,1),dtype=np.float32)
# c = n//4
# for i in range(4):
#     for j in range(c):
#         cfan[i*c+j,j,:,:]=cfa[i,0]
#         cfan[i*c+j,j+c,:,:]=cfa[i,1]
#         cfan[i*c+j,j+c*2,:,:]=cfa[i,2]
#         cfan[i*c+j,j+c*3,:,:]=cfa[i,3]
# cfan = torch.tensor(cfan).float()
# cfan_inv = cfan.transpose(0, 1)
###############################################

class ColorTransfer(nn.Module):
    def __init__(self):
        super(ColorTransfer, self).__init__()
        self.w = torch.nn.Parameter(cfa, requires_grad=True)
    # | gggg | bbbb | rrrr | gggg | => | yyyy | uuuu | vvvv | wwww |
    def forward(self, x):
        n = 4 ** (cfg.px_num-1)
        cfan = torch.zeros((n, n, 1, 1), device=self.w.device)
        c = 4 # n // 4
        for i in range(4):
                for j in range(c):
                    cfan[i * 4 + j, j, :, :] = self.w[i, 0]
                    cfan[i * 4 + j, j + c, :, :] = self.w[i, 1]
                    cfan[i * 4 + j, j + c * 2, :, :] = self.w[i, 2]
                    cfan[i * 4 + j, j + c * 3, :, :] = self.w[i, 3]
        out = F.conv2d(x, cfan, stride=1, padding=0, bias=None)
        return out

class ColorTransferInv(nn.Module):
    def __init__(self):
        super(ColorTransferInv, self).__init__()
        self.w = torch.nn.Parameter(cfa_inv, requires_grad=True)

    # | yyyy | uuuu | vvvv | wwww | => | gggg | bbbb | rrrr | gggg |
    def forward(self, x):
        n = 4 ** (cfg.px_num-1)
        cfan_inv = torch.zeros((n, n, 1, 1), device=self.w.device)
        c = 4 # n // 4
        for i in range(4):
            for j in range(c):
                cfan_inv[i * 4 + j, j, :, :] = self.w[i, 0]
                cfan_inv[i * 4 + j, j + c, :, :] = self.w[i, 1]
                cfan_inv[i * 4 + j, j + c * 2, :, :] = self.w[i, 2]
                cfan_inv[i * 4 + j, j + c * 3, :, :] = self.w[i, 3]
        out = F.conv2d(x, cfan_inv, stride=1, padding=0, bias=None)
        return out

class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        # dwt dec
        h0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
        h1 = np.array([-1 / math.sqrt(2), 1 / math.sqrt(2)])
        h0 = np.array(h0[::-1]).ravel()
        h1 = np.array(h1[::-1]).ravel()
        h0 = torch.tensor(h0).float().reshape((1, 1, -1))
        h1 = torch.tensor(h1).float().reshape((1, 1, -1))
        h0_col = h0.reshape((1, 1, -1, 1))  # col lowpass
        h1_col = h1.reshape((1, 1, -1, 1))  # col highpass
        h0_row = h0.reshape((1, 1, 1, -1))  # row lowpass
        h1_row = h1.reshape((1, 1, 1, -1))  # row highpass
        ll_filt = torch.cat([h0_row, h1_row], dim=0)
        self.w1 = torch.nn.Parameter(h0_row, requires_grad=True)
        self.w2 = torch.nn.Parameter(h1_row, requires_grad=True)
        h0_row = self.w1
        h1_row = self.w2
        h0_row_t = self.w1.transpose(2, 3)
        h1_row_t = self.w2.transpose(2, 3)
        h00_row = h0_row * h0_row_t  # 1,1,2,2
        h01_row = h0_row * h1_row_t
        h10_row = h1_row * h0_row_t
        h11_row = h1_row * h1_row_t
        self.h00_row = torch.nn.Parameter(h00_row, requires_grad=True)
        self.h01_row = torch.nn.Parameter(h01_row, requires_grad=True)
        self.h10_row = torch.nn.Parameter(h10_row, requires_grad=True)
        self.h11_row = torch.nn.Parameter(h11_row, requires_grad=True)

    # | yyyy | uuuu | vvvv | wwww |
    # => lly1 lly2 lly3 lly4 llu1 llu2 llu3 llu4 llv1 llv2 llv3 llv4 llw1 llw2 llw3 llw4
    def forward(self, x):

        filters1 = [self.h00_row, self.h01_row, self.h10_row, self.h11_row]
        n = 4 ** (cfg.px_num-1)
        filters_ft = torch.zeros((n*4, n, 2, 2), device=self.h00_row.device)
        for i in range(4):
            for j in range(n):
                filters_ft[n * i + j, j, :, :] = filters1[i][0, 0, :, :]
        out = F.conv2d(x, filters_ft, stride=(2, 2), padding=0, bias=None)
        return out

class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        # dwt rec
        g0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
        g1 = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
        g0 = np.array(g0).ravel()
        g1 = np.array(g1).ravel()
        g0 = torch.tensor(g0).float().reshape((1, 1, -1))
        g1 = torch.tensor(g1).float().reshape((1, 1, -1))
        g0_col = g0.reshape((1, 1, -1, 1))
        g1_col = g1.reshape((1, 1, -1, 1))
        g0_row = g0.reshape((1, 1, 1, -1))
        g1_row = g1.reshape((1, 1, 1, -1))
        self.w1 = torch.nn.Parameter(g0_col, requires_grad=True)
        self.w2 = torch.nn.Parameter(g1_col, requires_grad=True)
        g0_col = self.w1
        g1_col = self.w2
        g0_col_t = g0_col.transpose(2, 3)
        g1_col_t = g1_col.transpose(2, 3)
        g00_col = g0_col * g0_col_t
        g01_col = g0_col * g1_col_t
        g10_col = g1_col * g0_col_t
        g11_col = g1_col * g1_col_t
        self.g00_col = torch.nn.Parameter(g00_col, requires_grad=True)
        self.g01_col = torch.nn.Parameter(g01_col, requires_grad=True)
        self.g10_col = torch.nn.Parameter(g10_col, requires_grad=True)
        self.g11_col = torch.nn.Parameter(g11_col, requires_grad=True)

    # lly1 lly2 lly3 lly4 llu1 llu2 llu3 llu4 llv1 llv2 llv3 llv4 llw1 llw2 llw3 llw4
    # => | yyyy | uuuu | vvvv | wwww |
    def forward(self, x):

        filters2 = [self.g00_col, self.g10_col, self.g01_col, self.g11_col]
        n = 4 ** (cfg.px_num-1)
        filters_fti = torch.zeros((n*4, n, 2, 2), device=self.g00_col.device)
        for i in range(4):
            for j in range(n):
                filters_fti[n * i + j, j, :, :] = filters2[i][0, 0, :, :]
        out = F.conv_transpose2d(x, filters_fti, stride=(2, 2), padding=0, bias=None)
        return out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        n = 4 # ** (cfg.px_num - 1)
        cin = 5 if cfg.use_sigma else 4
        cin *= n
        c = 16 * n
        cout = n
        self.net1 = nn.Conv2d(cin, c, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(c, cout, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        net3 = self.net3(net2)
        out = self.sigmoid(net3)
        return out

class FusionM(nn.Module):
    def __init__(self):
        super(FusionM, self).__init__()
        n = 4 # ** (cfg.px_num - 1)
        cin = 6 if cfg.use_sigma else 5
        cin *= n
        c = 16 * n
        cout = n

        self.net1 = nn.Conv2d(cin, c, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(c, cout, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        net3 = self.net3(net2)
        if cfg.use_attFeatFusion:
            return net3
        else:
            out = self.sigmoid(net3)
            return out

class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        n = 4 # ** (cfg.px_num - 1)
        cin = 21 if cfg.use_sigma else 20
        cin *= n
        c = 16 * n
        cout = 16 * n
        self.net1 = nn.Conv2d(cin, c, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(c, cout, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        out = self.net3(net2)
        return out

class DenoiseM(nn.Module):
    def __init__(self):
        super(DenoiseM, self).__init__()
        n = 4  # ** (cfg.px_num - 1)
        cin = 25 if cfg.use_sigma else 24
        cin *= n
        c = 16 * n
        cout = 16 * n
        self.net1 = nn.Conv2d(cin, c, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(c, cout, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        out = self.net3(net2)
        return out

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        n = 4  # ** (cfg.px_num - 1)
        cin = 33 if cfg.use_sigma else 32
        cin *= n
        c = 16 * n
        cout = n
        self.net1 = nn.Conv2d(cin, c, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(c, cout, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        out = self.sigmoid(self.net3(net2))
        return out

c = 4 ** cfg.px_num
tilex16 = np.zeros((c, c // 16, 1, 1), dtype=np.float32)
diag = np.eye(c // 16, dtype=np.float32)
diag = np.reshape(diag, [c // 16, c // 16, 1, 1])
for i in range(16):
    tilex16[i * (c // 16):(i + 1) * (c // 16), :, :, :] = diag[:, :, :, :]
tilex16 = torch.tensor(tilex16).float()

class VD(nn.Module):
    def __init__(self):
        super(VD, self).__init__()
        self.fusion = Fusion()
        self.denoise = Denoise()
        c = 4 ** cfg.px_num
        self.conv = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight = torch.nn.Parameter(tilex16)
        self.conv.weight.requires_grad = False
        if cfg.use_attFeatFusion:
            channels = 32
            r = 4
            inter_channels = int(channels // r)
            self.local_att = nn.Sequential(
                nn.Conv2d(20, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0),)
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(20, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0),)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ft0, ft1, coeff_a, coeff_b):
        c = 4 ** (cfg.px_num-1)
        ll0 = ft0[:, 0:c, :, :]
        ll1 = ft1[:, 0:c, :, :]
        sigma_ll0 = torch.clamp(ll0[:, 0:(c//4), :, :], 0, 1) * coeff_a + coeff_b
        sigma_ll1 = torch.clamp(ll1[:, 0:(c//4), :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)

        if cfg.use_attFeatFusion:
            ll0_sigma = torch.cat([ll0, sigma_ll0], dim=1)
            ll1_sigma = torch.cat([ll1, sigma_ll1], dim=1)
            xa = ll0_sigma + ll1_sigma
            xl = self.local_att(xa)
            xg = self.global_att(xa)
            xlg = xl + xg
            gamma += xlg
            gamma = self.sigmoid(gamma)

        gammaM = self.conv(gamma)
        fusion_out = ft0 * (1-gammaM) + ft1 * gammaM

        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, ll1, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)
        return fusion_out, denoise_out, gamma

class MVD(nn.Module):
    def __init__(self):
        super(MVD, self).__init__()
        self.fusion = FusionM()
        self.denoise = DenoiseM()
        c = 4 ** cfg.px_num
        self.conv = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight = torch.nn.Parameter(tilex16)
        self.conv.weight.requires_grad = False
        if cfg.use_attFeatFusion:
            channels = 32 # 64
            r = 4
            inter_channels = int(channels // r)
            self.local_att = nn.Sequential(
                nn.Conv2d(24, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0),)
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(24, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0),)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ft0, ft1, gamma_up, denoise_down, coeff_a, coeff_b):
        c = 4 ** (cfg.px_num-1)
        ll0 = ft0[:, 0:c, :, :]
        ll1 = ft1[:, 0:c, :, :]
        # fusion
        sigma_ll0 = torch.clamp(ll0[:, 0:(c//4), :, :], 0, 1) * coeff_a + coeff_b
        sigma_ll1 = torch.clamp(ll1[:, 0:(c//4), :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), gamma_up, sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)

        if cfg.use_attFeatFusion:
            ll0_sigma = torch.cat([ll0, gamma_up, sigma_ll0], dim=1)
            ll1_sigma = torch.cat([ll1, gamma_up, sigma_ll1], dim=1)
            xa = ll0_sigma + ll1_sigma
            xl = self.local_att(xa)
            xg = self.global_att(xa)
            xlg = xl + xg
            gamma += xlg
            gamma = self.sigmoid(gamma)
        gammaM = self.conv(gamma)
        fusion_out = ft0 * (1-gammaM) + ft1 * gammaM

        # denoise
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, ll1, denoise_down, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        return fusion_out, denoise_out, gamma, sigma

class binnings(nn.Module): # 64 => 16
    def __init__(self):
        super(binnings, self).__init__()
        c = 4 ** cfg.px_num
        n = 4
        tmp = np.ones((c // n, n, 1, 1), dtype=np.float32) * (1 / n)
        tmp = torch.from_numpy(tmp).float() #.to(device)
        self.groupn = torch.nn.Parameter(tmp, requires_grad=True)

    # DRV
    # | noisy |  => | rggb | => | rrrr | gggg | gggg | bbbb | =>
    # | rrrrrrrrrrrrrrrr | gggggggggggggggg | gggggggggggggggg | bbbbbbbbbbbbbbbb |
    # => | rrrrggggggggbbbb |

    def forward(self, x):
        c = 4 ** cfg.px_num
        n = 4
        g = c // n
        filters = torch.zeros((g, c, 1, 1)).to(x.device)
        for i in range(g):
            for j in range(n):
                filters[i, i // 4 * g + j * n + i % 4, :, :] = self.groupn[i, j, :, :]
        out = F.conv2d(x, filters, stride=1, padding=0, bias=None)
        return out

class EMVD(nn.Module):
    def __init__(self, cfg):
        super(EMVD, self).__init__()
        self.cfg = cfg

        self.binnings = binnings()
        self.binnings_0 = binnings()
        self.binnings_1 = binnings()

        self.ct = ColorTransfer()
        self.ct_0 = ColorTransfer()
        self.ct_1 = ColorTransfer()

        self.cti = ColorTransferInv()
        self.cti_fu = ColorTransferInv()
        self.cti_de = ColorTransferInv()
        self.cti_re = ColorTransferInv()

        self.ft = FreTransfer()
        self.ft_00 = FreTransfer()
        self.ft_10 = FreTransfer()
        self.ft_01 = FreTransfer()
        self.ft_11 = FreTransfer()
        self.ft_02 = FreTransfer()
        self.ft_12 = FreTransfer()

        self.fti = FreTransferInv()
        self.fti_d2 = FreTransferInv()
        self.fti_d1 = FreTransferInv()
        self.fti_fu = FreTransferInv()
        self.fti_de = FreTransferInv()
        self.fti_re = FreTransferInv()

        self.vd = VD()
        self.md1 = MVD()
        self.md0 = MVD()
        self.refine = Refine()

        if self.cfg.use_realism:
            self.realism = Realism()
        if self.cfg.use_ecb:
            module_nums = 4
            n = 4 ** (cfg.px_num - 1)
            cin = 33 if cfg.use_sigma else 32
            cin *= n
            channel_nums = 16 * n
            cout = n * 16
            ecb_block = []
            ecb_block += [ecb.ECB(cin, channel_nums)]
            for i in range(module_nums):
                ecb_block += [ecb.ECB(channel_nums, channel_nums)]
            ecb_block += [ecb.ECB(channel_nums, cout)]
            self.ecb = nn.Sequential(*ecb_block)
        c = (4 ** cfg.px_num) * 4
        self.conv = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight = torch.nn.Parameter(tilex16)
        self.conv.weight.requires_grad = False

    def forward(self, ft0, ft1, coeff_a=1, coeff_b=1):
        c = 4 ** (cfg.px_num-1)

        if ft0.shape == ft1.shape:
            ft0 = self.binnings_0(ft0)

        ft1 = self.binnings_1(ft1)

        tmp0 = self.ct_0(ft0)
        tmp1 = self.ct_1(ft1)
        ft0_d0 = self.ft_00(tmp0)
        ft1_d0 = self.ft_10(tmp1)

        ft0_d1 = self.ft_01(ft0_d0[:,0:c,:,:])
        ft1_d1 = self.ft_11(ft1_d0[:, 0:c, :, :])
        ft0_d2 = self.ft_02(ft0_d1[:,0:c,:,:])
        ft1_d2 = self.ft_12(ft1_d1[:, 0:c, :, :])

        fusion_out_d2, denoise_out_d2, gamma_d2 = self.vd(ft0_d2, ft1_d2, coeff_a, coeff_b)
        denoise_up_d1 = self.fti_d2(denoise_out_d2)
        # print("gamma1",gamma.shape)
        gamma_up_d1 = F.interpolate(gamma_d2, scale_factor=2)

        fusion_out_d1, denoise_out_d1, gamma_d1, sigma_d1 = self.md1(ft0_d1, ft1_d1,
												                    gamma_up_d1, denoise_up_d1,
												                    coeff_a, coeff_b)
        denoise_up_d0 = self.fti_d1(denoise_out_d1)
        gamma_up_d0 = F.interpolate(gamma_d1, scale_factor=2)

        fusion_out, denoise_out, gamma, sigma = self.md0(ft0_d0, ft1_d0,
		                                                gamma_up_d0, denoise_up_d0,
												        coeff_a, coeff_b)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out, sigma], dim=1)
        omega = self.refine(refine_in)
        omegaM = self.conv(omega)
        refine_out = denoise_out * (1-omegaM) + fusion_out * omegaM
        if self.cfg.use_ecb:
            ecb = self.ecb(refine_in)
            refine_out += ecb

        tmp = self.fti_fu(fusion_out)
        fusion = self.cti_fu(tmp)

        tmp = self.fti_de(denoise_out)
        denoise = self.cti_de(tmp)

        tmp = self.fti_re(refine_out)
        refine = self.cti_re(tmp)

        # return output
        if self.cfg.use_realism:
            realism = self.realism(torch.cat([fusion_out_d2, denoise_out_d2], dim=1)) + refine # refine_out
            return fusion, denoise, refine, omega, gamma, realism
        else:
            return fusion, denoise, refine, omega, gamma
