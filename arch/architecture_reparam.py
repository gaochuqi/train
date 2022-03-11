import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg
from torch.autograd import Variable
import arch.block as B
import arch.EdgeOrientedConvBlock as ecb
from arch.modules import Realism

device = cfg.device

# cfa = np.array(
#     [[0.5, 0.5, 0.5, 0.5],
#      [-0.5, 0.5, 0.5, -0.5],
#      [0.65, 0.2784, -0.2784, -0.65],
#      [-0.2784, 0.65, -0.65, 0.2764]])

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


class ColorTransfer(nn.Module):
    def __init__(self):
        super(ColorTransfer, self).__init__()
        n = 4 ** (cfg.px_num - 1)
        self.net = nn.Conv2d(n, n, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.net(x)
        return out


class ColorTransferInv(nn.Module):
    def __init__(self):
        super(ColorTransferInv, self).__init__()
        n = 4 ** (cfg.px_num - 1)
        self.net = nn.Conv2d(n, n, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        out = self.net(x)
        return out


class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        n = 4 ** (cfg.px_num - 1)
        self.net = nn.Conv2d(n, n * 4, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        out = self.net(x)
        return out


class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        n = 4 ** (cfg.px_num - 1)
        self.net = nn.ConvTranspose2d(n * 4, n, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        out = self.net(x)
        return out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        n = 4  # ** (cfg.px_num - 1)
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
        if cfg.use_attFeatFusion:
            return net3
        else:
            out = self.sigmoid(net3)
            return out


class FusionM(nn.Module):
    def __init__(self):
        super(FusionM, self).__init__()
        n = 4  # ** (cfg.px_num - 1)
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
        n = 4  # ** (cfg.px_num - 1)
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
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0), )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(20, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0), )
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ft0, ft1, coeff_a, coeff_b):
        c = 4 ** (cfg.px_num - 1)
        ll0 = ft0[:, 0:c, :, :]
        ll1 = ft1[:, 0:c, :, :]
        sigma_ll0 = torch.clamp(ll0[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
        sigma_ll1 = torch.clamp(ll1[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
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
        fusion_out = ft0 * (1 - gammaM) + ft1 * gammaM

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
            channels = 32  # 64
            r = 4
            inter_channels = int(channels // r)
            self.local_att = nn.Sequential(
                nn.Conv2d(24, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0), )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(24, channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels, 4, kernel_size=1, stride=1, padding=0), )
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ft0, ft1, gamma_up, denoise_down, coeff_a, coeff_b):
        c = 4 ** (cfg.px_num - 1)
        ll0 = ft0[:, 0:c, :, :]
        ll1 = ft1[:, 0:c, :, :]
        # fusion
        sigma_ll0 = torch.clamp(ll0[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
        sigma_ll1 = torch.clamp(ll1[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
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
        fusion_out = ft0 * (1 - gammaM) + ft1 * gammaM

        # denoise
        sigma = (1 - gamma) * (1 - gamma) * sigma_ll0 + gamma * gamma * sigma_ll1
        denoise_in = torch.cat([fusion_out, ll1, denoise_down, sigma], dim=1)
        denoise_out = self.denoise(denoise_in)

        return fusion_out, denoise_out, gamma, sigma


class binnings(nn.Module):  # 64 => 16
    def __init__(self):
        super(binnings, self).__init__()
        c = 4 ** cfg.px_num
        n = 4
        self.conv_bin = nn.Conv2d(c, c // n, kernel_size=1, stride=1, padding=0, bias=False)

    # DRV
    # | noisy |  => | rggb | => | rrrr | gggg | gggg | bbbb | =>
    # | rrrrrrrrrrrrrrrr | gggggggggggggggg | gggggggggggggggg | bbbbbbbbbbbbbbbb |
    # => | rrrrggggggggbbbb |
    def forward(self, x):
        out = self.conv_bin(x)
        return out


class EMVD(nn.Module):
    def __init__(self, cfg):
        super(EMVD, self).__init__()
        # self.binnings = binnings()
        # self.binnings_0 = binnings()
        # self.binnings_1 = binnings()
        self.cfg = cfg
        # self.ct = ColorTransfer()
        self.ct_0 = ColorTransfer()
        self.ct_1 = ColorTransfer()
        # self.cti = ColorTransferInv()
        self.cti_fu = ColorTransferInv()
        self.cti_de = ColorTransferInv()
        self.cti_re = ColorTransferInv()
        # self.ft = FreTransfer()
        self.ft_00 = FreTransfer()
        self.ft_10 = FreTransfer()
        self.ft_01 = FreTransfer()
        self.ft_11 = FreTransfer()
        self.ft_02 = FreTransfer()
        self.ft_12 = FreTransfer()
        # self.fti = FreTransferInv()
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
            self.fti_realism = FreTransferInv()
            self.cti_realism = ColorTransferInv()
        if self.cfg.use_ecb:
            module_nums = 4
            n = 4 ** (cfg.px_num - 1)
            cin = 33 if cfg.use_sigma else 32
            cin *= n
            channel_nums = 16 * n
            cout = 16 * n
            ##########################################################
            reps = []
            reps += [ecb.Conv3X3(cin, channel_nums)]
            for i in range(module_nums):
                reps += [ecb.Conv3X3(channel_nums, channel_nums)]
            reps += [ecb.Conv3X3(channel_nums, cout)]
            self.eocb = nn.Sequential(*reps)
            ##########################################################
        c = (4 ** cfg.px_num) * 4
        self.conv = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight = torch.nn.Parameter(tilex16)
        self.conv.weight.requires_grad = False

    def forward(self, ft0, ft1, a=1, b=1):
        coeff_a = a / (cfg.white_level - cfg.black_level)
        coeff_b = b / (cfg.white_level - cfg.black_level) ** 2
        # print(coeff_a)
        # print(coeff_b)
        c = 4 ** (cfg.px_num - 1)

        # if ft0.shape == ft1.shape:
        #     ft0 = self.binnings(ft0)
        #
        # ft1 = self.binnings(ft1)

        tmp0 = self.ct_0(ft0)
        tmp1 = self.ct_1(ft1)
        ft0_d0 = self.ft_00(tmp0)
        ft1_d0 = self.ft_10(tmp1)

        ft0_d1 = self.ft_01(ft0_d0[:, 0:c, :, :])
        ft1_d1 = self.ft_11(ft1_d0[:, 0:c, :, :])
        ft0_d2 = self.ft_02(ft0_d1[:, 0:c, :, :])
        ft1_d2 = self.ft_12(ft1_d1[:, 0:c, :, :])

        fusion_out_d2, denoise_out_d2, gamma_d2 = self.vd(ft0_d2, ft1_d2, coeff_a, coeff_b)
        denoise_up_d1 = self.fti_d2(denoise_out_d2)

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
        # refine_out = torch.mul(denoise_out, (1 - omega)) + torch.mul(fusion_out, omega)
        omegaM = self.conv(omega)
        refine_out = denoise_out * (1 - omegaM) + fusion_out * omegaM
        if self.cfg.use_ecb:
            ecb = self.eocb(refine_in)
            refine_out += ecb

        tmp = self.fti_fu(fusion_out)
        fusion = self.cti_fu(tmp)

        tmp = self.fti_de(denoise_out)
        denoise = self.cti_de(tmp)

        tmp = self.fti_re(refine_out)
        refine = self.cti_re(tmp)

        return fusion, denoise, refine, omega, gamma

################################################################################
