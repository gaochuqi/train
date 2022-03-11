import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg
from torch.autograd import Variable

device = cfg.device
# device = 'cpu'

cfa = np.array(
    [[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0.65, 0.2784, -0.2784, -0.65], [-0.2784, 0.65, -0.65, 0.2764]])

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
        self.net = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.net.weight = torch.nn.Parameter(cfa)

    def forward(self, x):
        out = self.net(x)
        return out


class ColorTransferInv(nn.Module):
    def __init__(self):
        super(ColorTransferInv, self).__init__()
        self.net = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.net.weight = torch.nn.Parameter(cfa_inv)

    def forward(self, x):
        out = self.net(x)
        return out

class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        self.w1 = torch.nn.Parameter(h0_row, requires_grad=True)
        self.w2 = torch.nn.Parameter(h1_row, requires_grad=True)
        self.net = nn.Conv2d(4, 16, kernel_size=2, stride=2, padding=0, bias=False)

    def forward(self, x):
        out = self.net(x)
        return out

class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        self.w1 = torch.nn.Parameter(g0_col, requires_grad=True)
        self.w2 = torch.nn.Parameter(g1_col, requires_grad=True)
        self.net = nn.ConvTranspose2d(16,4, kernel_size=2, stride=2, padding=0, bias=False)
    def forward(self, x):
        out = self.net(x)
        return out

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.net1 = nn.Conv2d(20, 64, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        out = self.sigmoid(self.net3(net2))
        return out

class FusionM(nn.Module):
    def __init__(self):
        super(FusionM, self).__init__()
        self.net1 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        net1 = self.relu1(self.net1(x))
        net2 = self.relu2(self.net2(net1))
        out = self.sigmoid(self.net3(net2))
        return out

class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        self.net1 = nn.Conv2d(84, 64, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
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
        self.net1 = nn.Conv2d(100, 64, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
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
        self.net1 = nn.Conv2d(132, 64, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1)
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

    def forward(self, ft0, ft1, coeff_a, coeff_b):
        c = 4 ** (cfg.px_num - 1)
        ll0 = ft0[:, 0:c, :, :]
        ll1 = ft1[:, 0:c, :, :]
        sigma_ll0 = torch.clamp(ll0[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
        sigma_ll1 = torch.clamp(ll1[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)

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

    def forward(self, ft0, ft1, gamma_up, denoise_down, coeff_a, coeff_b):
        c = 4 ** (cfg.px_num - 1)
        ll0 = ft0[:, 0:c, :, :]
        ll1 = ft1[:, 0:c, :, :]
        # fusion
        sigma_ll0 = torch.clamp(ll0[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
        sigma_ll1 = torch.clamp(ll1[:, 0:(c // 4), :, :], 0, 1) * coeff_a + coeff_b
        fusion_in = torch.cat([abs(ll1 - ll0), gamma_up, sigma_ll1], dim=1)
        gamma = self.fusion(fusion_in)

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
        tmp = np.ones((c // n, n, 1, 1), dtype=np.float32) * (1 / n)
        tmp = torch.from_numpy(tmp).float().to(device)
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
        self.ct0 = ColorTransfer()
        self.ct1 = ColorTransfer()
        
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
        
        self.conv = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight = torch.nn.Parameter(tilex16)
        self.conv.weight.requires_grad = False


    def forward(self, ft0, ft1, coeff_a=1, coeff_b=1):
        c = 4 ** (cfg.px_num - 1)

        if ft0.shape == ft1.shape:
            ft0 = self.binnings_0(ft0)

        ft1 = self.binnings_1(ft1)

        # print("x.shape", x.shape)


        tmp0 = self.ct0(ft0)
        tmp1 = self.ct1(ft1)
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

        fusion_out_d1, denoise_out_d1, gamma_d1, sigma_d1 = self.md1(ft0_d1, ft1_d1, gamma_up_d1, denoise_up_d1, coeff_a, coeff_b)

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


################################################################################

# out = torch.ones 这种形式，转换dlc会报错
'''
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 576235436
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:537] Reading dangerously large protocol message.  If the message turns out to be larger than 2147483647 bytes, parsing will be halted for security reasons.  To increase the limit (or to disable these warnings), see CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.
[libprotobuf WARNING google/protobuf/io/coded_stream.cc:78] The total number of bytes read was 576343306
Segmentation fault (core dumped)
'''

# conv_transpose2d
'''
AssertionError: ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED: Rectangular strides for ConvTranspose ops is not supported
2021-10-21 14:24:29,954 - 194 - ERROR - Node ConvTranspose_534: ERROR_DECONV_RECTANGULAR_STRIDE_UNSUPPORTED: Rectangular strides for ConvTranspose ops is not supported
'''

# a= torch.rand(1,8,128,128)
# net = EMVD(cfg)
# b = net(a)
