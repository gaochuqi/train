import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg
import arch.EdgeOrientedConvBlock as ecb
from arch.modules import Realism

device = cfg.device

# r,gr,b,gb
cfa = np.array(
    [[0.5, 0.5, 0.5, 0.5],
     [-0.5, 0.5, 0.5, -0.5],
     [0.65, 0.2784, -0.2784, -0.65],
     [-0.2784, 0.65, -0.65, 0.2764]])

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
        self.w = torch.nn.Parameter(cfa, requires_grad=True)

    # | gggg | bbbb | rrrr | gggg | => | yyyy | uuuu | vvvv | wwww |
    def forward(self, x):
        n = 4 ** (cfg.px_num - 1)
        cfan = torch.zeros((n, n, 1, 1), device=self.w.device)
        c = 4  # n // 4
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
        n = 4 ** (cfg.px_num - 1)
        cfan_inv = torch.zeros((n, n, 1, 1), device=self.w.device)
        c = 4  # n // 4
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
        self.w1 = torch.nn.Parameter(h0_row, requires_grad=True)
        self.w2 = torch.nn.Parameter(h1_row, requires_grad=True)

    # | yyyy | uuuu | vvvv | wwww |
    # => lly1 lly2 lly3 lly4 llu1 llu2 llu3 llu4 llv1 llv2 llv3 llv4 llw1 llw2 llw3 llw4
    def forward(self, x):
        h0_row = self.w1
        h1_row = self.w2
        h0_row_t = self.w1.transpose(2, 3)
        h1_row_t = self.w2.transpose(2, 3)
        h00_row = h0_row * h0_row_t  # 1,1,2,2
        h01_row = h0_row * h1_row_t
        h10_row = h1_row * h0_row_t
        h11_row = h1_row * h1_row_t
        filters1 = [h00_row, h01_row, h10_row, h11_row]
        n = 4 ** (cfg.px_num - 1)
        filters_ft = torch.zeros((n * 4, n, 2, 2), device=h00_row.device)
        for i in range(4):
            for j in range(n):
                filters_ft[n * i + j, j, :, :] = filters1[i][0, 0, :, :]
        out = F.conv2d(x, filters_ft, stride=(2, 2), padding=0, bias=None)
        return out


class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        self.w1 = torch.nn.Parameter(g0_col, requires_grad=True)
        self.w2 = torch.nn.Parameter(g1_col, requires_grad=True)

    # lly1 lly2 lly3 lly4 llu1 llu2 llu3 llu4 llv1 llv2 llv3 llv4 llw1 llw2 llw3 llw4
    # => | yyyy | uuuu | vvvv | wwww |
    def forward(self, x):
        g0_col = self.w1
        g1_col = self.w2
        g0_col_t = g0_col.transpose(2, 3)
        g1_col_t = g1_col.transpose(2, 3)
        g00_col = g0_col * g0_col_t
        g01_col = g0_col * g1_col_t
        g10_col = g1_col * g0_col_t
        g11_col = g1_col * g1_col_t
        filters2 = [g00_col, g10_col, g01_col, g11_col]
        n = 4 ** (cfg.px_num - 1)
        filters_fti = torch.zeros((n * 4, n, 2, 2), device=g00_col.device)
        for i in range(4):
            for j in range(n):
                filters_fti[n * i + j, j, :, :] = filters2[i][0, 0, :, :]
        out = F.conv_transpose2d(x, filters_fti, stride=(2, 2), padding=0, bias=None)
        return out


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        n = 4
        cin = 5
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
        n = 4
        cin = 6
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


class Denoise(nn.Module):
    def __init__(self):
        super(Denoise, self).__init__()
        n = 4
        cin = 21
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
        n = 4
        cin = 25
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
        n = 4
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


class MainDenoise(nn.Module):
    def __init__(self):
        super(MainDenoise, self).__init__()
        self.ct = ColorTransfer()
        self.cti = ColorTransferInv()
        self.ft = FreTransfer()
        self.fti = FreTransferInv()
        self.vd = VD()
        self.md1 = MVD()
        self.md0 = MVD()
        self.refine = Refine()

        c = 4 ** cfg.px_num
        self.conv = nn.Conv2d(c // 16, c, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv.weight = torch.nn.Parameter(tilex16)
        self.conv.weight.requires_grad = False

    def forward(self, ft0, ft1, coeff_a=1, coeff_b=1):
        c = 4 ** (cfg.px_num - 1)

        tmp0 = self.ct(ft0)
        tmp1 = self.ct(ft1)
        ft0_d0 = self.ft(tmp0)
        ft1_d0 = self.ft(tmp1)

        ft0_d1 = self.ft(ft0_d0[:, 0:c, :, :])
        ft1_d1 = self.ft(ft1_d0[:, 0:c, :, :])
        ft0_d2 = self.ft(ft0_d1[:, 0:c, :, :])
        ft1_d2 = self.ft(ft1_d1[:, 0:c, :, :])

        fusion_out_d2, denoise_out_d2, gamma_d2 = self.vd(ft0_d2, ft1_d2, coeff_a, coeff_b)
        denoise_up_d1 = self.fti(denoise_out_d2)
        # print("gamma1",gamma.shape)
        gamma_up_d1 = F.interpolate(gamma_d2, scale_factor=2)

        fusion_out_d1, denoise_out_d1, gamma_d1, sigma_d1 = self.md1(ft0_d1, ft1_d1,
                                                                     gamma_up_d1, denoise_up_d1,
                                                                     coeff_a, coeff_b)
        denoise_up_d0 = self.fti(denoise_out_d1)
        gamma_up_d0 = F.interpolate(gamma_d1, scale_factor=2)

        fusion_out, denoise_out, gamma, sigma = self.md0(ft0_d0, ft1_d0,
                                                         gamma_up_d0, denoise_up_d0,
                                                         coeff_a, coeff_b)

        # refine
        refine_in = torch.cat([fusion_out, denoise_out, sigma], dim=1)
        omega = self.refine(refine_in)
        omegaM = self.conv(omega)
        refine_out = denoise_out * (1 - omegaM) + fusion_out * omegaM

        tmp = self.fti(fusion_out)
        fusion = self.cti(tmp)

        tmp = self.fti(denoise_out)
        denoise = self.cti(tmp)

        tmp = self.fti(refine_out)
        refine = self.cti(tmp)

        # return output
        return fusion, denoise, refine, omega, gamma


################################################################################

# out = torch.ones ?????????????????????dlc?????????
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
