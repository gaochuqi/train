import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



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
print(cfa.shape)
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


def FreTransfer(h0_row,h1_row):

    w1 = torch.nn.Parameter(h0_row, requires_grad=True)
    w2 = torch.nn.Parameter(h1_row, requires_grad=True)

    # | yyyy | uuuu | vvvv | wwww |
    # => lly1 lly2 lly3 lly4 llu1 llu2 llu3 llu4 llv1 llv2 llv3 llv4 llw1 llw2 llw3 llw4
    def forward( w1,w2):
        h0_row = w1
        h1_row = w2
        h0_row_t = w1.transpose(2, 3)
        h1_row_t = w2.transpose(2, 3)
        h00_row = h0_row * h0_row_t  # 1,1,2,2
        h01_row = h0_row * h1_row_t
        h10_row = h1_row * h0_row_t
        h11_row = h1_row * h1_row_t
        filters1 = [h00_row, h01_row, h10_row, h11_row]
        n = 4 ** (3 - 1)
        filters_ft = torch.zeros((n * 4, n, 2, 2), device=h00_row.device)
        print(filters1)
        for i in range(4):
            for j in range(n):
                print(filters1[i][0, 0, :, :])
                filters_ft[n * i + j, j, :, :] = filters1[i][0, 0, :, :]
        print(filters_ft.shape)
        out = F.conv2d(x, filters_ft, stride=(2, 2), padding=0, bias=None)
        return out
    return forward(w1, w2)
print(FreTransfer(h0_row,h1_row))