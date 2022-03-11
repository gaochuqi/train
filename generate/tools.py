import numpy as np
import rawpy
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import cv2
import random
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tensor2numpy(raw):  # raw: 1 * 4 * H * W
    input_full = raw.permute((0, 2, 3, 1))   # 1 * H * W * 4
    input_full = input_full.data.cpu().numpy()
    output = np.clip(input_full,0,1)
    return output

def pixel_unShuffle(im):
    # h, w -> 1, 4, h/2, w/2
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out = np.stack((im[0:H:2, 0:W:2],
                    im[0:H:2, 1:W:2],
                    im[1:H:2, 0:W:2],
                    im[1:H:2, 1:W:2]), axis=0)
    out = np.expand_dims(out, axis=0)
    return out

def pixel_unShuffle_RGBG(im, bayer_pattern='RGGB'):
    # h, w -> 1, 4, h/2, w/2
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    if bayer_pattern == 'RGGB':
        out = np.stack((im[0:H:2, 0:W:2],  # R
                        im[0:H:2, 1:W:2],  # G
                        im[1:H:2, 1:W:2],  # B
                        im[1:H:2, 0:W:2]), axis=0)  # G
    elif bayer_pattern == 'GBRG':
        out = np.stack((im[1:H:2, 0:W:2],  # R
                        im[1:H:2, 1:W:2],  # G
                        im[0:H:2, 1:W:2],  # B
                        im[0:H:2, 0:W:2]), axis=0)  # G
    elif bayer_pattern == 'GRBG':
        out = np.stack((im[0:H:2, 1:W:2],  # R
                        im[0:H:2, 0:W:2],  # G
                        im[1:H:2, 0:W:2],  # B
                        im[1:H:2, 1:W:2]), axis=0)  # G
    elif bayer_pattern == 'BGGR':
        out = np.stack((im[1:H:2, 1:W:2],  # R
                        im[1:H:2, 0:W:2],  # G
                        im[0:H:2, 0:W:2],  # B
                        im[0:H:2, 1:W:2]), axis=0)  # G
    out = np.expand_dims(out, axis=0)
    return out

# 1, 2
# 3, 4
def binning_raw(im):
    # h,w -> h/2,w/2
    bayer_images = pixel_unShuffle(im)
    avgpool = torch.nn.AvgPool2d(2, 2)
    ft = avgpool(torch.from_numpy(bayer_images.astype(np.float32)))
    px = nn.PixelShuffle(2)
    out = px(ft).squeeze().data.cpu().numpy() # .astype(np.uint16)
    return out


def convert_to_GBRG(im, bayer_pattern='RGGB'):
    # h, w -> 1, 4, h/2, w/2
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    if bayer_pattern == 'RGGB':
        out = np.stack((im[1:H:2, 0:W:2],
                        im[1:H:2, 1:W:2],
                        im[0:H:2, 0:W:2],
                        im[0:H:2, 1:W:2],
                        ), axis=0)
    elif bayer_pattern == 'GBRG':
        return im
        # out = np.stack((im[1:H:2, 0:W:2],
        #                 im[1:H:2, 1:W:2],
        #                 im[0:H:2, 1:W:2],
        #                 im[0:H:2, 0:W:2]), axis=0)
    elif bayer_pattern == 'GRBG':
        out = np.stack((im[1:H:2, 1:W:2],
                        im[1:H:2, 0:W:2],
                        im[0:H:2, 1:W:2],
                        im[0:H:2, 0:W:2],
                        ), axis=0)
    elif bayer_pattern == 'BGGR':
        out = np.stack((im[0:H:2, 1:W:2],
                        im[0:H:2, 0:W:2],
                        im[1:H:2, 1:W:2],
                        im[1:H:2, 0:W:2],), axis=0)
    ft = np.expand_dims(out, axis=0)
    px = nn.PixelShuffle(2)
    # 1, 2
    # 3, 4
    out = px(torch.from_numpy(ft)).squeeze().data.cpu().numpy().astype(np.uint16)
    return out

def get_fit_curve(X_train, Y_train, order = 3):
    poly = PolynomialFeatures(order)
    X_train_ploy = poly.fit_transform(X_train)
    lr_fit = LinearRegression()
    lr_fit.fit(X_train_ploy, Y_train)
    return lr_fit

def norm_raw(raw, black_level, white_level):
    im = raw.astype(np.float32)
    out = np.maximum(im - black_level, 0) / (white_level-black_level) # 0 - 1
    return out