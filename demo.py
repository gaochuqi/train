import argparse
import logging
import os
from datetime import datetime
from functools import partial
import torch
import torch.utils.data as torch_data
from torch.utils.data import Dataset
import torch.nn as nn

import onnx
import random
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from torchvision import models
import collections
from PIL import Image
import matplotlib
print(matplotlib.get_backend())
# matplotlib.use('Qt5Agg')
print(matplotlib.get_backend())
from matplotlib import pyplot as plt
# matplotlib.pyplot.switch_backend('agg')
import numpy as np
import cv2
import glob
import onnxruntime
import time
import config as cfg
from arch import arch_qat
from arch import architecture_reparam # architecture_qat
from utils import netloss
from utils.polynomialCurveFitting import *
from arch import architecture
from dataset.dataset import preprocess, tensor2numpy, pack_gbrg_raw, norm_raw
from generate import unprocess
from generate.unpack_rrggbbrr import pack_1raw_to_4raw
from generate.tools import setup_seed, pixel_unShuffle_RGBG, get_fit_curve
from generate.raw2uint16 import read_mipi10
from utils.models import models



def inference(test_name,
             frame_list,
             iso_list,
             lr_a, lr_b,
              log_dir):
    order = 3
    poly = PolynomialFeatures(order)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cfg.model_name = 'model_deLoss' # 'model' #
    # log_dir = 'log/{}/'.format(cfg.model_name)

    model_path = os.path.join(log_dir, 'model.pth') # 'model_best.pth')
    model = architecture.EMVD(cfg)
    model = model.to(device)
    ckpt = torch.load(model_path)
    state_dict = ckpt['model']
    temp = collections.OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            name = '.'.join(k.split('.')[1:])
            temp[name] = v
    if len(temp):
        state_dict = temp
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # for name, module_ref in model.named_modules():
    #     print(name, module_ref)

    isp = models.ISP().cuda()
    isp_state_dict = torch.load('utils/models/ISP.pth')
    isp.load_state_dict(isp_state_dict)
    isp.eval()

    pixel_shuffle = architecture.PixelShuffle(2)
    pixel_unshuffle = architecture.PixelShuffle(0.5)

    log_path = log_dir + '/demo/{}/'.format(test_name) + '{}/'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    output_dir = log_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir + 'noisy/'):
        os.mkdir(output_dir + 'noisy/')
    if not os.path.exists(output_dir + 'fusion/'):
        os.mkdir(output_dir + 'fusion/')
    if not os.path.exists(output_dir + 'denoise/'):
        os.mkdir(output_dir + 'denoise/')
    if not os.path.exists(output_dir + 'refine/'):
        os.mkdir(output_dir + 'refine/')
    if not os.path.exists(output_dir + 'omega/'):
        os.mkdir(output_dir + 'omega/')
    if not os.path.exists(output_dir + 'gamma/'):
        os.mkdir(output_dir + 'gamma/')

    pad_y = 0
    pad_x = 8

    with torch.no_grad():
        for idx, frame in enumerate(frame_list):
            name = os.path.basename(frame).split('.')[0]
            raw_noisy = np.load(frame, mmap_mode='r')
            raw_noisy_norm = norm_raw(raw_noisy,
                                        black_level=cfg.black_level,
                                        white_level=cfg.white_level)
            raw_noisy_isp = tensor2numpy(isp(torch.from_numpy(
                                    pixel_unShuffle_RGBG(raw_noisy_norm.astype(np.float32),
                                            cfg.bayer_pattern)).cuda()))[0]
            cv2.imwrite(output_dir + 'noisy/' + '{}.png'.format(name),
                        np.uint8(raw_noisy_isp * 255))
            ########################################################
            # run model
            ########################################################
            tmp = np.pad(raw_noisy_norm, [(pad_y, pad_y), (pad_x, pad_x)])
            noisy = torch.from_numpy(np.expand_dims(np.expand_dims(tmp,
                                    axis=0), axis=0)).cuda()
            ft1 = noisy
            for i in range(cfg.px_num):
                ft1 = pixel_unshuffle(ft1)

            if idx==0:
                ft0 = ft1

            iso = iso_list[i]
            X_test = np.asarray([iso], dtype=np.float32).reshape(-1, 1) / 25600
            X_test_ploy = poly.fit_transform(X_test)
            a = lr_a.predict(X_test_ploy)[0]
            b = lr_b.predict(X_test_ploy)[0]
            coeff_a = a / (cfg.white_level - cfg.black_level)
            coeff_b = b / (cfg.white_level - cfg.black_level) ** 2

            fusion, denoise, refine, omega, gamma = model(ft0, ft1, coeff_a, coeff_b)

            ft0 = fusion
            ########################################################
            # out
            ########################################################
            for i in range(cfg.px_num - 1):
                fusion = pixel_shuffle(fusion)
                denoise = pixel_shuffle(denoise)
                refine = pixel_shuffle(refine)
            for i in range(cfg.px_num - 2):
                omega = pixel_shuffle(omega)
                gamma = pixel_shuffle(gamma)

            fusion = fusion[0, 0, :, pad_x//2:-pad_x//2]
            denoise = denoise[0, 0, :, pad_x//2:-pad_x//2]
            refine = refine[0, 0, :, pad_x//2:-pad_x//2]
            omega = omega[0, 0]
            gamma = gamma[0, 0]

            fusion_np = fusion.data.cpu().numpy()       # h,w
            denoise_np = denoise.data.cpu().numpy()
            refine_np = refine.data.cpu().numpy()

            fusion_np_pxu = pixel_unShuffle_RGBG(fusion_np, cfg.bayer_pattern)
            denoise_np_pxu = pixel_unShuffle_RGBG(denoise_np, cfg.bayer_pattern)
            refine_np_pxu = pixel_unShuffle_RGBG(refine_np, cfg.bayer_pattern)

            # tmp = np.pad(fusion_np_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
            fusion_isp = tensor2numpy(isp(torch.from_numpy(fusion_np_pxu).cuda()))[0]
            cv2.imwrite(output_dir + 'fusion/{}.png'.format(name), np.uint8(fusion_isp * 255))
            # tmp = np.pad(denoise_np_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
            denoise_isp = tensor2numpy(isp(torch.from_numpy(denoise_np_pxu).cuda()))[0]
            cv2.imwrite(output_dir + 'denoise/{}.png'.format(name), np.uint8(denoise_isp * 255))
            # tmp = np.pad(refine_np_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
            refine_isp = tensor2numpy(isp(torch.from_numpy(refine_np_pxu).cuda()))[0]
            cv2.imwrite(output_dir + 'refine/{}.png'.format(name), np.uint8(refine_isp * 255))

            cv2.imwrite(output_dir + 'omega/%s.png' % name, np.uint8(omega.data.cpu().numpy() * 255))
            cv2.imwrite(output_dir + 'gamma/%s.png' % name, np.uint8(gamma.data.cpu().numpy() * 255))
    return output_dir + 'refine/'

def convert_RAWMIPI_to_uint16npy(data_dir,
                                 pattern='*.RAWMIPI10',
                                 H=3072, W=4080,
                                 out_dir=''):
    data_pattern = os.path.join(data_dir, pattern)
    data_list = glob.glob(data_pattern)
    for data_path in data_list:
        name = os.path.basename(data_path).split('.')[0]
        raw = read_mipi10(data_path, H, W).astype(np.uint16)
        np.save(os.path.join(out_dir, '{}.npy'.format(name)), raw)

def get_coeff_a_b():
    iso_list = cfg.iso_list
    a_list = cfg.a_list
    b_list = cfg.b_list
    X_train = np.asarray(iso_list, dtype=np.float32).reshape(-1, 1) / (max(iso_list)*1.1)
    a_train = np.asarray(a_list, dtype=np.float32)
    b_train = np.asarray(b_list, dtype=np.float32)
    order = 3
    lr_a = get_fit_curve(X_train, a_train, order)
    lr_b = get_fit_curve(X_train, b_train, order)
    return lr_a, lr_b

def images_to_video(images_pattern,
                    name):
    filelist = glob.glob(images_pattern)

    fps = 30  # 视频每秒 30 帧
    size = (2040, 1536)  # 需要转为视频的图片的尺寸
    # 可以使用cv2.resize()进行修改
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    # cv2.VideoWriter_fourcc('I', '4', '2', '0')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    video = cv2.VideoWriter(os.path.dirname(os.path.dirname(images_pattern))+"/{}.mp4".format(name), fourcc, fps, size)
    # 视频保存在当前目录下

    for item in filelist:
        if item.endswith('.png'):
            # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
            img = cv2.imread(item)
            video.write(img)

    video.release()
    cv2.destroyAllWindows()

def demo():
    cfg.model_name = 'model_deLoss' # 'model' #
    log_dir = 'log/minimal/{}/'.format(cfg.model_name)

    test_rawmipi = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/shuang/20211230/test'
    sub_folder = 'IMG_20211230'

    raw_dir = '/home/wen/Documents/dataset/Mi11Ultra/test'
    npy_dir = os.path.join(raw_dir, sub_folder)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    b_convert_RAW = False
    if b_convert_RAW:
        convert_RAWMIPI_to_uint16npy(test_rawmipi,
                                     out_dir=npy_dir)
    b_infer = False
    if b_infer:
        npy_list = glob.glob(os.path.join(npy_dir, '*.npy'))
        npy_list.sort()
        iso_list = [3200] * len(npy_list)
        lr_a, lr_b = get_coeff_a_b()
        refine_dir = inference(sub_folder,
                               npy_list,
                               iso_list,
                               lr_a, lr_b,
                               log_dir)
    else:
        time_folder = '2022-02-22_16:09:24'
        refine_dir = log_dir + 'demo/{}/'.format(sub_folder) + '{}/refine/'.format(time_folder)

    refine_pattern = os.path.join(refine_dir, '*.png')
    images_to_video(refine_pattern, sub_folder + '_refine')



def main():
    setup_seed(666)

    demo()

    print('end')

if __name__ == '__main__':
    main()