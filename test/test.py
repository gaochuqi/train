import argparse
import copy
import glob
import logging
import os
from datetime import datetime
from functools import partial
from typing import Tuple
from torchvision import models
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import collections
from torch.utils.data import Dataset
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import cv2
# torch.multiprocessing.set_start_method('spawn')
import rawpy

import config as cfg
from ..arch import arch
from ..arch import arch_qat
from ..arch import architecture
from ..arch import structure
from ..utils import netloss
from ..dataset.dataset import *

iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
isp = torch.load('isp/ISP_CNN.pth', map_location=torch.device('cpu'))


def evaluate(model, eval_loader, isp=None):
    psnr = netloss.PSNR().to(cfg.device)
    print('Evaluate...')
    cnt = 0
    total_psnr = 0
    model.eval()
    with torch.no_grad():
        for i, (input, label) in enumerate(eval_loader):
            input = input.to(cfg.device)
            label = label.to(cfg.device)
            fusion_out, denoise_out, refine_out, omega, gamma = model(input)
            if isp is not None:
                img_pred = tensor2numpy(isp(refine_out))
                img_gt = tensor2numpy(isp(label))
                for j in range(img_pred.shape[0]):
                    cv2.imwrite('./test/denoised_%d_frame_%d_sRGB.png'%(i, j),
                                np.uint8(img_pred[j] * 255))
            #     plt.subplot(1, 2, 1), plt.imshow(np.uint8(img_gt * 255))
            #     plt.subplot(1, 2, 2), plt.imshow(np.uint8(img_pred * 255))
            #     cv2.imwrite('scene{}_frame{}_denoised_sRGB.png'.format(i, i),
            #                 np.uint8(img_pred * 255))
            #     plt.show()
            #     print('.')
            frame_psnr = psnr(refine_out, label)
            print('PSNR: ', '%.8f' % frame_psnr.item())
            total_psnr += frame_psnr
            cnt += 1
        print('cnt',cnt)
        total_psnr = total_psnr / cnt
    print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
    torch.cuda.empty_cache()
    return total_psnr


def generate_file_list(scene_list):
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6]
    iso_list = [1600, 3200, 6400, 12800, 25600]
    file_num = 0
    data_name = []
    for scene_ind in scene_list:
        for iso in iso_list:
            for frame_ind in range(7):
                for fid in frame_list[frame_ind:cfg.batch_size + frame_ind]:
                    gt_name = os.path.join('ISO{}/scene{}_frame{}_gt_sRGB.png'.format(
                        iso, scene_ind, fid))
                    data_name.append(gt_name)
                    file_num += 1
    return data_name
    # random_index = np.random.permutation(file_num)
    # data_random_list = []
    # for i, idx in enumerate(random_index):
    #     data_random_list.append(data_name[idx])
    # return data_random_list


def save_feature(feat,name):
    import os
    import cv2
    PROJ = '/home/wen/Documents/project/video/denoising/emvd'
    sub_dir = 'test'
    sub_folder = 'feature'
    output_dir = os.path.join(PROJ, sub_dir, sub_folder)
    b,c,h,w = feat.shape
    feat = feat.data.cpu().numpy()
    for i in range(c):
        f = feat[0,i,:,:]
        v_max = np.max(f)
        v_min = np.min(f)
        tmp = (f - v_min) / (v_max - v_min)
        cv2.imwrite(output_dir + '/{}_{}.png'.format(name, str(i)), np.uint8(tmp * 255))

def check_ft_err():
    scene_id = 7
    iso = 3200
    frame_ind = 1
    raw_name = os.path.join(cfg.data_root[1],
                            'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy0.tiff'.format(
                                scene_id, scene_id, iso, frame_ind))

    raw = cv2.imread(raw_name, -1)
    input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
    input_full = input_full.transpose((0, 3, 1, 2))
    input_full = torch.from_numpy(input_full).float()
    ct = structure.ColorTransfer()
    inct = ct(input_full)
    ft = structure.FreTransfer()
    inctft = ft(inct)
    h0_row = ft.net1.weight[0:1]
    h1_row = ft.net1.weight[1:2]
    ##############################################
    h0_row_t = h0_row.transpose(2, 3)
    h1_row_t = h1_row.transpose(2, 3)
    h00_row = h0_row * h0_row_t
    h01_row = h0_row * h1_row_t
    h10_row = h1_row * h0_row_t
    h11_row = h1_row * h1_row_t
    filters1 = [h00_row, h01_row, h10_row, h11_row]
    zeros = torch.zeros_like(h00_row, device=h00_row.device)
    filters_g1 = []
    for i in range(4):
        tmp = filters1[i]
        g = torch.cat([torch.cat([tmp, zeros, zeros, zeros], dim=1),
                       torch.cat([zeros, tmp, zeros, zeros], dim=1),
                       torch.cat([zeros, zeros, tmp, zeros], dim=1),
                       torch.cat([zeros, zeros, zeros, tmp], dim=1)], dim=0)
        filters_g1.append(g)
    filters_ft = torch.cat(filters_g1, dim=0)
    ##############################################
    inctft22 = F.conv2d(inct, filters_ft, stride=(2, 2), padding=0, bias=None)
    arr1 = inctft.data.cpu().numpy()
    arr2 = inctft22.detach().numpy()
    f = arr1[0,0]
    v_max = np.max(f)
    v_min = np.min(f)
    tmp = (f - v_min) / (v_max - v_min)
    print(np.max(np.abs(arr1-arr2))) # 4.7683716e-07
    cv2.imwrite('arr1.png', np.uint8(tmp * 255))
    f = arr2[0, 0]
    v_max = np.max(f)
    v_min = np.min(f)
    tmp = (f - v_min) / (v_max - v_min)
    cv2.imwrite('arr2.png', np.uint8(tmp * 255))

    print(torch.where(inctft.data.cpu()!=inctft22))
    print('end')

import torch.distributions as tdist
def add_hg_noise(image, shot_noise=13.486051, read_noise=130.818508, noise_type = 'hg'):
    # torch 版本
    if noise_type == 'hg':
        variance = image * shot_noise / 3855 + read_noise / (3855 * 3855)
    elif noise_type == 'gs':
        variance = image * 0 + shot_noise / 3855
    elif noise_type == 'no':
        variance = image * 0
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise = n.sample()
    img_n = image + noise
    return torch.clamp(img_n, -1, 1)

def check_ct_ft():
    scene_id = 7
    iso = 3200
    frame_ind = 1
    raw_name = os.path.join(cfg.data_root[1],
                            'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy7.tiff'.format(
                                scene_id, scene_id, iso, frame_ind))
    gt_name = os.path.join(cfg.data_root[1],
                           'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
                               scene_id, scene_id, iso, frame_ind))
    # '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_raw_gt/indoor_raw_gt_scene7/scene7/ISO3200'
    raw = cv2.imread(raw_name, -1).astype(np.float32) # value in [145,4094]
    gt = cv2.imread(gt_name, -1).astype(np.float32) # value in [250,4095]
    black_level = 240
    white_level = 2 ** 12 - 1
    rgb = False
    # hg = False
    suffix = ''
    # if not rgb:
    #     raw = np.maximum(raw - black_level, 0) / (white_level - black_level) # value in [0.0,0.9997406]
    #     gt = np.maximum(gt - black_level, 0) / (white_level - black_level) # value in [0.0025940337,1.0]
    # if hg:
    #     suffix = '_hg'
    #     hg_noise = add_hg_noise(torch.from_numpy(gt).float(), shot_noise=a_list[1], read_noise=b_list[1])
    #     raw = hg_noise.data.cpu().numpy()
    # delta = raw - gt
    # if rgb:
    #     suffix += '_rgb'
    #     delta = np.expand_dims(pack_gbrg_raw(delta), axis=0)
    #     delta = delta.transpose((0, 3, 1, 2))
    #     delta = torch.from_numpy(delta).float()
    #     delta = tensor2numpy(isp(delta))[0]
    # else:
    #     suffix += '_raw'
    # # v_max = np.max(delta)
    # # v_min = np.min(delta)
    # # tmp = (delta - v_min) / (v_max - v_min)
    # cv2.imwrite('test/feature/scene{}_ISO{}_frame{}'.format(scene_id, iso, frame_ind)+'_noise%s.png'%suffix,
    #             np.uint8(delta * 255))

    pixel_shuffle = architecture.PixelShuffle(2)
    pixel_unshuffle = architecture.PixelShuffle(0.5)
    if cfg.use_pixel_shuffle:
        raw = np.expand_dims(raw, axis=0)
        raw = np.expand_dims(raw, axis=0)
        raw = torch.from_numpy(raw).float()
        for i in range(cfg.px_num):
            raw = pixel_unshuffle(raw)
        input_full = raw
    else:
        input_full = np.expand_dims(pack_gbrg_raw(raw), axis=0)
        input_full = input_full.transpose((0, 3, 1, 2))
        input_full = torch.from_numpy(input_full).float()

    ct = architecture.ColorTransfer() # structure.ColorTransfer()
    ft = architecture.FreTransfer() # structure.FreTransfer()
    cti = architecture.ColorTransferInv()
    fti = architecture.FreTransferInv()

    f_ct = ct(input_full)
    save_feature(f_ct, 'scene{}_ISO{}_frame{}'.format(scene_id, iso, frame_ind)+'_ct')
    # f_ft = ft(f_ct)
    # save_feature(f_ft, 'scene{}_ISO{}_frame{}'.format(scene_id, iso, frame_ind)+'_ft')
    # f_fti = fti(f_ft)
    # save_feature(f_fti, 'scene{}_ISO{}_frame{}'.format(scene_id, iso, frame_ind)+'_fti')
    # f_cti = cti(f_fti)
    f_cti = cti(f_ct)
    if cfg.use_pixel_shuffle:
        for i in range(cfg.px_num):
            f_cti = pixel_shuffle(f_cti)
    save_feature(f_cti, 'scene{}_ISO{}_frame{}'.format(scene_id, iso, frame_ind)+'_cti')


def check_dialateConv():
    from torch.autograd import Variable
    data = Variable(torch.randn(1, 32, 544//8, 960//8))
    refine = Variable(torch.randn(1, 16, 544//2, 960//2))
    ResBlock = arch.Realism()
    torch.save(ResBlock.state_dict(), os.path.join('./test/Realism.pth'))
    inputs = {"input": data, "refine": refine }
    torch.onnx.export(ResBlock,
                      (data, refine),
                      "./test/Realism.onnx",
                      opset_version=11,
                      do_constant_folding=False,
                      input_names=['input','refine'],
                      output_names=['output']
                      )


def read_img(img_name, xx, yy):
    raw = cv2.imread(img_name, -1)
    raw_full = raw
    raw_patch = raw_full[yy:yy + cfg.image_height * 2,
                xx:xx + cfg.image_width * 2]  # 256 * 256
    raw_pack_data = pack_gbrg_raw(raw_patch)
    return raw_pack_data


def decode_data(data_name, xx, yy):
    scene_ind = data_name.split('/')[1].split('_')[0]
    frame_ind = int(data_name.split('/')[1].split('_')[1][5:])
    iso_ind = data_name.split('/')[0]

    noisy_level_ind = iso_list.index(int(iso_ind[3:]))
    noisy_level = [a_list[noisy_level_ind], b_list[noisy_level_ind]]
    # print('Scene: ', ('%s' % scene_ind),
    #       'Noisy_level: ', ('%02d' % noisy_level_ind),
    #       'Frame_Ind: ', ('%02d' % frame_ind))
    gt_name = os.path.join(cfg.data_root[1],
                           'indoor_raw_gt/indoor_raw_gt_{}/{}/{}/frame{}_clean_and_slightly_denoised.tiff'.format(
                               scene_ind, scene_ind, iso_ind, frame_ind))

    noisy_frame_index_for_current = np.random.randint(0, 10)
    noisy_name = os.path.join(cfg.data_root[1],
                              'indoor_raw_noisy/indoor_raw_noisy_{}/{}/{}/frame{}_noisy{}.tiff'.format(
                                  scene_ind, scene_ind, iso_ind, frame_ind,
                                  noisy_frame_index_for_current))
    # gt_raw_data = read_img(gt_name, xx, yy)
    # noisy_data = read_img(noisy_name, xx, yy)
    gt_raw_data = cv2.imread(gt_name, -1)
    noisy_data = cv2.imread(noisy_name, -1)
    gt_gbrg_raw_data = pack_gbrg_raw(gt_raw_data)
    noisy_gbrg_raw_data = pack_gbrg_raw(noisy_data)
    gt_gbrg_raw_data = np.pad(gt_gbrg_raw_data, [(2, 2), (0, 0), (0, 0)], mode='constant')
    noisy_gbrg_raw_data = np.pad(noisy_gbrg_raw_data, [(2, 2), (0, 0), (0, 0)], mode='constant')
    return noisy_gbrg_raw_data, gt_gbrg_raw_data, noisy_level


class loadImgs(Dataset):
    def __init__(self, filelist, model, cfg):
        self.cfg = cfg
        self.filelist = filelist
        print(len(self.filelist))
        self.model = model.to(torch.device('cpu'))
        self.fused = None
        self.H = 1080
        self.W = 1920
        self.image_width = self.cfg.image_width
        self.image_height = self.cfg.image_height
        self.xx = 0
        self.yy = 0

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        # print('item', item)
        if item % self.cfg.batch_size == 0:
            self.xx = np.random.randint(0, (self.W - self.image_width * 2 + 1) / 2) * 2
            self.yy = np.random.randint(0, (self.H - self.image_height * 2 + 1) / 2) * 2
        self.data_name = self.filelist[item]
        image, label, noisy_level = decode_data(self.data_name, self.xx, self.yy)
        if item % self.cfg.batch_size == 0:
            print('---------')
            scene_ind = self.data_name.split('/')[1].split('_')[0]
            frame_ind = int(self.data_name.split('/')[1].split('_')[1][5:])
            iso_ind = self.data_name.split('/')[0]
            noisy_level_ind = iso_list.index(int(iso_ind[3:]))
            noisy_level = [a_list[noisy_level_ind], b_list[noisy_level_ind]]
            print('Scene: ', ('%s' % scene_ind),
                  'Noisy_level: ', ('%02d' % noisy_level_ind),
                  'Frame_Ind: ', ('%02d' % frame_ind))
            print('---------')
            self.fused = image
        tmp = np.concatenate([self.fused, image], axis=2)
        tmp = tmp.transpose(2, 0, 1)
        tmp = np.expand_dims(tmp, axis=0)
        tmp = torch.from_numpy(tmp)
        # input = tmp.cuda()
        fusion_out, denoise_out, refine_out, omega, gamma = self.model(tmp)
        self.fused = fusion_out.numpy()[0] # .data.cpu().numpy()
        self.fused = self.fused.transpose(1, 2, 0)
        self.image = np.concatenate([self.fused, image], axis=2)
        self.label = label
        self.noisy_level = noisy_level
        self.image = self.image.transpose(2, 0, 1)
        self.label = self.label.transpose(2, 0, 1)
        return self.image, self.label  # , self.noisy_level


def load_weight():
    checkpoint = torch.load('./%s/model.pth' % (cfg.model_name))
    state_dict = checkpoint['model']
    if cfg.ngpu > 1:
        temp = collections.OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = '.'.join(k.split('.')[1:])
                temp[name] = v
        if len(temp):
            state_dict = temp
    return state_dict

def eval_video():
    # Load the pretrained model
    model = arch_qat.EMVD(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    state_dict = torch.load('./%s/model_qua_arch.pth' % (cfg.model_name))
    model.load_state_dict(state_dict, strict=True)

    from dill import dumps, loads
    model_deepcopy = loads(dumps(model))
    model_deepcopy = model_deepcopy.eval()
    model = model.eval()

    eval_data_name_queue = generate_file_list(['7', '8', '9', '10', '11'])
    eval_dataset = loadImgs(eval_data_name_queue, model_deepcopy, cfg)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=cfg.batch_size,
                                              num_workers=1,  # cfg.num_workers,
                                              shuffle=False,
                                              pin_memory=True)
    # Calculate FP32 accuracy
    eval_psnr = evaluate(model, eval_loader, isp)
    print(eval_psnr)

def load_model():
    # Load the pretrained model
    model = arch_qat.EMVD(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint = torch.load(cfg.best_model_save_root)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=False)
    # ft
    h0_row = model.ft.w1
    h1_row = model.ft.w2
    h0_row_t = model.ft.w1.transpose(2, 3)
    h1_row_t = model.ft.w2.transpose(2, 3)
    h00_row = h0_row * h0_row_t
    h01_row = h0_row * h1_row_t
    h10_row = h1_row * h0_row_t
    h11_row = h1_row * h1_row_t
    filters1 = [h00_row, h01_row, h10_row, h11_row]
    zeros = torch.zeros_like(h00_row, device=h00_row.device)
    filters_g1 = []
    for i in range(4):
        tmp = filters1[i]
        g = torch.cat([torch.cat([tmp, zeros, zeros, zeros], dim=1),
                       torch.cat([zeros, tmp, zeros, zeros], dim=1),
                       torch.cat([zeros, zeros, tmp, zeros], dim=1),
                       torch.cat([zeros, zeros, zeros, tmp], dim=1)], dim=0)
        filters_g1.append(g)
    filters_ft = torch.cat(filters_g1, dim=0)#.to('cuda')
    model.ft.net.weight = nn.Parameter(filters_ft)
    # fti
    g0_col = model.fti.w1
    g1_col = model.fti.w2
    g0_col_t = model.fti.w1.transpose(2, 3)
    g1_col_t = model.fti.w2.transpose(2, 3)
    g00_col = g0_col * g0_col_t
    g01_col = g0_col * g1_col_t
    g10_col = g1_col * g0_col_t
    g11_col = g1_col * g1_col_t
    filters2 = [g00_col, g01_col, g10_col, g11_col]
    zeros = torch.zeros_like(g00_col, device=g00_col.device)
    filters_g2 = []
    for i in range(4):
        tmp = filters2[i]
        g = torch.cat([torch.cat([tmp, zeros, zeros, zeros], dim=1),
                       torch.cat([zeros, tmp, zeros, zeros], dim=1),
                       torch.cat([zeros, zeros, tmp, zeros], dim=1),
                       torch.cat([zeros, zeros, zeros, tmp], dim=1)], dim=0)
        filters_g2.append(g)
    filters_fti = torch.cat(filters_g2, dim=0)#.to('cuda')
    model.fti.net.weight = nn.Parameter(filters_fti)
    #########################################################################
    model.ct0.net1.weight = model.ct.net1.weight
    model.ct1.net1.weight = model.ct.net1.weight

    model.cti_fu.net1.weight = model.cti.net1.weight
    model.cti_de.net1.weight = model.cti.net1.weight
    model.cti_re.net1.weight = model.cti.net1.weight

    model.ft_00.net.weight = model.ft.net.weight
    model.ft_10.net.weight = model.ft.net.weight
    model.ft_01.net.weight = model.ft.net.weight
    model.ft_11.net.weight = model.ft.net.weight
    model.ft_02.net.weight = model.ft.net.weight
    model.ft_12.net.weight = model.ft.net.weight

    model.fti_d2.net.weight = model.fti.net.weight
    model.fti_d1.net.weight = model.fti.net.weight
    model.fti_fu.net.weight = model.fti.net.weight
    model.fti_de.net.weight = model.fti.net.weight
    model.fti_re.net.weight = model.fti.net.weight
    #########################################################################
    model_name = type(model).__name__
    module_to_name = {}
    for name, module in model.named_modules(prefix=model_name):
        module_to_name[module] = name

    model = model.eval()

def test_param_name():
    model = arch.EMVD(cfg)
    for name, child in model.named_children():
        if 'realism' in name:
            print(name)
        for param in child.parameters():
            if param.name:
                print(param)
    for param in model.parameters():
        print(param)

from ..arch import architecture
def test_pixel_shuffle():
    pixel_shuffle = nn.PixelShuffle(2)
    # pixel_unshuffle = nn.PixelUnshuffle(2)
    px = architecture.PixelShuffle(2)
    pux = architecture.PixelShuffle(0.5)
    arr = range(1,32+1)
    arr = np.array(arr,dtype=np.float32)
    arr = np.reshape(arr, (1,2,4,4))
    map = torch.from_numpy(arr)
    map_pux = pux(map)
    map_px = px(map_pux)
    map_px_ = pixel_shuffle(map_pux)
    print(map_px_==map_px)

pixel_shuffle = architecture.PixelShuffle(2)
pixel_unshuffle = architecture.PixelShuffle(0.5)

def test():
    arw = '/home/wen/Downloads/package/0001.ARW'
    raw = rawpy.imread(arw)
    print(raw)
    # subfolder = '/home/wen/Documents/project/video/denoising/emvd_bin/data/device/fake_ab'
    # for idx in range(5):
    #     iso = iso_list[idx]
    #     coeff_a = torch.tensor(a_list[idx] / (2 ** 12 - 1 - 240)).float().to(cfg.device)
    #     coeff_a*=100
    #     print(coeff_a)
    #     coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1])
    #     coeff_b = torch.tensor(b_list[idx] / (2 ** 12 - 1 - 240) ** 2).float().to(cfg.device)
    #     coeff_b*=100000
    #     print(coeff_b)
    #     coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1])
    #     coeff_b.data.cpu().numpy().tofile('%s/coeff_b_%d.raw' % (subfolder,iso))
    #     coeff_a.data.cpu().numpy().tofile('%s/coeff_a_%d.raw' % (subfolder,iso))

    isp = torch.load('isp/ISP_CNN.pth')
    path = '/home/wen/Documents/dataset/Mi11Ultra/20211213194854_0/VideoNight_Time_1639396134466.000000_FrameID_00000001_width_4080_height_3072_Input.raw'
    input_data = np.fromfile(path, dtype=np.int16)
    input_data = np.reshape(input_data, (1, 1, 3072, 4080))
    black_level = 64
    white_level = 2 ** 10 - 1
    im = input_data.astype(np.float32)
    im = np.maximum(im - black_level, 0) / (white_level - black_level)  # 0 - 1
    raw = torch.from_numpy(im)
    data = pixel_unshuffle(raw)

    def gbrg2rgbg(data):
        data = torch.cat([data[:, 2:3, :, :],
                          data[:, 3:4, :, :],
                          data[:, 1:2, :, :],
                          data[:, 0:1, :, :]], dim=1)
        return data
    isp = isp.to(cfg.device)
    data_rgb = np.zeros((768,1020,3))
    for i in range(4):
        tmp = pixel_unshuffle(data[:,i:(i+1),:,:])
        rgb = torch.cat([tmp[:, 2:3, :, :],
                              (tmp[:, 3:4, :, :] + tmp[:, 0:1, :, :]) / 2,
                              tmp[:, 1:2, :, :]], dim=1)
        data_rgb += tensor2numpy(rgb)[0]
    cv2.imwrite('frame%d.png' % (i), np.uint8(data_rgb * 255))
        # tmp = gbrg2rgbg(tmp).float().to(cfg.device) # tmp.float().to(cfg.device) #
        # frame = tensor2numpy(isp(tmp))[0]
        # cv2.imwrite('frame%d.png'%(i), np.uint8(frame * 255))

    one = np.ones([1,1,4,4])
    gbrg = np.concatenate([one,one,one,one],axis=1)
    patch = np.concatenate([gbrg,gbrg*2,gbrg*3,gbrg*4],axis=1)
    patch4 = np.concatenate([patch,patch*10,patch*100,patch*1000],axis=1)
    bin = architecture.bin()
    patch4 = torch.tensor(patch4).float()
    out = bin(patch4)
    print(out)
    # cfa = np.array(
    #     [[0.5, 0.5, 0.5, 0.5],
    #      [-0.5, 0.5, -0.5, 0.5],
    #      [-0.65, -0.2784, 0.65, 0.2784],
    #      [0.2764, -0.65, -0.2784, 0.65]])
    # cfa = np.array(
    #     [[0.5, 0.5, 0.5, 0.5],
    #      [-0.5, 0.5, 0.5, -0.5],
    #      [0.65, 0.2784, -0.2784, -0.65],
    #      [-0.2784, 0.65, -0.65, 0.2764]])
    # cfa = np.expand_dims(cfa, axis=2)
    # cfa = np.expand_dims(cfa, axis=3)
    # cfa = torch.tensor(cfa).float()  # .cuda()
    # cfa_inv = cfa.transpose(0, 1).squeeze()
    # cfa = cfa.squeeze()
    # weight_squared = torch.matmul(cfa, cfa_inv)
    # print(weight_squared)
    ct = architecture.ColorTransfer()
    w = ct.w
    n = 4 ** cfg.px_num
    cfan = torch.zeros((n, n, 1, 1), device=w.device)
    c = 4  # n // 4
    for i in range(4):
        for k in range(4):
            for j in range(c):
                cfan[i * 16 + k * 4 + j, k * 16 + j, :, :] = w[i, 0]
                cfan[i * 16 + k * 4 + j, k * 16 + j + c, :, :] = w[i, 1]
                cfan[i * 16 + k * 4 + j, k * 16 + j + c * 2, :, :] = w[i, 2]
                cfan[i * 16 + k * 4 + j, k * 16 + j + c * 3, :, :] = w[i, 3]
    cti = architecture.ColorTransferInv()
    w = cti.w
    n = 4 ** cfg.px_num
    cfan_inv = torch.zeros((n, n, 1, 1), device=w.device)
    c = 4  # n // 4
    for k in range(4):
        for i in range(4):
            for j in range(c):
                cfan_inv[k * 16 + i * 4 + j, k * 4 + j, :, :] = w[i, 0]
                cfan_inv[k * 16 + i * 4 + j, k * 4 + j + 16, :, :] = w[i, 1]
                cfan_inv[k * 16 + i * 4 + j, k * 4 + j + 16 * 2, :, :] = w[i, 2]
                cfan_inv[k * 16 + i * 4 + j, k * 4 + j + 16 * 3, :, :] = w[i, 3]
    weight_squared = torch.matmul(cfan.squeeze(), cfan_inv.squeeze())
    print(torch.max(weight_squared))
    print(torch.min(weight_squared))
    print(torch.where(weight_squared!=(weight_squared.transpose(0, 1))))
    # n = 4 ** cfg.px_num
    # c = n // 4
    # for i in range(4):
    #     for j in range(c):
    #         print(i * c + j, j)
    #         print(i * c + j, j + c)
    #         print(i * c + j, j + c * 2)
    #         print(i * c + j, j + c * 3)
    # c = 4
    # for i in range(4):
    #     for k in range(4):
    #         for j in range(c):
    #             print(i * 16 + k * 4 + j, k * 16 + j)
    #             print(i * 16 + k * 4 + j, k * 16 + j + c)
    #             print(i * 16 + k * 4 + j, k * 16 + j + c * 2)
    #             print(i * 16 + k * 4 + j, k * 16 + j + c * 3)
    # for k in range(4):
    #     for i in range(4):
    #         for j in range(c):
    #             print(k * 16 + i * 4 + j, k * 4 + j)
    #             print(k * 16 + i * 4 + j, k * 4 + j + 16)
    #             print(k * 16 + i * 4 + j, k * 4 + j + 16 * 2)
    #             print(k * 16 + i * 4 + j, k * 4 + j + 16 * 3)
    c = 64
    n = 4
    one = np.ones((1,n,4,4))
    map = one
    for i in range(2,c//n+1):
        map = np.concatenate([map, one * i], axis=1)
    x = torch.from_numpy(map).float()
    group4 = np.zeros((c // n, c, 1, 1), dtype=np.float32)
    tmp = np.ones((c // n, n, 1, 1), dtype=np.float32)
    for i in range(c // n):
        group4[i, i * n:(i + 1) * n, :, :] = tmp[i, :, :, :]

    y = F.conv2d(x, torch.from_numpy(group4).float(), stride=1, padding=0, bias=None)

    tile4to64 = np.zeros((64, 4, 1, 1), dtype=np.float32)
    diag = np.eye(4, dtype=np.float32)
    diag = np.reshape(diag, [4, 4, 1, 1])
    for i in range(16):
        tile4to64[i*4:(i + 1) * 4, :, :, :] = diag
    tile4to64 = torch.tensor(tile4to64).float()
    conv = nn.Conv2d(4, 64, kernel_size=1, stride=1, padding=0, bias=False)
    conv.weight = torch.nn.Parameter(tile4to64)
    conv.weight.requires_grad = False
    map = np.array(range(16),dtype=np.float32)
    map = np.reshape(map,[1,4,2,2])
    map = torch.tensor(map)
    out = conv(map)

    arr = np.array([[0.5, 0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5, -0.5],
                    [0.65, 0.2784, -0.2784, -0.65],
                    [-0.2784, 0.65, -0.65, 0.2764]])
    cfa16 = np.zeros((16, 16, 1, 1), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            cfa16[i * 4 + j, j, :, :] = arr[i, 0]
            cfa16[i * 4 + j, j + 4, :, :] = arr[i, 1]
            cfa16[i * 4 + j, j + 8, :, :] = arr[i, 2]
            cfa16[i * 4 + j, j + 12, :, :] = arr[i, 3]
    print(cfa16)

def get_list():
    path = '/home/wen/Documents/dataset/Mi11Ultra/20211230/image/Camera/*.jpg'
    imgs = glob.glob(path)
    f = open('/home/wen/Documents/dataset/Mi11Ultra/20211230/k1_211230_jpg.txt','w')
    imgs.sort()
    for img in imgs:
        name = os.path.basename(img).split('.')[0]
        f.write(name+'\n')
    f.close()

def snpe_err_debug_inputs():
    path = '/home/wen/Documents/project/video/denoising/emvd_bin/log/model/'
    name_ft1 = 'INPUT_ft1_00000001.raw'
    name_ft0 = 'INPUT_ft0_00000001.raw'
    name_coeff_a = 'INPUT_coeff_a_00000001.raw'
    name_coeff_b = 'INPUT_coeff_b_00000001.raw'
    ft0 = np.fromfile(path+name_ft0, dtype=np.float32)
    ft0 = np.reshape(ft0, (1, 16, 272, 480))
    ft1 = np.fromfile(path + name_ft1, dtype=np.float32)
    ft1 = np.reshape(ft1, (1, 16, 272, 480))
    a = np.fromfile(path + name_coeff_a, dtype=np.float32)
    a = np.reshape(a, (1, 1, 1, 1))
    b = np.fromfile(path + name_coeff_b, dtype=np.float32)
    b = np.reshape(b, (1, 1, 1, 1))
    print(np.max(ft0),np.min(ft0))
    print(np.max(ft1),np.min(ft1))
    print(a)
    print(b)
    print('end')


import torch.onnx
class testModel(nn.Module):
    def __init__(self):
        super(testModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act2 = nn.ReLU()

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.act1(o1)
        o3 = self.conv2(o2)
        out = self.act2(o3)
        return out

from torch.autograd import Variable
def test_onnx():
    m = testModel()
    f = open('./test/raw_list.txt','w')
    proj = '/home/wen/Documents/project/video/denoising/emvd_bin/test/'
    input = Variable(torch.randn(1, 3, 128, 128))
    input = input.cuda()
    for i in range(100):
        input = Variable(torch.randn(1, 3, 128, 128))
        input = input.cuda()
        raw = input.permute(0, 2, 3, 1).data.cpu().numpy().astype('float32')
        raw.tofile('./test/test%d.raw'%i)
        f.write(proj+'test%d.raw'%i+'\n')
    f.close()
    if torch.cuda.is_available():
        m.cuda()
    torch.onnx.export(m,
                      (input),
                      "./test/test.onnx",
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
    print('end')


def main():
    # test_onnx()
    # snpe_err_debug_inputs()
    # get_list()
    # check_ft_err()
    # test_pixel_shuffle()
    test()
    # check_dialateConv()
    # check_ct_ft()
    # eval_video()
    # load_model()



if __name__ == '__main__':
    main()

'''
tensor(0.0009, device='cuda:0')
tensor(8.0194e-07, device='cuda:0')
tensor(0.0018, device='cuda:0')
tensor(2.5650e-06, device='cuda:0')
tensor(0.0035, device='cuda:0')
tensor(8.8028e-06, device='cuda:0')
tensor(0.0069, device='cuda:0')
tensor(3.2605e-05, device='cuda:0')
tensor(0.0135, device='cuda:0')
tensor(0.0001, device='cuda:0')
'''