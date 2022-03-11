import os
from datetime import datetime
import torch
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
from utils import netloss
from utils.polynomialCurveFitting import *
from arch import architecture
from arch import architecture_reparam # architecture_qat
from arch.modules import PixelShuffle
from dataset.dataset import preprocess, tensor2numpy, pack_gbrg_raw, norm_raw
from generate import unprocess
from generate.unpack_rrggbbrr import pack_1raw_to_4raw
from generate.tools import pixel_unShuffle_RGBG, setup_seed, pixel_unShuffle, binning_raw
from generate.visualization import gbrg_to_rgb_dispaly
from utils.models import models



def eval_DRV():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.model_name = 'model_fuLoss_deLoss' # 'model_deLoss' # 'model' #
    log_dir = 'log/log/{}/'.format(cfg.model_name)
    iso_list = cfg.iso_list
    iso_list.sort(reverse=True)
    a_list = cfg.a_list
    a_list.sort(reverse=True)
    b_list = cfg.b_list
    b_list.sort(reverse=True)
    scene_list = cfg.val_list
    scene_list.sort(reverse=True)

    frame_list = cfg.frame_list_eval

    model_path = os.path.join(log_dir, 'model_best.pth') # 'model.pth') #
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

    pixel_shuffle = PixelShuffle(2)
    pixel_unshuffle = PixelShuffle(0.5)

    psnr = netloss.PSNR().to(cfg.device)

    avg_pool = torch.nn.AvgPool2d(2, 2)

    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    log_path = log_dir + '/results/' + '{}'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f = open(log_path + '/{}_test_psnr_and_ssim.txt'.format(os.path.basename(model_path)), 'w')
    with torch.no_grad():
        for scene_ind in scene_list:
            if scene_ind in cfg.obj_motion:
                video_type = 'obj_motion'
            else:
                video_type = 'camera_motion'
            scene_ind = '{:0>4d}'.format(scene_ind)
            iso_average_raw_psnr = 0
            iso_average_raw_ssim = 0
            for noisy_level_ind, iso_ind in enumerate(iso_list):
                output_dir = log_path + '/ISO{}_scene{}/'.format(iso_ind, scene_ind)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(output_dir+'gt/'):
                    os.makedirs(output_dir+'gt/')
                if not os.path.exists(output_dir+'noisy/'):
                    os.makedirs(output_dir+'noisy/')
                if not os.path.exists(output_dir+'fusion/'):
                    os.makedirs(output_dir+'fusion/')
                if not os.path.exists(output_dir+'denoise/'):
                    os.makedirs(output_dir+'denoise/')
                if not os.path.exists(output_dir+'refine/'):
                    os.makedirs(output_dir+'refine/')
                if not os.path.exists(output_dir+'omega/'):
                    os.makedirs(output_dir+'omega/')
                if not os.path.exists(output_dir+'gamma/'):
                    os.makedirs(output_dir+'gamma/')
                frame_avg_raw_psnr = 0
                frame_avg_raw_ssim = 0
                for idx in range(1, len(frame_list)+1):
                    name = 'ISO{}_scene{}_frame{}'.format(iso_ind, scene_ind, str(idx))
                    frame_ind = frame_list[idx-1]
                    ########################################################
                    # gt
                    ########################################################
                    gt_name = os.path.join(cfg.data_root, '{}/{}/raw_bin/frame_{}.npy'
                                           .format(scene_ind, video_type, frame_ind))
                    raw_gt = np.load(gt_name, mmap_mode='r')
                    raw_gt_norm = norm_raw(raw_gt,
                                           black_level=cfg.black_level,
                                           white_level=cfg.white_level)
                    fgt = pixel_unshuffle(torch.from_numpy(np.expand_dims(np.expand_dims(raw_gt_norm,axis=0),axis=0)))  # (b,4,h/2,w/2)
                    raw_gt_norm_pxu = pixel_unShuffle_RGBG(raw_gt_norm.astype(np.float32), cfg.bayer_pattern)
                    tmp = np.pad(raw_gt_norm_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
                    raw_gt_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(output_dir + 'gt/' + '{}.png'.format(name),
                                np.uint8(raw_gt_isp * 255))
                    ########################################################
                    # noisy
                    ########################################################
                    noisy_frame_index_for_current = np.random.randint(1, cfg.noisy_num + 1)
                    input_name = os.path.join(cfg.data_root,
                                              '{}/{}/noisy/iso{}/frame_{}_iso{}_noisy{}.npy'.format(
                                                  scene_ind, video_type, iso_ind,
                                                  frame_ind,  iso_ind, noisy_frame_index_for_current))
                    raw_noisy = np.load(input_name, mmap_mode='r')
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
                    tmp = np.pad(raw_noisy_norm, [(20, 20), (4, 4)])
                    noisy = torch.from_numpy(np.expand_dims(np.expand_dims(tmp,
                                            axis=0), axis=0)).cuda()
                    ft1 = noisy
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)

                    if idx==1:
                        ft0 = ft1

                    coeff_a = a_list[noisy_level_ind] / (cfg.white_level - cfg.black_level)
                    coeff_b = b_list[noisy_level_ind] / (cfg.white_level - cfg.black_level) ** 2

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

                    fusion = fusion[0, 0, 10:-10, 2:-2]
                    denoise = denoise[0, 0, 10:-10, 2:-2]
                    refine = refine[0, 0, 10:-10, 2:-2]
                    omega = omega[0, 0]
                    gamma = gamma[0, 0]

                    fusion_np = fusion.data.cpu().numpy()       # h,w
                    denoise_np = denoise.data.cpu().numpy()
                    refine_np = refine.data.cpu().numpy()

                    fusion_np_pxu = pixel_unShuffle_RGBG(fusion_np, cfg.bayer_pattern)
                    denoise_np_pxu = pixel_unShuffle_RGBG(denoise_np, cfg.bayer_pattern)
                    refine_np_pxu = pixel_unShuffle_RGBG(refine_np, cfg.bayer_pattern)

                    tmp = np.pad(fusion_np_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
                    fusion_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(output_dir + 'fusion/{}.png'.format(name), np.uint8(fusion_isp * 255))
                    tmp = np.pad(denoise_np_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
                    denoise_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(output_dir + 'denoise/{}.png'.format(name), np.uint8(denoise_isp * 255))
                    tmp = np.pad(refine_np_pxu, [(0, 0), (0, 0), (1, 1), (1, 1)])
                    refine_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(output_dir + 'refine/{}.png'.format(name), np.uint8(refine_isp * 255))

                    cv2.imwrite(output_dir + 'omega/%s.png' % name, np.uint8(omega.data.cpu().numpy() * 255))
                    cv2.imwrite(output_dir + 'gamma/%s.png' % name, np.uint8(gamma.data.cpu().numpy() * 255))

                    fnoisy = pixel_unshuffle(torch.from_numpy(np.expand_dims(np.expand_dims(raw_noisy_norm,axis=0),axis=0)))  # (b,4,h/2,w/2)
                    fnoisy = avg_pool(fnoisy)

                    fout = pixel_unshuffle(torch.from_numpy(np.expand_dims(np.expand_dims(refine_np,axis=0),axis=0)))  # (b,4,h/2,w/2)

                    print(psnr(fgt, fout).item(),
                          psnr(fgt, fnoisy).item())

                    xx = 200
                    yy = 200
                    print(psnr(fgt[:,:,yy:yy + cfg.image_height, xx:xx + cfg.image_width],
                               fout[:,:,yy:yy + cfg.image_height, xx:xx + cfg.image_width]).item(),
                          psnr(fgt[:,:,yy:yy + cfg.image_height, xx:xx + cfg.image_width],
                               fnoisy[:,:,yy:yy + cfg.image_height, xx:xx + cfg.image_width]).item())
                    test_raw_psnr = compare_psnr(fgt.data.cpu().numpy(),
                                                 fout.data.cpu().numpy(),
                                                 data_range=1.0)
                    test_raw_ssim = 0
                    for i in range(4):
                        test_raw_ssim += compare_ssim(fgt.data.cpu().numpy()[0, i, :, :],
                                                      fout.data.cpu().numpy()[0, i, :, :],
                                                      data_range=1.0)
                    test_raw_ssim /=4
                    context = 'scene {} iso {} frame{} ' \
                              'test raw psnr : {}, ' \
                              'test raw ssim : {} '.format(scene_ind,
                                                           iso_ind,
                                                           str(idx),
                                                           test_raw_psnr,
                                                           test_raw_ssim) + '\n'
                    f.write(context)
                    print(context)
                    frame_avg_raw_psnr += test_raw_psnr
                    frame_avg_raw_ssim += test_raw_ssim
                frame_avg_raw_psnr = frame_avg_raw_psnr / len(frame_list)
                frame_avg_raw_ssim = frame_avg_raw_ssim / len(frame_list)
                context = 'frame average raw psnr:{}, ' \
                          'frame average raw ssim:{}'.format(frame_avg_raw_psnr,
                                                             frame_avg_raw_ssim) + '\n'
                f.write(context)
                print(context)
                iso_average_raw_psnr += frame_avg_raw_psnr
                iso_average_raw_ssim += frame_avg_raw_ssim
            iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
            iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)
            context = 'iso average raw psnr:{}, ' \
                      'iso average raw ssim:{}'.format(iso_average_raw_psnr,
                                                       iso_average_raw_ssim) + '\n'
            f.write(context)
            print(context)
            scene_avg_raw_psnr += iso_average_raw_psnr
            scene_avg_raw_ssim += iso_average_raw_ssim
        scene_avg_raw_psnr = scene_avg_raw_psnr / len(scene_list)
        scene_avg_raw_ssim = scene_avg_raw_ssim / len(scene_list)
        context = 'scene average raw psnr:{},' \
                  'scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
                                                           scene_avg_raw_ssim) + '\n'
        f.write(context)
        print(context)
        f.close()

def eval_Mi_k1():
    data_dir = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/tangyouyun/' \
               'mi_k1_pair_data/'
    device = cfg.device
    cfg.model_name = 'model_fuLoss_deLoss' # 'model_deLoss' # 'model' #
    log_dir = cfg.log_dir + 'log/{}/'.format(cfg.model_name)
    scene_list = [1, 2, 3]
    a_list = [0.109696, 0.21690887, 0.43752982, 0.89052248, 1.88346616]
    b_list = [0.30756074, 0.50924099, 1.1088187, 3.12988574, 8.53779323]
    iso_list = [200, 400, 800, 1600, 3200]
    iso_list.sort(reverse=True)
    a_list.sort(reverse=True)
    b_list.sort(reverse=True)
    # scene_list.sort(reverse=True)
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4]

    model_path = os.path.join(log_dir, 'model_best.pth') # 'model.pth') #
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

    pixel_shuffle = PixelShuffle(2)
    pixel_unshuffle = PixelShuffle(0.5)

    psnr = netloss.PSNR().to(cfg.device)

    avg_pool = torch.nn.AvgPool2d(2, 2)

    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    log_path = log_dir + '/results_K1/' + '{}'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f = open(log_path + '/{}_test_psnr_and_ssim.txt'.format(os.path.basename(model_path)), 'w')
    with torch.no_grad():
        for scene_ind in scene_list:
            iso_average_raw_psnr = 0
            iso_average_raw_ssim = 0
            for noisy_level_ind, iso_ind in enumerate(iso_list):
                output_dir = log_path + '/scene{}/ISO{}/'.format(scene_ind, iso_ind)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(output_dir+'gt/'):
                    os.makedirs(output_dir+'gt/')
                if not os.path.exists(output_dir+'noisy/'):
                    os.makedirs(output_dir+'noisy/')
                if not os.path.exists(output_dir+'fusion/'):
                    os.makedirs(output_dir+'fusion/')
                if not os.path.exists(output_dir+'denoise/'):
                    os.makedirs(output_dir+'denoise/')
                if not os.path.exists(output_dir+'refine/'):
                    os.makedirs(output_dir+'refine/')
                if not os.path.exists(output_dir+'omega/'):
                    os.makedirs(output_dir+'omega/')
                if not os.path.exists(output_dir+'gamma/'):
                    os.makedirs(output_dir+'gamma/')
                frame_avg_raw_psnr = 0
                frame_avg_raw_ssim = 0
                for idx in range(1, len(frame_list) + 1):
                    ########################################################
                    # gt
                    ########################################################
                    name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_gt{}.npy'.format(scene_ind, iso_ind,
                                                                                 iso_ind, scene_ind,
                                                                                 frame_list[idx - 1], 400)
                    gt_name = os.path.join(data_dir, name)
                    raw_gt = np.load(gt_name, mmap_mode='r')
                    raw_gt_bin = binning_raw(raw_gt)
                    fgt = pixel_unshuffle(torch.from_numpy(np.expand_dims(np.expand_dims(raw_gt_bin,axis=0),axis=0)))  # (b,4,h/2,w/2)
                    raw_gt_bin_path = os.path.join(output_dir,'gt/frame_{}.npy'.format(idx))
                    # np.save(raw_gt_bin_path, raw_gt_bin)
                    gbrg_to_rgb_dispaly(raw_gt_bin_path, raw_gt_bin)
                    ########################################################
                    # noisy
                    ########################################################
                    noisy_frame_index_for_current = np.random.randint(0, 10)
                    name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_noisy{}.npy'.format(scene_ind, iso_ind,
                                                                                    iso_ind, scene_ind,
                                                                                    frame_list[idx - 1],
                                                                                    noisy_frame_index_for_current)
                    input_name = os.path.join(data_dir, name)
                    raw_noisy = np.load(input_name, mmap_mode='r')
                    raw_noisy_path = os.path.join(output_dir, 'noisy/frame_{}.npy'.format(idx))
                    # np.save(raw_noisy_path, raw_noisy)
                    gbrg_to_rgb_dispaly(raw_noisy_path, raw_noisy)
                    ########################################################
                    # run model
                    ########################################################
                    tmp = np.pad(raw_noisy, [(0, 0), (8, 8)])
                    noisy = torch.from_numpy(np.expand_dims(np.expand_dims(tmp,
                                            axis=0), axis=0)).cuda()
                    ft1 = noisy
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)

                    if idx==1:
                        ft0 = ft1

                    coeff_a = a_list[noisy_level_ind] # / (cfg.white_level - cfg.black_level)
                    coeff_b = b_list[noisy_level_ind] # / (cfg.white_level - cfg.black_level) ** 2

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

                    fusion = fusion[0, 0, :, 4:-4]
                    denoise = denoise[0, 0, :, 4:-4]
                    refine = refine[0, 0, :, 4:-4]
                    omega = omega[0, 0]
                    gamma = gamma[0, 0]

                    fusion_np = fusion.data.cpu().numpy()       # h,w
                    denoise_np = denoise.data.cpu().numpy()
                    refine_np = refine.data.cpu().numpy()

                    raw_path = os.path.join(output_dir, 'fusion/frame_{}.npy'.format(idx))
                    # np.save(raw_path, fusion_np)
                    gbrg_to_rgb_dispaly(raw_path, fusion_np)

                    raw_path = os.path.join(output_dir, 'denoise/frame_{}.npy'.format(idx))
                    # np.save(raw_path, denoise_np)
                    gbrg_to_rgb_dispaly(raw_path, denoise_np)

                    raw_path = os.path.join(output_dir, 'refine/frame_{}.npy'.format(idx))
                    # np.save(raw_path, refine_np)
                    gbrg_to_rgb_dispaly(raw_path, refine_np)

                    cv2.imwrite(output_dir + 'omega/frame_%s.png' % idx, np.uint8(omega.data.cpu().numpy() * 255))
                    cv2.imwrite(output_dir + 'gamma/frame_%s.png' % idx, np.uint8(gamma.data.cpu().numpy() * 255))

                    fout = pixel_unshuffle(torch.from_numpy(np.expand_dims(np.expand_dims(refine_np,axis=0),axis=0)))  # (b,4,h/2,w/2)

                    # print(psnr(fgt, fout).item())
                    test_noisy_psnr = compare_psnr(raw_gt,
                                                 raw_noisy,
                                                 data_range=1.0)

                    test_raw_psnr = compare_psnr(fgt.data.cpu().numpy(),
                                                 fout.data.cpu().numpy(),
                                                 data_range=1.0)
                    test_raw_ssim = 0
                    for i in range(4):
                        test_raw_ssim += compare_ssim(fgt.data.cpu().numpy()[0, i, :, :],
                                                      fout.data.cpu().numpy()[0, i, :, :],
                                                      data_range=1.0)
                    test_raw_ssim /=4
                    context = 'scene {} iso {} frame{} ' \
                              'test noisy psnr : {}, ' \
                              'test raw psnr : {}, ' \
                              'test raw ssim : {} '.format(scene_ind,
                                                           iso_ind,
                                                           str(idx),
                                                           test_noisy_psnr,
                                                           test_raw_psnr,
                                                           test_raw_ssim) + '\n'
                    f.write(context)
                    print(context)
                    frame_avg_raw_psnr += test_raw_psnr
                    frame_avg_raw_ssim += test_raw_ssim
                frame_avg_raw_psnr = frame_avg_raw_psnr / len(frame_list)
                frame_avg_raw_ssim = frame_avg_raw_ssim / len(frame_list)
                context = 'frame average raw psnr:{}, ' \
                          'frame average raw ssim:{}'.format(frame_avg_raw_psnr,
                                                             frame_avg_raw_ssim) + '\n'
                f.write(context)
                print(context)
                iso_average_raw_psnr += frame_avg_raw_psnr
                iso_average_raw_ssim += frame_avg_raw_ssim
            iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
            iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)
            context = 'iso average raw psnr:{}, ' \
                      'iso average raw ssim:{}'.format(iso_average_raw_psnr,
                                                       iso_average_raw_ssim) + '\n'
            f.write(context)
            print(context)
            scene_avg_raw_psnr += iso_average_raw_psnr
            scene_avg_raw_ssim += iso_average_raw_ssim
        scene_avg_raw_psnr = scene_avg_raw_psnr / len(scene_list)
        scene_avg_raw_ssim = scene_avg_raw_ssim / len(scene_list)
        context = 'scene average raw psnr:{},' \
                  'scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
                                                           scene_avg_raw_ssim) + '\n'
        f.write(context)
        print(context)
        f.close()

'''
scene1室内配对数据已上传至服务器A100（10.53.66.11）： /home/work/ssd1/dataset/k1/mi_k1_pair_data
'''

def eval_Mi_k1_dlc():
    log_dir = cfg.log_dir + 'log/{}/'.format(cfg.model_name)
    dlc_path = os.path.join(log_dir, 'model_best_reparam.dlc')
    # out_names=['fusion', 'denoise', 'refine', 'omega', 'gamma']
    out_nodes = '#Conv_218 Conv_220 Conv_222 Sigmoid_210 Sigmoid_182\n'
    raw_shape = (1, 384, 512, 16)
    ########################################################
    data_dir = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/tangyouyun/' \
               'mi_k1_pair_data/'
    scene_list = [1, 2, 3]
    a_list = [0.109696, 0.21690887, 0.43752982, 0.89052248, 1.88346616]
    b_list = [0.30756074, 0.50924099, 1.1088187, 3.12988574, 8.53779323]
    iso_list = [200, 400, 800, 1600, 3200]
    iso_list.sort(reverse=True)
    a_list.sort(reverse=True)
    b_list.sort(reverse=True)
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4]
    ########################################################
    bin_np = np.load('{}/binning.npy'.format(log_dir)).astype(np.float32)
    bin_w = torch.from_numpy(bin_np)
    binnings = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
    binnings.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    conv_bin = binnings.to(cfg.device)
    ########################################################
    pixel_shuffle = PixelShuffle(2)
    pixel_unshuffle = PixelShuffle(0.5)

    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    log_path = log_dir + '/results_dlc/' + '{}'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    ########################################################
    output_dir_input = os.path.join(data_dir, 'device_input_after_binning_bhwc')
    if not os.path.exists(output_dir_input):
        os.makedirs(output_dir_input)
    output_dir_coeff = os.path.join(data_dir, 'device_coeff')
    if not os.path.exists(output_dir_coeff):
        os.makedirs(output_dir_coeff)
    ########################################################
    fdevice = open(os.path.join(data_dir, 'raw_list.txt'), 'w')
    fdevice.write(out_nodes)
    device_data_dir = '/data/local/tmp/wen/'
    ft_dir = device_data_dir + 'data/image/'
    result_dir = device_data_dir + 'result/'
    coef_dir = device_data_dir + 'data/coeff/'
    ########################################################
    fpc = open(os.path.join(log_dir, 'raw_list.txt'), 'w')
    fpc.write(out_nodes)
    ########################################################
    cnt = 0
    f = open(log_path + '/{}_test_psnr_and_ssim.txt'.format(cfg.model_name), 'w')
    with torch.no_grad():
        for noisy_level_ind, iso_ind in enumerate(iso_list):
            coeff_a = np.reshape(a_list[noisy_level_ind], (1, 1, 1, 1)).astype(np.float32)
            coeff_b = np.reshape(b_list[noisy_level_ind], (1, 1, 1, 1)).astype(np.float32)
            coef_a_name = 'coeff_a_ISO{}.raw'.format(iso_ind)
            coef_b_name = 'coeff_b_ISO{}.raw'.format(iso_ind)
            coef_a_path = os.path.join(output_dir_coeff, coef_a_name)
            coef_b_path = os.path.join(output_dir_coeff, coef_b_name)
            coeff_a.tofile(coef_a_path)
            coeff_b.tofile(coef_b_path)
            coeff_a = torch.from_numpy(coeff_a).to(cfg.device)
            coeff_b = torch.from_numpy(coeff_b).to(cfg.device)

            iso_average_raw_psnr = 0
            iso_average_raw_ssim = 0
            for scene_ind in scene_list:
                output_dir = log_path + '/scene{}/ISO{}/'.format(scene_ind, iso_ind)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(output_dir+'gt/'):
                    os.makedirs(output_dir+'gt/')
                if not os.path.exists(output_dir+'noisy/'):
                    os.makedirs(output_dir+'noisy/')
                if not os.path.exists(output_dir+'fusion/'):
                    os.makedirs(output_dir+'fusion/')
                if not os.path.exists(output_dir+'denoise/'):
                    os.makedirs(output_dir+'denoise/')
                if not os.path.exists(output_dir+'refine/'):
                    os.makedirs(output_dir+'refine/')
                if not os.path.exists(output_dir+'omega/'):
                    os.makedirs(output_dir+'omega/')
                if not os.path.exists(output_dir+'gamma/'):
                    os.makedirs(output_dir+'gamma/')
                frame_avg_raw_psnr = 0
                frame_avg_raw_ssim = 0
                for idx in range(1, len(frame_list) + 1):
                    ########################################################
                    # gt
                    ########################################################
                    name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_gt{}.npy'.format(scene_ind, iso_ind,
                                                                                 iso_ind, scene_ind,
                                                                                 frame_list[idx - 1], 400)
                    gt_name = os.path.join(data_dir, name)
                    raw_gt = np.load(gt_name, mmap_mode='r')
                    raw_gt_bin = binning_raw(raw_gt)
                    fgt = pixel_unShuffle(raw_gt_bin)
                    raw_gt_bin_path = os.path.join(output_dir,'gt/frame_{}.npy'.format(idx))
                    # np.save(raw_gt_bin_path, raw_gt_bin)
                    gbrg_to_rgb_dispaly(raw_gt_bin_path, raw_gt_bin)
                    ########################################################
                    # noisy
                    ########################################################
                    noisy_id = 6 # np.random.randint(0, 10)
                    name = 'ISO{}_scene{}_frame{}_noisy{}'.format(iso_ind, scene_ind,
                                                                  frame_list[idx - 1],
                                                                  noisy_id)
                    spath = 'scene{}/ISO{}/{}.npy'.format(scene_ind, iso_ind, name)
                    input_name = os.path.join(data_dir, spath)
                    raw_noisy = np.load(input_name, mmap_mode='r')
                    raw_noisy_path = os.path.join(output_dir, 'noisy/frame_{}.npy'.format(idx))
                    # np.save(raw_noisy_path, raw_noisy)
                    gbrg_to_rgb_dispaly(raw_noisy_path, raw_noisy)
                    ########################################################
                    tmp = np.pad(raw_noisy, [(0, 0), (8, 8)])
                    noisy = torch.from_numpy(np.expand_dims(np.expand_dims(tmp,
                                            axis=0), axis=0)).cuda()
                    ft1 = noisy
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)
                    ft1 = conv_bin(ft1)
                    ######################################################
                    ft1_np = ft1.data.cpu().numpy().astype(np.float32)
                    ft1_np_bhwc = np.transpose(ft1_np, (0, 2, 3, 1))
                    ft1_np_bhwc_name = '{}.raw'.format(name) # '{:0>4d}_{}.raw'.format(cnt, name)
                    ft1_np_bhwc_path = os.path.join(output_dir_input,
                                                    ft1_np_bhwc_name)
                    ft1_np_bhwc.tofile(ft1_np_bhwc_path)
                    ######################################################
                    tmp_path = os.path.join(log_path, 'raw_list_tmp.txt')
                    ftmp = open(tmp_path, 'w')
                    ftmp.write(out_nodes)
                    ft1_device_path = os.path.join(ft_dir, ft1_np_bhwc_name)
                    if cnt % len(frame_list) == 0:
                        ft0_np_bhwc_path_tmp = ft1_np_bhwc_path
                        ft0_np_bhwc_path = ft1_np_bhwc_path
                        ft0_device_path = ft1_device_path
                    else:
                        ft0_np_bhwc_name = 'Result_{}/fusion.raw'.format(cnt-1)
                        ft0_np_bhwc_path_tmp = os.path.join(log_path, 'Result_0/fusion.raw')
                        ft0_np_bhwc_path = os.path.join(output_dir_input, '{:0>4d}_ft0.raw'.format(cnt))
                        ft0_np_bhwc.tofile(ft0_np_bhwc_path)
                        ft0_device_path = os.path.join(result_dir, ft0_np_bhwc_name)
                    content = '{} {} {} {}\n'.format(ft0_np_bhwc_path_tmp,
                                                     ft1_np_bhwc_path,
                                                     coef_a_path,
                                                     coef_b_path)
                    ftmp.write(content)
                    ftmp.close()
                    ######################################################
                    content = '{} {} {} {}\n'.format(ft0_np_bhwc_path,
                                                     ft1_np_bhwc_path,
                                                     coef_a_path,
                                                     coef_b_path)
                    fpc.write(content)
                    fpc.flush()
                    ######################################################
                    content = '{} {} {} {}\n'.format(ft0_device_path,
                                                     ft1_device_path,
                                                     os.path.join(coef_dir, coef_a_name),
                                                     os.path.join(coef_dir, coef_b_name))
                    fdevice.write(content)
                    fdevice.flush()
                    ######################################################
                    command = 'snpe-net-run --container {} --input_list {} --output_dir {}'.format(dlc_path,
                                                                                                   tmp_path,
                                                                                                   log_path)
                    os.system(command)
                    ########################################################
                    # out
                    ########################################################
                    fusion_path = os.path.join(log_path, 'Result_0/fusion.raw')
                    fusion = np.fromfile(fusion_path, dtype=np.float32)
                    ft0_np_bhwc = fusion
                    fusion = np.reshape(fusion, raw_shape)
                    fusion = np.transpose(fusion, (0, 3, 1, 2))

                    denoise_path = os.path.join(log_path, 'Result_0/denoise.raw')
                    denoise = np.fromfile(denoise_path, dtype=np.float32)
                    denoise = np.reshape(denoise, raw_shape)
                    denoise = np.transpose(denoise, (0, 3, 1, 2))

                    refine_path = os.path.join(log_path, 'Result_0/refine.raw')
                    refine = np.fromfile(refine_path, dtype=np.float32)
                    refine = np.reshape(refine, raw_shape)
                    refine = np.transpose(refine, (0, 3, 1, 2))

                    gamma_path = os.path.join(log_path, 'Result_0/gamma.raw')
                    gamma = np.fromfile(gamma_path, dtype=np.float32)
                    gamma = np.reshape(gamma, (1, 192, 256, 4))
                    gamma = np.transpose(gamma, (0, 3, 1, 2))

                    omega_path = os.path.join(log_path, 'Result_0/omega.raw')
                    omega = np.fromfile(omega_path, dtype=np.float32)
                    omega = np.reshape(omega, (1, 192, 256, 4))
                    omega = np.transpose(omega, (0, 3, 1, 2))

                    cnt += 1

                    fusion = torch.from_numpy(fusion)
                    denoise = torch.from_numpy(denoise)
                    refine = torch.from_numpy(refine)
                    omega = torch.from_numpy(omega)
                    gamma = torch.from_numpy(gamma)

                    for i in range(cfg.px_num - 1):
                        fusion = pixel_shuffle(fusion)
                        denoise = pixel_shuffle(denoise)
                        refine = pixel_shuffle(refine)
                    for i in range(cfg.px_num - 2):
                        omega = pixel_shuffle(omega)
                        gamma = pixel_shuffle(gamma)

                    fusion = fusion[0, 0, :, 4: -4].data.cpu().numpy()
                    gbrg_to_rgb_dispaly(os.path.join(output_dir, 'fusion/frame_{}.npy'.format(idx)), raw_npy=fusion)

                    denoise = denoise[0, 0, :, 4: -4].data.cpu().numpy()
                    gbrg_to_rgb_dispaly(os.path.join(output_dir, 'denoise/frame_{}.npy'.format(idx)), raw_npy=denoise)

                    refine = refine[0, 0, :, 4: -4].data.cpu().numpy()
                    gbrg_to_rgb_dispaly(os.path.join(output_dir, 'refine/frame_{}.npy'.format(idx)), raw_npy=refine)

                    cv2.imwrite(output_dir + 'omega/frame_%s.png' % idx,
                                np.uint8(omega[0, 0].data.cpu().numpy() * 255))
                    cv2.imwrite(output_dir + 'gamma/frame_%s.png' % idx,
                                np.uint8(gamma[0, 0].data.cpu().numpy() * 255))

                    fout = pixel_unShuffle(refine)

                    test_noisy_psnr = compare_psnr(raw_gt,
                                                 raw_noisy,
                                                 data_range=1.0)

                    test_raw_psnr = compare_psnr(fgt,
                                                 fout,
                                                 data_range=1.0)
                    test_raw_ssim = 0
                    for i in range(4):
                        test_raw_ssim += compare_ssim(fgt[0, i, :, :],
                                                      fout[0, i, :, :],
                                                      data_range=1.0)
                    test_raw_ssim /=4
                    context = 'scene {} iso {} frame{} ' \
                              'test noisy psnr : {}, ' \
                              'test raw psnr : {}, ' \
                              'test raw ssim : {} '.format(scene_ind,
                                                           iso_ind,
                                                           str(idx),
                                                           test_noisy_psnr,
                                                           test_raw_psnr,
                                                           test_raw_ssim) + '\n'
                    f.write(context)
                    print(context)
                    frame_avg_raw_psnr += test_raw_psnr
                    frame_avg_raw_ssim += test_raw_ssim
                frame_avg_raw_psnr = frame_avg_raw_psnr / len(frame_list)
                frame_avg_raw_ssim = frame_avg_raw_ssim / len(frame_list)
                context = 'frame average raw psnr:{}, ' \
                          'frame average raw ssim:{}'.format(frame_avg_raw_psnr,
                                                             frame_avg_raw_ssim) + '\n'
                f.write(context)
                print(context)
                iso_average_raw_psnr += frame_avg_raw_psnr
                iso_average_raw_ssim += frame_avg_raw_ssim
            iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
            iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)
            context = 'iso average raw psnr:{}, ' \
                      'iso average raw ssim:{}'.format(iso_average_raw_psnr,
                                                       iso_average_raw_ssim) + '\n'
            f.write(context)
            print(context)
            scene_avg_raw_psnr += iso_average_raw_psnr
            scene_avg_raw_ssim += iso_average_raw_ssim
        scene_avg_raw_psnr = scene_avg_raw_psnr / len(scene_list)
        scene_avg_raw_ssim = scene_avg_raw_ssim / len(scene_list)
        context = 'scene average raw psnr:{},' \
                  'scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
                                                           scene_avg_raw_ssim) + '\n'
        f.write(context)
        print(context)
        f.close()
        fdevice.close()
        fpc.close()

def eval_Mi_k1_out():
    data_dir = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/tangyouyun/' \
               'mi_k1_pair_data/'
    device = cfg.device
    cfg.model_name = 'model_fuLoss_deLoss'  # 'model_deLoss' # 'model' #
    log_dir = cfg.log_dir + 'log/{}/'.format(cfg.model_name)
    scene_list = [3]
    a_list = [0.109696, 0.21690887, 0.43752982, 0.89052248, 1.88346616]
    b_list = [0.30756074, 0.50924099, 1.1088187, 3.12988574, 8.53779323]
    iso_list = [200, 400, 800, 1600, 3200]
    iso_list.sort(reverse=True)
    a_list.sort(reverse=True)
    b_list.sort(reverse=True)
    # scene_list.sort(reverse=True)
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4]

    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    log_path = log_dir + '/results_K1_night_view/' + '{}'.format(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f = open(log_path + '/{}_test_psnr_and_ssim.txt'.format('night_view'), 'w')
    with torch.no_grad():
        for scene_ind in scene_list:
            iso_average_raw_psnr = 0
            iso_average_raw_ssim = 0
            for noisy_level_ind, iso_ind in enumerate(iso_list):
                output_dir = log_path + '/scene{}/ISO{}/'.format(scene_ind, iso_ind)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(output_dir + 'gt/'):
                    os.makedirs(output_dir + 'gt/')
                if not os.path.exists(output_dir + 'night_view/'):
                    os.makedirs(output_dir + 'night_view/')
                frame_avg_raw_psnr = 0
                frame_avg_raw_ssim = 0
                for idx in range(1, len(frame_list) + 1):
                    ########################################################
                    # gt
                    ########################################################
                    name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_gt{}.npy'.format(scene_ind, iso_ind,
                                                                                 iso_ind, scene_ind,
                                                                                 frame_list[idx - 1], 400)
                    gt_name = os.path.join(data_dir, name)
                    raw_gt = np.load(gt_name, mmap_mode='r')
                    raw_gt_bin = binning_raw(raw_gt)
                    fgt = pixel_unShuffle(raw_gt_bin)
                    raw_gt_bin_path = os.path.join(output_dir, 'gt/frame_{}.npy'.format(idx))
                    # np.save(raw_gt_bin_path, raw_gt_bin)
                    gbrg_to_rgb_dispaly(raw_gt_bin_path, raw_gt_bin)
                    ########################################################
                    # night_view
                    ########################################################
                    frame_index = 6 # np.random.randint(0, 10)
                    name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_output{}.npy'.format(scene_ind, iso_ind,
                                                                                    iso_ind, scene_ind,
                                                                                    frame_list[idx - 1],
                                                                                    frame_index)
                    input_name = os.path.join(data_dir, name)
                    raw_night_view = np.load(input_name, mmap_mode='r')
                    raw_night_view_path = os.path.join(output_dir, 'night_view/frame_{}.npy'.format(idx))
                    # np.save(raw_night_view_path, raw_night_view)
                    gbrg_to_rgb_dispaly(raw_night_view_path, raw_night_view)
                    fn = pixel_unShuffle(raw_night_view)

                    test_psnr = compare_psnr(fgt,
                                                   fn,
                                                   data_range=1.0)
                    test_raw_ssim = 0
                    for i in range(4):
                        test_raw_ssim += compare_ssim(fgt[0, i, :, :],
                                                      fn[0, i, :, :],
                                                      data_range=1.0)
                    test_raw_ssim /= 4
                    context = 'scene {} iso {} frame{} ' \
                              'test raw psnr : {}, ' \
                              'test raw ssim : {} '.format(scene_ind,
                                                           iso_ind,
                                                           str(idx),
                                                           test_psnr,
                                                           test_raw_ssim) + '\n'
                    f.write(context)
                    print(context)
                    frame_avg_raw_psnr += test_psnr
                    frame_avg_raw_ssim += test_raw_ssim
                frame_avg_raw_psnr = frame_avg_raw_psnr / len(frame_list)
                frame_avg_raw_ssim = frame_avg_raw_ssim / len(frame_list)
                context = 'frame average raw psnr:{}, ' \
                          'frame average raw ssim:{}'.format(frame_avg_raw_psnr,
                                                             frame_avg_raw_ssim) + '\n'
                f.write(context)
                f.flush()
                print(context)
                iso_average_raw_psnr += frame_avg_raw_psnr
                iso_average_raw_ssim += frame_avg_raw_ssim
            iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
            iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)
            context = 'iso average raw psnr:{}, ' \
                      'iso average raw ssim:{}'.format(iso_average_raw_psnr,
                                                       iso_average_raw_ssim) + '\n'
            f.write(context)
            f.flush()
            print(context)
            scene_avg_raw_psnr += iso_average_raw_psnr
            scene_avg_raw_ssim += iso_average_raw_ssim
        scene_avg_raw_psnr = scene_avg_raw_psnr / len(scene_list)
        scene_avg_raw_ssim = scene_avg_raw_ssim / len(scene_list)
        context = 'scene average raw psnr:{},' \
                  'scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
                                                           scene_avg_raw_ssim) + '\n'
        f.write(context)
        print(context)
        f.close()


def main():
    setup_seed(666)
    # eval_Mi_k1_out()
    eval_Mi_k1_dlc()
    # eval_Mi_k1()
    # eval_DRV()

    print('end')

if __name__ == '__main__':
    main()