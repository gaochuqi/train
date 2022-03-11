import glob
import math
import csv
import os.path
import random
import torch
import torch.nn as nn
import cv2
import numpy as np
import yaml
import rawpy
import cv2
from PIL import Image
import os
from scipy.stats import poisson
import time

from tools import get_fit_curve, \
    PolynomialFeatures, pixel_unShuffle, \
    tensor2numpy, pixel_unShuffle_RGBG, \
    setup_seed
from statistical_distance import *
import config as cfg

from visualization import process_simple, raw2srgb_isp

isp = torch.load('ISP_CNN.pth')


def generate_noisy_raw(gt_raw, a, b, black_level=240, white_level=2 ** 12 - 1):
    """
    a: sigma_s^2
    b: sigma_r^2
    """
    gaussian_noise_var = b
    # tmp = poisson((gt_raw-black_level) / a)
    tmp = poisson(np.maximum((gt_raw - black_level), 0) / a)
    poisson_noisy_img = tmp.rvs() * a
    gaussian_noise = np.sqrt(gaussian_noise_var) * np.random.randn(gt_raw.shape[0], gt_raw.shape[1])
    noisy_img = poisson_noisy_img + gaussian_noise + black_level
    noisy_img = np.minimum(np.maximum(noisy_img, 0), white_level)

    return noisy_img


def generate_noisy_MOT17():
    iso_list = [1600, 3200, 6400, 12800, 25600]
    a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
    b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

    for data_id in ['02', '09', '10', '11']:

        raw_paths = glob.glob('./data/SRVD_data/raw_clean/MOT17-{}_raw/*.tiff'.format(data_id))
        save_path = './data/SRVD_data/raw_noisy/MOT17-{}_raw/'.format(data_id)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        for raw_path in raw_paths:
            clean_raw = cv2.imread(raw_path, -1)
            for noisy_level in range(1, 5 + 1):
                iso = iso_list[noisy_level - 1]
                a = a_list[noisy_level - 1]
                b = b_list[noisy_level - 1]
                for noisy_id in range(0, 1 + 1):
                    noisy_raw = generate_noisy_raw(clean_raw.astype(np.float32), a, b)
                    base_name = os.path.basename(raw_path)[:-5]
                    np.save(save_path + base_name + '_iso{}_noisy{}.npy'.format(iso, noisy_id),
                            np.uint16(noisy_raw))
                    # noisy_save = Image.fromarray(np.uint16(noisy_raw))
                    # noisy_save.save(save_path + base_name + '_iso{}_noisy{}.tiff'.format(iso, noisy_id))
            # print('have synthesized noise on MOT17-{}_raw '.format(data_id) + base_name + '.tiff')
            print('have synthesized noise on MOT17-{}_raw '.format(data_id) + base_name + '.npy')


iso_list = [50, 100, 200, 400, 800, 1600, 3200]
a_list = [0.008301621831566883, 0.012232987794999914, 0.01897553040386134, 0.026815828637268613,
          0.04951847675832405, 0.09320560848193736, 0.19296701929339524]
b_list = [5.020177574588825, 9.297401628983224, 17.065627015159304, 31.831289977080548, 63.908340772328636,
          129.48238303547149, 269.0831211355436]


def get_CRVD_coef_a_b(X_test):
    iso_list = [1600, 3200, 6400, 12800, 25600]
    a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
    b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
    X_train = np.asarray(iso_list, dtype=np.float32).reshape(-1, 1) / 25600
    a_train = np.asarray(a_list, dtype=np.float32)
    b_train = np.asarray(b_list, dtype=np.float32)
    order = 3
    poly = PolynomialFeatures(order)
    lr_a = get_fit_curve(X_train, a_train, order)
    lr_b = get_fit_curve(X_train, b_train, order)
    a_test = []
    b_test = []
    for iso in X_test:
        X_test = np.asarray([iso], dtype=np.float32).reshape(-1, 1) / 25600
        X_test_ploy = poly.fit_transform(X_test)
        a = lr_a.predict(X_test_ploy)[0]
        b = lr_b.predict(X_test_ploy)[0]
        a_test.append(a)
        b_test.append(b)
    return a_test, b_test


def generate_noisy_DRV(dataset_dir='/home/wen/Documents/dataset/DRV/',
                       sub_folder = 'trainval_v3/',
                       height=3672,
                       width=5496,
                       ):

    iso_list = cfg.iso_list
    a_list = cfg.a_list
    b_list = cfg.b_list

    iso_list.sort(reverse=True)
    a_list.sort(reverse=True)
    b_list.sort(reverse=True)

    data_dir = dataset_dir + sub_folder

    black_level = cfg.black_level   # 800
    white_level = cfg.white_level   # 16380
    bayer_pattern = cfg.bayer_pattern # 'RGGB'  # 'GBRG' # 'GRBG' # 'BGGR' #
    noisy_num = cfg.noisy_num
    # valid_list = cfg.train_list + cfg.val_list
    num = 50
    valid_list = range(1, num + 1)
    for i in valid_list:
        if i==1:
            continue
        sub_folder = '{:0>4d}'.format(i)
        sub_dir = data_dir + sub_folder

        raw_pattern = sub_dir + '/*/raw/*.npy'
        raw_list = glob.glob(raw_pattern)
        raw_list.sort()
        video_type = raw_list[0].split('/')[-3]
        save_path = sub_dir + '/{}/noisy/'.format(video_type)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        for noisy_level in range(len(iso_list)):
            iso = iso_list[noisy_level]
            a = a_list[noisy_level]
            b = b_list[noisy_level]
            if not os.path.isdir(save_path + 'iso{}/'.format(iso)):
                os.makedirs(save_path + 'iso{}/'.format(iso))
            out_dir = save_path + 'iso{}/'.format(iso)
            for raw_path in raw_list:
                clean_raw = np.load(raw_path, mmap_mode='r')
                for noisy_id in range(1, noisy_num + 1):
                    noisy_raw = generate_noisy_raw(clean_raw.astype(np.float32),
                                                   a, b,
                                                   black_level, white_level)
                    base_name = os.path.basename(raw_path).split('.')[0]
                    np.save(out_dir + base_name + '_iso{}_noisy{}.npy'.format(iso, noisy_id),
                            np.uint16(noisy_raw))

                    with torch.no_grad():
                        if base_name == 'frame_7':
                            noisy_raw_norm = np.maximum(noisy_raw - black_level, 0) / (white_level - black_level)
                            noisy_raw_isp = tensor2numpy(isp(torch.from_numpy(
                                pixel_unShuffle_RGBG(noisy_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
                            cv2.imwrite(out_dir + base_name + '_iso{}_noisy{}.png'.format(iso, noisy_id),
                                    np.uint8(noisy_raw_isp * 255))

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  'have synthesized noise on DRV {} iso {}'.format(sub_folder, iso))


def test_generate_noisy_one_raw(raw_path,
                                height, width,
                                iso, a, b,
                                black_level, white_level,
                                red_gains, green_gains, blue_gains,
                                bayer_pattern,
                                out_dir='./'):
    if raw_path.endswith('.raw'):
        clean_raw = np.fromfile(raw_path, dtype=np.uint16)
        clean_raw = np.reshape(clean_raw, (height, width))
    else:
        clean_raw = cv2.imread(raw_path, -1)
    clean_raw_norm = np.maximum(clean_raw - black_level, 0) / (white_level - black_level)

    noisy_raw = generate_noisy_raw(clean_raw.astype(np.float32), a, b,
                                   black_level, white_level)
    base_name = os.path.basename(raw_path).split('.')[0]
    np.save(out_dir + base_name + '_iso{}_noisy.npy'.format(iso),
            np.uint16(noisy_raw))
    # noisy_save = Image.fromarray(np.uint16(noisy_raw))
    # noisy_save.save(out_dir + base_name + '_iso{}_noisy.tiff'.format(iso))

    noisy_process_sim = process_simple(np.uint16(noisy_raw),
                                       black_level, white_level,
                                       red_gains, green_gains, blue_gains,
                                       bayer_pattern)
    noisy_process_sim_save = Image.fromarray(np.uint8(noisy_process_sim * 255))
    noisy_process_sim_save.save(out_dir + base_name + '_iso{}_noisy_sISP_PIL.png'.format(iso))

    noisy_raw_norm = np.maximum(noisy_raw - black_level, 0) / (white_level - black_level)
    noisy_raw_norm_save = Image.fromarray(np.uint8(noisy_raw_norm * 255))
    noisy_raw_norm_save.save(out_dir + base_name + '_iso{}_noisy_PIL.png'.format(iso))

    isp = torch.load('ISP_CNN.pth')
    with torch.no_grad():
        clean_raw_norm_isp = \
            tensor2numpy(
                isp(torch.from_numpy(pixel_unShuffle_RGBG(clean_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
        cv2.imwrite(out_dir + base_name + '_iso{}_clean_aiISP_cv2.png'.format(iso), np.uint8(clean_raw_norm_isp * 255))

        noisy_raw_isp = \
        tensor2numpy(isp(torch.from_numpy(pixel_unShuffle_RGBG(noisy_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[
            0]
        cv2.imwrite(out_dir + base_name + '_iso{}_noisy_aiISP_cv2.png'.format(iso), np.uint8(noisy_raw_isp * 255))

    return

def test_one_raw_CRVD():
    raw_path = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/' \
               'indoor_raw_noisy/indoor_raw_noisy_scene5/scene5/ISO3200/' \
               'frame5_clean.tiff'
    iso_list = [1600, 3200, 6400, 12800, 25600]
    a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
    b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
    noise_level = 1
    iso = iso_list[noise_level]
    a = a_list[noise_level]
    b = b_list[noise_level]
    black_level = 240
    white_level = 2 ** 12 - 1
    height = 1080
    width = 1920
    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)
    green_gain = 1.0
    bayer_pattern = 'GBRG'
    test_generate_noisy_one_raw(raw_path,
                                height, width,
                                iso, a, b,
                                black_level, white_level,
                                red_gain, green_gain, blue_gain,
                                bayer_pattern)


def test_one_raw_Mi11Ultra():
    raw_path = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/' \
               'shuang/20220112/20220112/coef/ISO50exT256000000/camera/raw/' \
               'rawIMG_20220112_202128-310_req[13]_b[0]_BPS[0][0]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotYUVHAL.raw'
    height = 3072
    width = 4080
    iso_list = [50, 100, 200, 400, 800, 1600, 3200]
    a_list = [0.2879409, 0.3933852, 0.6042123, 1.0256209, 1.8674616, 3.5472803, 6.8918157]
    b_list = [1.3714607, 1.5073164, 1.8218268, 2.621904, 4.9051676, 12.195202, 37.59787]
    noise_level = 6
    iso = iso_list[noise_level]
    a = a_list[noise_level]
    b = b_list[noise_level]
    black_level = 64
    white_level = 2 ** 10 - 1
    # Red and blue gains represent white balance.
    red_gain = 2.110677  # random.uniform(1.9, 2.4)
    blue_gain = 1.465880  # random.uniform(1.5, 1.9)
    green_gain = 1.0
    bayer_pattern = 'GBRG'  # 'GRBG' # 'BGGR' # 'RGGB' #
    a = 3.5472803
    b = 37.59787
    out_dir = './synthesisVideo/out/' + 'a_{}_b_{}/'.format(a,b)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    test_generate_noisy_one_raw(raw_path,
                                height, width,
                                iso, a, b,
                                black_level, white_level,
                                red_gain, green_gain, blue_gain,
                                bayer_pattern,
                                out_dir)


def test_one_raw_Mi11Ultra_GrayScaleCard():
    isp = torch.load('ISP_CNN.pth')

    raw_patten = '/media/wen/C14D581BDA18EBFA1/work/dataset/' \
                 'Mi11Ultra/10600/2022_01_25/2022_01_25/' \
                 'iso800ex3000000/camera/raw/*.raw'
    base_name = 'iso800ex3000000'

    # meta data
    height = 3072
    width = 4080
    black_level = 64
    white_level = 2 ** 10 - 1
    # Red and blue gains represent white balance.
    red_gain = 2.139081  # random.uniform(1.9, 2.4)
    green_gain = 1.0
    blue_gain = 2.014098  # random.uniform(1.5, 1.9)
    bayer_pattern = 'GBRG'  # 'GRBG' # 'BGGR' # 'RGGB' #

    out_dir = './synthesisVideo/out/Mi11Ultra/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    raw_clean_path = out_dir + base_name + '_clean.tiff'
    raw_noisy_path = out_dir + base_name + '_noisy.tiff'
    if not os.path.exists(raw_clean_path):
        # multi frames average, to get clean frame
        raw_list = glob.glob(raw_patten)

        # get one noisy for compare
        raw_path_noisy = raw_list[len(raw_list) // 2]
        noisy_raw = np.fromfile(raw_path_noisy, dtype=np.uint16)
        noisy_raw = np.reshape(noisy_raw, (height, width))
        # PIL Image, tiff
        noisy_raw_save = Image.fromarray(np.uint16(noisy_raw))
        noisy_raw_save.save(out_dir + base_name + '_noisy.tiff')
        # PIL Image, png
        noisy_raw_norm = np.maximum(noisy_raw - black_level, 0) / (white_level - black_level)
        noisy_raw_norm_save = Image.fromarray(np.uint8(noisy_raw_norm * 255))
        noisy_raw_norm_save.save(out_dir + base_name + '_noisy_PIL.png')
        # Ai ISP model, cv2
        with torch.no_grad():
            noisy_raw_norm_isp = \
                tensor2numpy(
                    isp(torch.from_numpy(
                        pixel_unShuffle_RGBG(noisy_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
            cv2.imwrite(out_dir + base_name + '_noisy_aiISP_cv2.png', np.uint8(noisy_raw_norm_isp * 255))

        clean_raw_list = []
        for raw_path in raw_list:
            clean_raw = np.fromfile(raw_path, dtype=np.uint16)
            clean_raw = np.reshape(clean_raw, (height, width))
            clean_raw_list.append(clean_raw)
        raw_clean = np.stack(clean_raw_list, axis=-1)
        raw_clean = np.mean(raw_clean, axis=-1)
    else:
        raw_clean = cv2.imread(raw_clean_path, -1)
        raw_noisy = cv2.imread(raw_noisy_path, -1)

    bSave = False
    if bSave: # Save Clean
        ###################################################
        # Save
        ###################################################
        # PIL Image, tiff
        raw_clean_save = Image.fromarray(np.uint16(raw_clean))
        raw_clean_save.save(out_dir + base_name + '_clean.tiff')
        # Simple ISP, RGB gain, gamma
        raw_clean_process_sim = process_simple(np.uint16(raw_clean),
                                           black_level, white_level,
                                           red_gain, green_gain, blue_gain,
                                           bayer_pattern)
        raw_clean_process_sim_save = Image.fromarray(np.uint8(raw_clean_process_sim * 255))
        raw_clean_process_sim_save.save(out_dir + base_name + '_clean_sISP_PIL.png')

        clean_raw_norm = np.maximum(raw_clean - black_level, 0) / (white_level - black_level)
        # PIL Image, png
        clean_raw_norm_save = Image.fromarray(np.uint8(clean_raw_norm * 255))
        clean_raw_norm_save.save(out_dir + base_name + '_clean_PIL.png')

        # Ai ISP model, cv2
        with torch.no_grad():
            clean_raw_norm_isp = \
                tensor2numpy(
                    isp(torch.from_numpy(pixel_unShuffle_RGBG(clean_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
            cv2.imwrite(out_dir + base_name + '_clean_aiISP_cv2.png', np.uint8(clean_raw_norm_isp * 255))

    ###################################################
    # Add Noise
    ###################################################
    iso_list = [50, 100, 200, 400, 800, 1600, 3200]
    a_list = [0.2879409, 0.3933852, 0.6042123, 1.0256209, 1.8674616, 3.5472803, 6.8918157]
    b_list = [1.3714607, 1.5073164, 1.8218268, 2.621904, 4.9051676, 12.195202, 37.59787]
    noise_level = 4
    iso = iso_list[noise_level]
    a = a_list[noise_level]
    b = b_list[noise_level]

    # a = 3.5472803
    # b = 37.59787

    bAddNoise = False
    if bAddNoise:
        test_generate_noisy_one_raw(raw_clean_path,
                                    height, width,
                                    iso, a, b,
                                    black_level, white_level,
                                    red_gain, green_gain, blue_gain,
                                    bayer_pattern,
                                    out_dir)

    bEval = True
    if bEval:
        raw_noisy_path_gen = out_dir + base_name + '_clean_iso{}_noisy.tiff'.format(iso)
        noisy_gen = cv2.imread(raw_noisy_path_gen, -1)
        noisy_gen_norm = np.maximum(noisy_gen - black_level, 0) / (white_level - black_level)

        clean_norm = np.maximum(raw_clean - black_level, 0) / (white_level - black_level)
        noisy_norm = np.maximum(raw_noisy - black_level, 0) / (white_level - black_level)

        p = np.reshape(clean_norm, [-1])
        q1 = np.reshape(noisy_norm, [-1])
        q2 = np.reshape(noisy_gen_norm, [-1])
        print(p)
        print(q1)
        print(q2)

        dist1_ori = calculate_kl_divergence_v1(p, q1)
        dist1_gen = calculate_kl_divergence_v1(p, q2)
        print('distance V1', dist1_ori, dist1_gen)

        dist2_ori = calculate_kl_divergence_v2(np.reshape(raw_clean.astype(np.int), [-1]),
                                               np.reshape(raw_noisy.astype(np.int), [-1]))
        dist2_gen = calculate_kl_divergence_v2(np.reshape(raw_clean.astype(np.int), [-1]),
                                               np.reshape(noisy_gen.astype(np.int), [-1]))
        print('distance V2', dist2_ori, dist2_gen)

        dist3_ori = calculate_kl_divergence_v3(q1, p)
        dist3_gen = calculate_kl_divergence_v3(q2, p)
        print('distance V3', dist3_ori, dist3_gen)

        dist4_ori = calculate_kl_divergence_v4(p, q1)
        dist4_gen = calculate_kl_divergence_v4(p, q2)
        print('distance V4', dist4_ori, dist4_gen)


def test_one_raw_DRV():
    scene = '0007'
    frameID = 3
    raw_path = '/media/wen/09C1B27DA5EB573A/work/dataset/' \
               'DRV/trainval/{}/raw/frame_{}.raw'.format(scene, frameID)
    height = 3672
    width = 5496
    iso_list = [50, 100, 200, 400, 800, 1600, 3200]
    a_list = [0.2879409, 0.3933852, 0.6042123, 1.0256209, 1.8674616, 3.5472803, 6.8918157]
    b_list = [1.3714607, 1.5073164, 1.8218268, 2.621904, 4.9051676, 12.195202, 37.59787]
    noise_level = 6
    iso = iso_list[noise_level]
    a = a_list[noise_level]
    b = b_list[noise_level]
    black_level = 800
    white_level = 16380
    # Red and blue gains represent white balance.
    red_gain = random.uniform(1.9, 2.4)
    blue_gain = random.uniform(1.5, 1.9)
    green_gain = 1.0
    bayer_pattern = 'RGGB' # 'GBRG'  # 'GRBG' # 'BGGR' #
    out_dir = './synthesisVideo/out/' + 'DRV_{}_frame_{}/'.format(scene, frameID)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    test_generate_noisy_one_raw(raw_path,
                                height, width,
                                iso, a, b,
                                black_level, white_level,
                                red_gain, green_gain, blue_gain,
                                bayer_pattern,
                                out_dir)

def DRV_raw2rgb():
    isp = torch.load('ISP_CNN.pth')
    height = 3672
    width = 5496
    black_level = 800
    white_level = 16380
    bayer_pattern = 'RGGB'
    data_pattern = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/trainval/*/raw/*.raw'
    raw_list = glob.glob(data_pattern)
    raw_list.sort()
    for raw_path in raw_list:
        raw2srgb_isp(raw_path, isp,
                     height, width,
                     black_level, white_level,
                     bayer_pattern)

###################################################################
# CRVD
###################################################################
def generate_noisy_CRVD():
    data_root = '/media/wen/C14D581BDA18EBFA/work/dataset/CRVD_dataset'

    iso_list = [1600, 3200, 6400, 12800, 25600]
    a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
    b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

    iso_list.sort(reverse=True)
    a_list.sort(reverse=True)
    b_list.sort(reverse=True)

    black_level = 240
    white_level = 2 ** 12 - 1
    bayer_pattern = 'GBRG' # 'RGGB'  # 'GBRG' # 'GRBG' # 'BGGR' #
    noisy_num = cfg.noisy_num
    for scene_ind in range(1,12):
        for iso_ind in iso_list:
            for frame_ind in range(1,8):
                video_type = 'obj_motion'
                out_dir = data_root + '/synthesis/{}/scene{}/ISO{}/'.format(video_type, scene_ind, iso_ind)
                raw_path = out_dir + 'raw/frame_{}.npy'.format(frame_ind)
                if not os.path.exists(out_dir + '/noisy'):
                    os.makedirs(out_dir + '/noisy')
                clean_raw = np.load(raw_path, mmap_mode='r')
                for noisy_id in range(1, noisy_num + 1):
                    idx = iso_list.index(iso_ind)
                    a = a_list[idx]
                    b = b_list[idx]
                    noisy_raw = generate_noisy_raw(clean_raw.astype(np.float32),
                                                   a, b,
                                                   black_level, white_level)

                    out_path = data_root + '/synthesis/{}/' \
                                'scene{}/ISO{}/noisy/frame_{}_noisy_{}.npy'.format(video_type, scene_ind,
                                                                                  iso_ind, frame_ind, noisy_id)
                    np.save(out_path, np.uint16(noisy_raw))

                    with torch.no_grad():
                        if frame_ind == 1:
                            noisy_raw_norm = np.maximum(noisy_raw - black_level, 0) / (white_level - black_level)
                            noisy_raw_isp = tensor2numpy(isp(torch.from_numpy(
                                pixel_unShuffle_RGBG(noisy_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
                            out_path = data_root + '/synthesis/{}/' \
                                                   'scene{}/ISO{}/noisy/frame_{}_noisy_{}.png'.format(video_type,
                                                                                                     scene_ind, iso_ind,
                                                                                                     frame_ind,
                                                                                                     noisy_id)
                            cv2.imwrite(out_path, np.uint8(noisy_raw_isp * 255))
                video_type = 'cam_motion'
                out_dir = data_root + '/synthesis/{}/scene{}/ISO{}/'.format(video_type, scene_ind, iso_ind)
                raw_path = out_dir + 'raw/frame_{}.npy'.format(frame_ind)
                if not os.path.exists(out_dir + '/noisy'):
                    os.makedirs(out_dir + '/noisy')
                clean_raw = np.load(raw_path, mmap_mode='r')
                for noisy_id in range(1, noisy_num + 1):
                    idx = iso_list.index(iso_ind)
                    a = a_list[idx]
                    b = b_list[idx]
                    noisy_raw = generate_noisy_raw(clean_raw.astype(np.float32),
                                                   a, b,
                                                   black_level, white_level)

                    out_path = data_root + '/synthesis/{}/' \
                                           'scene{}/ISO{}/noisy/frame_{}_noisy_{}.npy'.format(video_type,
                                                                                             scene_ind, iso_ind,
                                                                                             frame_ind, noisy_id)
                    np.save(out_path, np.uint16(noisy_raw))

                    with torch.no_grad():
                        if frame_ind == 1:
                            noisy_raw_norm = np.maximum(noisy_raw - black_level, 0) / (white_level - black_level)
                            noisy_raw_isp = tensor2numpy(isp(torch.from_numpy(
                                pixel_unShuffle_RGBG(noisy_raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
                            out_path = data_root + '/synthesis/{}/' \
                                                   'scene{}/ISO{}/noisy/frame_{}_noisy_{}.png'.format(video_type,
                                                                                                     scene_ind, iso_ind,
                                                                                                     frame_ind,
                                                                                                     noisy_id)
                            cv2.imwrite(out_path, np.uint8(noisy_raw_isp * 255))

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                  'have synthesized noise on CRVD scene {} iso {}'.format(scene_ind, iso_ind))


def main():
    setup_seed(666)

    generate_noisy_CRVD()

    # test_one_raw_DRV()
    # test_one_raw_CRVD()
    # test_one_raw_Mi11Ultra_GrayScaleCard()

    # generate_noisy_DRV()

    # DRV_raw2rgb()
    # test_one_raw_Mi11Ultra()

    print('end')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
