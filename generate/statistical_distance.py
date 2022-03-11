import os
import glob
from pathlib import Path
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import sklearn.metrics as metrics
from scipy.stats import entropy
import cv2
from PIL import Image

from tools import setup_seed

def calculate_kl_divergence_v1(p,q):
    return np.sum(rel_entr(p, q))

def calculate_kl_divergence_v2(p,q):
    metrics.mutual_info_score(p,q)

def calculate_kl_divergence_v3(p,q):
    a = np.asarray(p, dtype=np.float)
    b = np.asarray(q, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def calculate_kl_divergence_v4(p,q):
    return entropy(p, q)

def normal(v, vmin, vmax):
    return np.clip(np.maximum(v - vmin, 0) / (vmax - vmin), 1e-8, 1.0)

def test_Mi11Ultra():
    raw_patten = '/media/wen/C14D581BDA18EBFA1/work/dataset/' \
                 'Mi11Ultra/10600/2022_01_25/2022_01_25/' \
                 'iso3200ex1000000/camera/raw/*.raw'

    base_name = 'iso800ex3000000'

    out_dir = './synthesisVideo/out/Mi11Ultra/20220125/'

    iso = 3200
    height = 3072
    width = 4080
    black_level = 64
    white_level = 2 ** 10 - 1

    raw_list = glob.glob(raw_patten)
    raw_path_noisy_1 = raw_list[len(raw_list) // 2]
    raw_path_noisy_2 = raw_list[len(raw_list) // 3]

    raw_path_clean = out_dir + base_name + '_clean.tiff'
    raw_clean = cv2.imread(raw_path_clean, -1)

    noisy_raw = np.fromfile(raw_path_noisy_1, dtype=np.uint16)
    noisy_raw_1 = np.reshape(noisy_raw, (height, width))

    noisy_raw = np.fromfile(raw_path_noisy_2, dtype=np.uint16)
    noisy_raw_2 = np.reshape(noisy_raw, (height, width))

    raw_noisy_path = out_dir + base_name + '_clean_iso{}_noisy.tiff'.format(iso)
    raw_noisy = cv2.imread(raw_noisy_path, -1)

    # KL1 = noise_KL(noisy_raw_1 - black_level,
    #                noisy_raw_2 - black_level,
    #                raw_clean - black_level)
    #
    # KL2 = noise_KL(noisy_raw_1 - black_level,
    #                raw_noisy - black_level,
    #                raw_clean - black_level)
    #
    # print(KL1, KL2)

    KL1 = noise_KL_v2(noisy_raw_1 - black_level,
                      noisy_raw_2 - black_level,
                      raw_clean - black_level)
    KL2 = noise_KL_v2(noisy_raw_1 - black_level,
                      raw_noisy - black_level,
                      raw_clean - black_level)

    print(KL1, KL2)

def test_CRVD():
    data_dir = '/media/wen/C14D581BDA18EBFA/work/dataset/CRVD_dataset' \
               '/indoor_raw_noisy/' \
               'indoor_raw_noisy_scene5/scene5/ISO3200'
    base_name = 'indoor_raw_noisy_scene5_frame5_ISO3200'

    out_dir = './synthesisVideo/out/Mi11Ultra/'

    iso = 3200  # 800
    height = 1080
    width = 1920
    black_level = 240
    white_level = 2 ** 12 - 1

    raw_path_clean = data_dir + '/frame5_clean.tiff'
    raw_path_noisy_1 = data_dir + '/frame5_noisy1.tiff'
    raw_path_noisy_2 = data_dir + '/frame5_noisy5.tiff'

    raw_clean = cv2.imread(raw_path_clean, -1)

    noisy_raw_1 = cv2.imread(raw_path_noisy_1, -1)
    noisy_raw_2 = cv2.imread(raw_path_noisy_2, -1)

    raw_noisy_path = './synthesisVideo/out/crvd_scene5_frame5/frame5_clean_iso3200_noisy.tiff'
    raw_noisy = cv2.imread(raw_noisy_path, -1)

    # KL1 = noise_KL(noisy_raw_1 - black_level,
    #                noisy_raw_2 - black_level,
    #                raw_clean - black_level)
    #
    # KL2 = noise_KL(noisy_raw_1 - black_level,
    #                raw_noisy - black_level,
    #                raw_clean - black_level)
    #
    # print(KL1, KL2)

    KL1 = noise_KL_v2(noisy_raw_1 - black_level,
                   noisy_raw_2 - black_level,
                   raw_clean - black_level)
    KL2 = noise_KL_v2(noisy_raw_1 - black_level,
                   raw_noisy - black_level,
                   raw_clean - black_level)

    print(KL1, KL2)


def pmf(RVs, all_possible_values):
    pmf = []
    ele_num = len(RVs)
    for v in all_possible_values:
        pmf.append(np.sum(RVs == v) / ele_num)
    return np.asarray(pmf)

def noise_KL(noisy1, noisy2, clean):
    noise1 = noisy1 - clean
    noise2 = noisy2 - clean
    y_values = list(np.unique(clean))
    KL = 0.
    for y in y_values:
        indices = (clean == y)
        discrete_vals1 = noise1[indices]
        discrete_vals2 = noise2[indices]
        all_possible_values = np.unique(np.concatenate((discrete_vals1, discrete_vals2)))
        pmf1 = pmf(discrete_vals1, all_possible_values)
        pmf2 = pmf(discrete_vals2, all_possible_values)
        pmf_indices = np.logical_and(pmf1 > 0, pmf2 > 0)
        KL_y = entropy(pmf1[pmf_indices], pmf2[pmf_indices])
        KL += KL_y
    KL /= len(y_values)
    return KL

def noise_KL_v2(noisy1, noisy2, clean):
    noise1 = noisy1 - clean
    noise2 = noisy2 - clean
    all_possible_values = np.unique(np.concatenate((noise1, noise2)))
    pmf1 = pmf(noise1, all_possible_values)
    pmf2 = pmf(noise2, all_possible_values)
    pmf_indices = np.logical_and(pmf1 > 0, pmf2 > 0)
    KL = entropy(pmf1[pmf_indices], pmf2[pmf_indices])
    return KL

def test():
    test_Mi11Ultra() # 64.24003183686075 135.37353971407154
    test_CRVD() # 374.7748112342128 386.42416969500044


def main():
    setup_seed(666)

    test()

    print('end')

if __name__ == '__main__':
    main()


# Mi11 iso 800  distance V4 0.0020296737019001513 0.001721901379031239 0.0030137253036782445 0.0056631568118141965 0.007694942206046501
# Mi11 iso 3200 distance V4 0.0020296737019001513 0.001721901379031239 0.0030137253036782445 0.0345687703205165 0.03661162205137862
# CRVD iso 3200 distance V4 0.007495169472145363 0.007561524943415529 0.021436743914942373 0.00832064800988565 0.02225486780201463

# 卡方分布 (Chi-squared distribution)