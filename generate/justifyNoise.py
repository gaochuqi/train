import os
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.stats import poisson, entropy
from dataset import *
import rawpy
import imageio
from scipy import special

# path = '/home/pb/yushu/DRV/long/0001/0001_5.ARW'
# with rawpy.imread(path) as raw:
#     rgb = raw.postprocess()
# imageio.imsave('0001_5.tiff', rgb)

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


black_level = 240.0
iso_list = [1600, 3200, 6400, 12800, 25600]
####################### dataset a b ###########################
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
####################### simulated a b ###########################
# a_list = [3.2106428, 6.2345686, 12.120859, 23.718119, 46.112926]
# b_list = [5.668213, 32.32422, 126.640625, 439.41406, 1664.7812]

scene_ind = 5
noisy_level = 0
frame_ind = 4

data_dir = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/'

gt_name = os.path.join(data_dir,
						'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
						scene_ind, scene_ind, iso_list[noisy_level],
						frame_ind))

noisy_frame_index_for_current = 2 # np.random.randint(0, 10)
noisy_name1 = os.path.join(data_dir,
                        'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(
                        scene_ind, scene_ind, iso_list[noisy_level],
                        frame_ind, noisy_frame_index_for_current))

noisy_frame_index_for_current = 9
noisy_name2 = os.path.join(data_dir,
                        'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(
                        scene_ind, scene_ind, iso_list[noisy_level],
                        frame_ind, noisy_frame_index_for_current))


######################## Mi 11 Ultra #####################
# gt_name = './rawIMG_20220112_202128-310_req[13]_b[0]_BPS[0][0]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotYUVHAL_iso1600_clean_isp.png'
# noisy_name1 = './rawIMG_20220112_202128-310_req[13]_b[0]_BPS[0][0]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotYUVHAL_iso1600_noisy_isp.png'
# noisy_name2 = './rawIMG_20220112_202128-310_req[13]_b[0]_BPS[0][0]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotYUVHAL_iso1600_noisy.png'
# black_level = 64.0
# iso_list = [50, 100, 200, 400, 800, 1600, 3200]
# noisy_level = 5
# a_list = [0.02886277, 0.05565189, 0.109696, 0.21690887, 0.43752982, 0.89052248, 1.88346616]
# b_list = [0.31526701, 0.49276135, 0.30756074, 0.50924099, 1.1088187, 3.12988574, 8.53779323]
##########################################################

gt_raw = cv2.imread(gt_name, -1) # (1080, 1920)
noisy_raw1 = cv2.imread(noisy_name1, -1)
noisy_raw2 = cv2.imread(noisy_name2, -1)
noisy_im = generate_noisy_raw(gt_raw, a_list[noisy_level], b_list[noisy_level], black_level=black_level)


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
    return KL


KL1 = noise_KL(noisy_raw1, noisy_raw2, gt_raw)
KL2 = noise_KL(noisy_raw1, noisy_im.astype(np.uint16), gt_raw)
print(KL1, KL2)
# KL2_ = noise_KL(noisy_im, noisy_raw1, gt_raw)
# KL3 = noise_KL(noisy_raw2, noisy_im, gt_raw)
# print(KL1, KL2, KL2_, KL3)
# print(noise_KL(noisy_raw2, noisy_raw1, gt_raw))