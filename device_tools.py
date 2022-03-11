import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import warnings
warnings.filterwarnings('ignore')
import cv2

import config as cfg
from arch.modules import PixelShuffle
from arch import architecture_reparam
from generate.visualization import gbrg_to_rgb_dispaly

pixel_shuffle = PixelShuffle(2)
pixel_unshuffle = PixelShuffle(0.5)
# Mi11Ultra calibrate
a_list = [0.109696, 0.21690887, 0.43752982, 0.89052248, 1.88346616]
b_list = [0.30756074, 0.50924099, 1.1088187, 3.12988574, 8.53779323]
iso_list = [200, 400, 800, 1600, 3200]
iso_list.sort(reverse=True)
a_list.sort(reverse=True)
b_list.sort(reverse=True)
data_dir = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/tangyouyun/' \
           'mi_k1_pair_data/'

def get_device_data():
    scene_list = [1, 2, 3]
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4]
    ########################################################
    output_dir = os.path.join(data_dir, 'device_input_after_binning_bhwc')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir_coeff = os.path.join(data_dir, 'device_coeff')
    if not os.path.exists(output_dir_coeff):
        os.makedirs(output_dir_coeff)
    ########################################################
    device_data_dir = '/data/local/tmp/wen/'
    ft_dir = device_data_dir + 'data/image/'
    coef_dir = device_data_dir + 'data/coeff/'
    ########################################################
    log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    bin_np = np.load('{}/binning.npy'.format(log_dir)).astype(np.float32)
    bin_w = torch.from_numpy(bin_np)
    binnings = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
    binnings.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    conv_bin = binnings.to(cfg.device)
    ########################################################
    name = 'model_best'
    state_dict = torch.load(os.path.join(log_dir, '{}_reparam.pth'.format(name)))
    model = architecture_reparam.EMVD(cfg)
    model = model.to(cfg.device)
    model.load_state_dict(state_dict, strict=True)
    ########################################################
    fpc = open(os.path.join(log_dir, 'raw_list.txt'), 'w')
    # out_names=['fusion', 'denoise', 'refine', 'omega', 'gamma']
    fpc.write('#Conv_216 Conv_218 Conv_220 Sigmoid_208 Sigmoid_180\n')
    fdevice = open(os.path.join(data_dir, 'raw_list.txt'), 'w')
    fdevice.write('#Conv_216 Conv_218 Conv_220 Sigmoid_208 Sigmoid_180\n')
    ########################################################
    cnt = 0
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
            for scene_ind in scene_list:
                for idx in frame_list:
                    noisy_id = 6 # np.random.randint(0, 10)
                    name = 'ISO{}_scene{}_frame{}_noisy{}'.format(iso_ind, scene_ind, idx,
                                                                  noisy_id)
                    spath = 'scene{}/ISO{}/{}.npy'.format(scene_ind, iso_ind, name)
                    input_name = os.path.join(data_dir, spath)
                    raw_noisy = np.load(input_name, mmap_mode='r')

                    tmp = np.pad(raw_noisy, [(0, 0), (8, 8)]).astype(np.float32)
                    ft1 = torch.from_numpy(np.expand_dims(np.expand_dims(tmp, axis=0), axis=0)).cuda()
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)
                    ft1 = conv_bin(ft1)

                    if cnt % len(frame_list) == 0:
                        ft0 = ft1
                    cnt += 1
                    ######################################################
                    ft1_np = ft1.data.cpu().numpy().astype(np.float32)
                    ft1_np_bhwc = np.transpose(ft1_np, (0, 2, 3, 1))
                    ft1_np_bhwc_name = '{:0>4d}_{}.raw'.format(cnt, name)
                    ft1_np_bhwc_path = os.path.join(output_dir, ft1_np_bhwc_name)
                    ft1_np_bhwc.tofile(ft1_np_bhwc_path)

                    ft0_np = ft0.data.cpu().numpy().astype(np.float32)
                    ft0_np_bhwc = np.transpose(ft0_np, (0, 2, 3, 1))
                    ft0_np_bhwc_name = '{:0>4d}_ft0.raw'.format(cnt)
                    ft0_np_bhwc_path = os.path.join(output_dir, ft0_np_bhwc_name)
                    ft0_np_bhwc.tofile(ft0_np_bhwc_path)
                    ######################################################
                    content = '{} {} {} {}\n'.format(ft0_np_bhwc_path,
                                                     ft1_np_bhwc_path,
                                                     coef_a_path,
                                                     coef_b_path)
                    fpc.write(content)
                    ######################################################
                    content = '{} {} {} {}\n'.format(os.path.join(ft_dir, ft0_np_bhwc_name),
                                                     os.path.join(ft_dir, ft1_np_bhwc_name),
                                                     os.path.join(coef_dir, coef_a_name),
                                                     os.path.join(coef_dir, coef_b_name))
                    fdevice.write(content)
                    ######################################################
                    fusion, denoise, refine, omega, gamma = model(ft0, ft1, coeff_a, coeff_b)
                    ft0 = fusion
    print(cnt)
    fpc.close()
    fdevice.close()




def temp():
    import shutil
    ddir = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/tangyouyun/mi_k1_pair_data/scene2/ISO3200/*'
    dlist = glob.glob(ddir)
    for opath in dlist:
        oname = os.path.basename(opath)
        nname = oname[:13] + '2' + oname[14:]
        npath = os.path.join(os.path.dirname(opath), nname)
        shutil.copy(opath, npath)

def device_raw_list():
    scene_list = [1, 2, 3]
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4]
    device_data_dir = '/data/local/tmp/wen/'
    ft0_dir = device_data_dir + 'data/image/'
    ft1_dir = device_data_dir + 'data/image/'
    coef_dir = device_data_dir + 'data/coeff/'
    f = open(os.path.join(data_dir,'raw_list.txt'), 'w')
    # out_names=['fusion', 'denoise', 'refine', 'omega', 'gamma']
    f.write('#Conv_216 Conv_218 Conv_220 Sigmoid_208 Sigmoid_180\n')
    for noisy_level_ind, iso_ind in enumerate(iso_list):
        for scene_ind in scene_list:
            for idx in frame_list:
                noisy_id = 6
                ft1_name = 'ISO{}_scene{}_frame{}_noisy{}.raw'.format(iso_ind, scene_ind, idx,
                                                              noisy_id)
                ft0_name = ft1_name
                content = '{} {} {} {}\n'.format(ft0_dir + ft0_name,
                                                ft1_dir + ft1_name,
                                                coef_dir + 'coeff_a_ISO{}.raw'.format(iso_ind),
                                                coef_dir + 'coeff_b_ISO{}.raw'.format(iso_ind))
                f.write(content)

def pc_raw_list():
    scene_list = [1, 2, 3]
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4]
    log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    ft0_dir = os.path.join(data_dir, 'device_input_after_binning_bhwc/')
    ft1_dir = os.path.join(data_dir, 'device_input_after_binning_bhwc/')
    coef_dir = os.path.join(data_dir, 'device_coeff/')
    f = open(os.path.join(log_dir,'raw_list.txt'), 'w')
    # out_names=['fusion', 'denoise', 'refine', 'omega', 'gamma']
    f.write('#Conv_216 Conv_218 Conv_220 Sigmoid_208 Sigmoid_180\n')
    for noisy_level_ind, iso_ind in enumerate(iso_list):
        for scene_ind in scene_list:
            for idx in frame_list:
                noisy_id = 6
                ft1_name = 'ISO{}_scene{}_frame{}_noisy{}.raw'.format(iso_ind, scene_ind, idx,
                                                              noisy_id)
                ft0_name = ft1_name
                content = '{} {} {} {}\n'.format(ft0_dir + ft0_name,
                                                ft1_dir + ft1_name,
                                                coef_dir + 'coeff_a_ISO{}.raw'.format(iso_ind),
                                                coef_dir + 'coeff_b_ISO{}.raw'.format(iso_ind))
                f.write(content)

def device_results():
    raw_shape = (1, 384, 512, 16)
    results_pattern = '/media/wen/09C1B27DA5EB573A1/work/proj/VideoDenoise/EMVDs/' \
                      'vd_v000014/log/log/model_fuLoss_deLoss/results_quantized_dlc/' \
                      'Result_*/'
    fusion_pattern = results_pattern + 'fusion.raw'
    fusion_list = glob.glob(fusion_pattern)
    fusion_list.sort()

    denoise_pattern = results_pattern + 'denoise.raw'
    denoise_list = glob.glob(denoise_pattern)
    denoise_list.sort()

    refine_pattern = results_pattern + 'refine.raw'
    refine_list = glob.glob(refine_pattern)
    refine_list.sort()

    gamma_pattern = results_pattern + 'gamma.raw'
    gamma_list = glob.glob(gamma_pattern)
    gamma_list.sort()

    omega_pattern = results_pattern + 'omega.raw'
    omega_list = glob.glob(omega_pattern)
    omega_list.sort()

    assert len(fusion_list) == len(denoise_list) == len(refine_list) ==len(omega_list) == len(gamma_list)
    for i in range(len(fusion_list)):
        fusion_path = fusion_list[i]
        fusion = np.fromfile(fusion_path, dtype=np.float32)
        fusion = np.reshape(fusion, raw_shape)
        fusion = np.transpose(fusion, (0, 3, 1, 2))

        denoise_path = denoise_list[i]
        denoise = np.fromfile(denoise_path, dtype=np.float32)
        denoise = np.reshape(denoise, raw_shape)
        denoise = np.transpose(denoise, (0, 3, 1, 2))

        refine_path = refine_list[i]
        refine = np.fromfile(refine_path, dtype=np.float32)
        refine = np.reshape(refine, raw_shape)
        refine = np.transpose(refine, (0, 3, 1, 2))

        gamma_path = gamma_list[i]
        gamma = np.fromfile(gamma_path, dtype=np.float32)
        gamma = np.reshape(gamma, (1, 192, 256, 4))
        gamma = np.transpose(gamma, (0, 3, 1, 2))

        omega_path = omega_list[i]
        omega = np.fromfile(omega_path, dtype=np.float32)
        omega = np.reshape(omega, (1, 192, 256, 4))
        omega = np.transpose(omega, (0, 3, 1, 2))

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
        gbrg_to_rgb_dispaly(fusion_path, raw_npy=fusion)

        denoise = denoise[0, 0, :, 4: -4].data.cpu().numpy()
        gbrg_to_rgb_dispaly(denoise_path, raw_npy=denoise)

        refine = refine[0, 0, :, 4: -4].data.cpu().numpy()
        gbrg_to_rgb_dispaly(refine_path, raw_npy=refine)

        cv2.imwrite('{}/omega.png'.format(os.path.dirname(refine_path)),
                    np.uint8(omega[0,0].data.cpu().numpy() * 255))
        cv2.imwrite('{}/gamma.png'.format(os.path.dirname(refine_path)),
                    np.uint8(gamma[0,0].data.cpu().numpy() * 255))


def check_ft0_raw():
    raw_shape = (1, 384, 512, 16)
    ddir = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/' \
           'tangyouyun/mi_k1_pair_data/device_input_after_binning_bhwc'
    dpattern = ddir + '/*ft0.raw'
    ft0_list = glob.glob(dpattern)
    ft0_list.sort()
    for path in ft0_list:
        ft0 = np.fromfile(path, dtype=np.float32)
        ft0 = np.reshape(ft0, raw_shape)
        ft0 = np.transpose(ft0, (0, 3, 1, 2))
        ft0 = torch.from_numpy(ft0)
        for i in range(cfg.px_num - 1):
            ft0 = pixel_shuffle(ft0)
        ft0 = ft0[0, 0, :, 4: -4].data.cpu().numpy()
        gbrg_to_rgb_dispaly(path, raw_npy=ft0)


def main():
    # check_ft0_raw()
    # get_device_data()
    # pc_raw_list()
    # device_raw_list()
    device_results()



if __name__ == '__main__':
    main()
