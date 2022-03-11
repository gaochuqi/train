import collections
from torch.autograd import Variable
import torch.onnx
import torchvision
import torch
import torch.nn as nn
import argparse
import numpy as np
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from arch import architecture
from dataset.dataset import preprocess, tensor2numpy, pack_gbrg_raw, depack_gbrg_raw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_crvd_raw_bhwc():
    crvd_folder = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_rgb/'
    raw_list = glob.glob(crvd_folder+'*_input_data.raw')
    for raw_path in raw_list:
        name = os.path.basename(raw_path).split('.')[0]
        input_data = np.fromfile(raw_path, dtype=np.float32)
        input_data = np.reshape(input_data, (1, 8, 544, 960))
        input_data_bhwc = np.transpose(input_data, (0, 2, 3, 1))
        input_data_bhwc.tofile(crvd_folder + name + '_bhwc.raw')

def show_dlc_results():
    tmp = 'output_dlc_local'
    # tmp = 'results'
    rpath = '/home/wen/Documents/project/video/denoising/emvd/model/%s/'%tmp
    raw_path_list = glob.glob(rpath+'Result_*/*.raw')
    for ipath in raw_path_list:
        cnt = int(ipath.split('/')[-2].split('_')[-1])
        if 'fusion' in ipath:
            fusion = np.fromfile(ipath, dtype=np.float32)
            fusion = np.reshape(fusion,(1,544,960,4))
            # fusion_rgbg = np.concatenate([fusion[:, :, :, 0:1],
            #                     (fusion[:, :, :, 1:2] + fusion[:, :, :, 3:4]) / 2,
            #                     fusion[:, :, :, 2:3]], axis=3)
            # fusion = np.reshape(fusion, (1, 4, 544, 960))
            # fusion_rgbg = np.concatenate([fusion[:, 0:1, :, :],
            #                               (fusion[:, 1:2, :, :] + fusion[:, 3:4, :, :]) / 2,
            #                               fusion[:, 2:3, :, :]], axis=3)
            # fusion_rgbg = np.transpose(fusion_rgbg, (0, 2, 3, 1))
            # cv2.imwrite(rpath+'%d_fusion_raw_rgbg.png'%cnt, np.uint8(fusion_rgbg[0] * 255))
            fusion = np.transpose(fusion, (0, 3, 1, 2))
            fusion = torch.from_numpy(fusion).to(device)
            fusion_frame = tensor2numpy(isp(fusion))[0]
            cv2.imwrite(rpath+'%d_fusion.png'%cnt, np.uint8(fusion_frame * 255))
        elif 'denoise' in ipath:
            denoise = np.fromfile(ipath, dtype=np.float32)
            denoise = np.reshape(denoise,(1,544,960,4))
            # denoise_rgbg = np.concatenate([denoise[:, :, :, 0:1],
            #                               (denoise[:, :, :, 1:2] + denoise[:, :, :, 3:4]) / 2,
            #                               denoise[:, :, :, 2:3]], axis=3)
            # denoise = np.reshape(denoise, (1, 4, 544, 960))
            # denoise_rgbg = np.concatenate([denoise[:, 0:1, :, :],
            #                               (denoise[:, 1:2, :, :] + denoise[:, 3:4, :, :]) / 2,
            #                               denoise[:, 2:3, :, :]], axis=3)
            # denoise_rgbg = np.transpose(denoise_rgbg, (0, 2, 3, 1))
            # cv2.imwrite(rpath + '%d_denoise_raw_rgbg.png' % cnt, np.uint8(denoise_rgbg[0] * 255))
            denoise = np.transpose(denoise, (0, 3, 1, 2))
            denoise = torch.from_numpy(denoise).to(device)
            denoise_frame = tensor2numpy(isp(denoise))[0]
            cv2.imwrite(rpath+'%d_denoise.png'%cnt, np.uint8(denoise_frame * 255))
        elif 'refine' in ipath:
            refine = np.fromfile(ipath, dtype=np.float32)
            refine = np.reshape(refine, (1, 544, 960, 4))
            # refine_rgbg = np.concatenate([refine[:, :, :, 0:1],
            #                               (refine[:, :, :, 1:2] + refine[:, :, :, 3:4]) / 2,
            #                               refine[:, :, :, 2:3]], axis=3)
            # refine = np.reshape(refine, (1, 4, 544, 960))
            # refine_rgbg = np.concatenate([refine[:, 0:1, :, :],
            #                                (refine[:, 1:2, :, :] + refine[:, 3:4, :, :]) / 2,
            #                                refine[:, 2:3, :, :]], axis=3)
            # refine_rgbg = np.transpose(refine_rgbg, (0, 2, 3, 1))
            # cv2.imwrite(rpath + '%d_refine_raw_rgbg.png' % cnt, np.uint8(refine_rgbg[0] * 255))
            refine = np.transpose(refine, (0, 3, 1, 2))
            refine = torch.from_numpy(refine).to(device)
            refine_frame = tensor2numpy(isp(refine))[0]
            cv2.imwrite(rpath+'%d_refine.png'%cnt, np.uint8(refine_frame * 255))

def check_device_results():
    device_test = 'raw_list_crvd_iso1600_scene7.txt'
    data_list_file = '/home/wen/Documents/project/video/denoising/emvd/test/'+device_test
    f = open(data_list_file,'r')
    content = f.readlines()
    f.close()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = '/home/wen/Documents/project/video/denoising/emvd/test/results/'
    raw_path = folder+'Result_*/*.raw'
    raw_path_list = glob.glob(raw_path)
    raw_path_list.sort()
    for ipath in raw_path_list:
        idx = ipath.split('/')[-2]
        idx = int(idx.split('_')[-1])+1
        name = content[idx].split('/')[-1].strip()
        crvd_path = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_rgb/'+name
        input_data = np.fromfile(crvd_path, dtype=np.float32)
        input_data = np.reshape(input_data, (1, 8, 544, 960))
        ft0 = input_data[:,0:4,:,:]
        ft0 = torch.from_numpy(ft0).to(device)
        ft0_frame = tensor2numpy(isp(ft0))[0]
        cv2.imwrite(folder + name +'_ft0.png', np.uint8(ft0_frame * 255))
        ft1 = input_data[:,4:8,:,:]
        ft1 = torch.from_numpy(ft1).to(device)
        ft1_frame = tensor2numpy(isp(ft1))[0]
        cv2.imwrite(folder + name +'_ft1.png', np.uint8(ft1_frame * 255))
        tmp = '_'.join(name.split('_')[0:3])
        if 'fusion' in ipath:
            fusion = np.fromfile(ipath, dtype=np.float32)
            fusion = np.reshape(fusion,(1,4,544,960))
            fusion = torch.from_numpy(fusion).to(device)
            fusion_frame = tensor2numpy(isp(fusion))[0]
            cv2.imwrite(folder+'%3d_'%idx+tmp + '_fusion.png', np.uint8(fusion_frame * 255))
        elif 'denoise' in ipath:
            denoise = np.fromfile(ipath, dtype=np.float32)
            denoise = np.reshape(denoise,(1,4,544,960))
            denoise = torch.from_numpy(denoise).to(device)
            denoise_frame = tensor2numpy(isp(denoise))[0]
            cv2.imwrite(folder +'%3d_'%idx+tmp+ '_denoise.png',np.uint8(denoise_frame * 255))
        elif 'refine' in ipath:
            refine = np.fromfile(ipath, dtype=np.float32)
            refine = np.reshape(refine, (1, 4, 544, 960))
            refine = torch.from_numpy(refine).to(device)
            refine_frame = tensor2numpy(isp(refine))[0]
            cv2.imwrite(folder  +'%3d_'%idx+tmp+  '_refine.png', np.uint8(refine_frame * 255))

    # omega = np.fromfile(raw_path+'omega.raw', dtype=np.float32)
    # gamma = np.fromfile(raw_path+'gamma.raw', dtype=np.float32)
    # fusion.shape = 1,4,544,960
    # denoise.shape = 1,4,544,960
    # refine.shape = 1,4,544,960

def create_raw_list():
    iso_list = [1600, 3200, 6400, 12800, 25600]
    scene_list = range(1,12)
    frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    for iso in iso_list:
        for scene in scene_list:
            f = open('raw_list_crvd_iso%d_scene%d.txt'%(iso,scene),'w')
            for frame in frame_list:
                content = 'ISO%d_scene%d_frame%d_input_data.raw'%(iso,scene,frame)+'\n'
                f.write(content)
            f.close()
    # input_data_list = glob.glob('/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_rgb/*_input_data.raw')
    # input_data_list.sort(reverse = False)
    # for path in input_data_list:
    #     name = os.path.basename(path)
    #     f.write(name+'\n')
    # f.close()

def create_crvd_raw_rgb_bk():
    # Load the pretrained model
    model_qua_arch_path = '/home/wen/Documents/project/video/denoising/emvd/benchmark_output/adaround2021-11-04-21-49-43'
    model = arch_qat.EMVD(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    state_dict = torch.load(os.path.join(model_qua_arch_path, 'adaround_model.pth'))
    model.load_state_dict(state_dict)
    # #########################################################################
    model = model.eval()
    output_dir = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_rgb/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    ft0_fusion = np.zeros([1, 4, 544, 960])
    ft0_fusion_torch = torch.from_numpy(ft0_fusion)
    ft0_fusion_torch = ft0_fusion_torch.cuda()
    iso_average_raw_psnr = 0
    iso_average_raw_ssim = 0
    for iso_ind in range(0, len(iso_list)):
        iso = iso_list[iso_ind]
        scene_avg_raw_psnr = 0
        scene_avg_raw_ssim = 0
        f = open(output_dir + 'denoise_model_test_psnr_and_ssim_on_iso{}.txt'.format(iso), 'w')
        context = 'ISO{}'.format(iso) + '\n'
        f.write(context)
        for scene_id in range(1,11+1):
            context = 'scene{}'.format(scene_id) + '\n'
            f.write(context)
            frame_avg_raw_psnr = 0
            frame_avg_raw_ssim = 0
            for time_ind in range(1, 8):
                output_path = output_dir + 'ISO{}_scene{}_frame{}'.format(iso, scene_id, time_ind)
                # noisy
                raw_noisy_path = os.path.join(cfg.data_root[1],
                                        'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy0.tiff'.format(
                                        scene_id, scene_id, iso, time_ind))
                if not os.path.exists(raw_noisy_path):
                    print(raw_noisy_path)
                raw_noisy = cv2.imread(raw_noisy_path, -1)
                raw_noisy_frame = np.expand_dims(pack_gbrg_raw(raw_noisy), axis=0)
                raw_noisy_frame = np.pad(raw_noisy_frame, [(0,0), (2,2), (0,0), (0,0)], mode='constant')
                raw_noisy_frame = raw_noisy_frame.transpose((0, 3, 1, 2))
                raw_noisy_snpe = raw_noisy_frame.astype('float32')
                raw_noisy_snpe.tofile(output_path+'_noisy.raw')
                raw_noisy_torch = torch.from_numpy(raw_noisy_frame)
                raw_noisy_torch = raw_noisy_torch.cuda()
                srgb_noisy_frame = tensor2numpy(isp(raw_noisy_torch[:, :, 2:-2, :]))[0]
                cv2.imwrite(output_path+'_noisy.png', np.uint8(srgb_noisy_frame * 255))
                # gt
                raw_gt_path = os.path.join(cfg.data_root[1],
                                           'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
                                            scene_id, scene_id, iso, time_ind))
                raw_gt = cv2.imread(raw_gt_path, -1)
                raw_gt_frame = np.expand_dims(pack_gbrg_raw(raw_gt), axis=0)
                # raw_gt_frame = np.pad(raw_gt_frame, [(0,0), (2,2), (0,0), (0,0)], mode='constant')
                raw_gt_frame = raw_gt_frame.transpose((0, 3, 1, 2))
                # raw_gt_snpe = raw_gt_frame.astype('float32')
                # raw_gt_snpe.tofile(output_path + '_gt.raw')
                raw_gt_torch = torch.from_numpy(raw_gt_frame)
                raw_gt_torch = raw_gt_torch.cuda()
                srgb_gt_frame = tensor2numpy(isp(raw_gt_torch))[0]
                cv2.imwrite(output_path + '_gt.png', np.uint8(srgb_gt_frame * 255))
                # input
                if time_ind == 1:
                    ft0_fusion = raw_noisy_torch
                else:
                    ft0_fusion = ft0_fusion_torch
                input_data = torch.cat([ft0_fusion, raw_noisy_torch], dim=1)
                input_data_np = input_data.data.cpu().numpy().astype('float32')
                input_data_np.tofile(output_path + '_input_data.raw')

                fusion_out, denoise_out, refine_out, omega, gamma = model(input_data)
                # fusion
                ft0_fusion_torch = fusion_out

                refine_out_np = refine_out.data.cpu().numpy()
                refine_out_np = refine_out_np[:, :, 2:-2, :]
                # refine_out_np.tofile(output_path + '_refine.raw')
                srgb_refine_frame = tensor2numpy(isp(refine_out[:, :, 2:-2, :]))[0]
                cv2.imwrite(output_path+'_refine.png', np.uint8(srgb_refine_frame * 255))

                psnr = compare_psnr(raw_gt_frame, refine_out_np, data_range=1.0)
                # ssim = compute_ssim_for_packed_raw(raw_gt_frame, refine_out_np)
                test_raw_ssim = 0
                for i in range(4):
                    test_raw_ssim += compare_ssim(raw_gt_frame[0, i, :, :], refine_out_np[0, i,:, :], data_range=1.0)
                ssim = test_raw_ssim / 4
                frame_avg_raw_psnr += psnr
                frame_avg_raw_ssim += ssim
                context = 'frame {} , psnr/ssim: {}/{}'.format(time_ind, psnr, ssim) + '\n'
                f.write(context)
            frame_avg_raw_psnr = frame_avg_raw_psnr / 7
            frame_avg_raw_ssim = frame_avg_raw_ssim / 7
            scene_avg_raw_psnr += frame_avg_raw_psnr
            scene_avg_raw_ssim += frame_avg_raw_ssim
            context = 'frame average , psnr/ssim: {}/{}'.format(frame_avg_raw_psnr, frame_avg_raw_ssim) + '\n'
            f.write(context)
        scene_avg_raw_psnr = scene_avg_raw_psnr / 11
        scene_avg_raw_ssim = scene_avg_raw_ssim / 11
        context = 'scene average , psnr/ssim: {}/{}'.format(scene_avg_raw_psnr, scene_avg_raw_ssim) + '\n'
        f.write(context)
        iso_average_raw_psnr += scene_avg_raw_psnr
        iso_average_raw_ssim += scene_avg_raw_ssim
    iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
    iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)
    context = 'iso average , psnr/ssim: {}/{}'.format(iso_average_raw_psnr, iso_average_raw_ssim) + '\n'
    f.write(context)
    f.close()

from arch import architecture_reparam
def create_crvd_raw():
    # Load the pretrained model
    log_dir = 'log_bk/models' # 'log/model/yushu'
    model_path = os.path.join(log_dir, 'model_qua_arch.pth')
    model = architecture_reparam.EMVD(cfg)
    model = model.to(device)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    #########################################################################
    subfolder = log_dir # '/home/wen/Documents/project/video/denoising/emvd_bin/log/model/yushu'
    bin_np = np.fromfile('%s/bin.raw' % (subfolder), dtype=np.float32)
    bin_np = np.reshape(bin_np, (16, 64, 1, 1))
    bin_w = torch.from_numpy(bin_np).to(device)
    conv_bin = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False).to(device)
    conv_bin.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    #########################################################################
    pixel_shuffle = architecture.PixelShuffle(2)
    pixel_unshuffle = architecture.PixelShuffle(0.5)
    #########################################################################
    model = model.eval()
    output_dir = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_raw_list/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    iso_list = [1600, 3200, 6400, 12800, 25600]
    scene_list = ['1','2','3','4','5','6', '7', '8', '9', '10', '11']
    frame_list = [1, 2, 3, 4, 5, 6, 7]

    f = open(output_dir + '/raw_list.txt', 'w')
    f.write('#Conv_216 Conv_218 Conv_220 Sigmoid_208 Sigmoid_180\n')
    ft0 = np.zeros((1, 272, 480, 16), dtype=np.float32)
    for scene_ind in scene_list:
        for noisy_level_ind, iso_ind in enumerate(iso_list):
            for idx in range(1, len(frame_list) + 1):
                frame_ind = frame_list[idx - 1]
                name = 'ISO{}_scene{}_frame{}'.format(iso_ind, scene_ind, str(idx))
                # noisy_frame_index_for_current = np.random.randint(0, 10)
                input_pack_list = []
                noisy_frame_index_for_current = np.random.choice(10, 4, replace=False)
                for i in noisy_frame_index_for_current:
                    noisy_name = os.path.join(cfg.data_root[1],
                                              'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(
                                                  scene_ind, scene_ind, iso_ind, frame_ind, i))
                    raw_noisy = cv2.imread(noisy_name, -1)
                    raw_noisy_frame = np.expand_dims(pack_gbrg_raw(raw_noisy), axis=0)
                    raw_noisy_frame = np.pad(raw_noisy_frame, [(0, 0), (4, 4), (0, 0), (0, 0)], mode='constant')
                    raw_noisy_frame = raw_noisy_frame.astype('float32')
                    input_pack_list.append(raw_noisy_frame)
                input_pack = np.concatenate(input_pack_list, axis=-1)
                raw_noisy_frame = input_pack.transpose((0, 3, 1, 2))
                noisy = torch.from_numpy(raw_noisy_frame).to(device)
                ft1 = noisy
                for i in range(cfg.px_num):
                    ft1 = pixel_unshuffle(ft1)
                if idx == 1:
                    ft0 = conv_bin(ft1)
                ft0.permute(0, 2, 3, 1).data.cpu().numpy().tofile('%s/%s_ft0.raw' % (output_dir,name))
                ft1 = conv_bin(ft1)
                ft1.permute(0, 2, 3, 1).data.cpu().numpy().tofile('%s/%s_ft1.raw' % (output_dir,name))
                coeff_a = torch.tensor(a_list[noisy_level_ind] / (2 ** 12 - 1 - 240))
                coeff_b = torch.tensor(b_list[noisy_level_ind] / (2 ** 12 - 1 - 240) ** 2)
                coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1]).to(device)
                coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1]).to(device)
                fusion, denoise, refine, omega, gamma = model(ft0, ft1, coeff_a, coeff_b)
                ft0 = fusion
                f.write('%s/%s_ft0.raw '
                        '%s/%s_ft1.raw '
                        '%s/fake_ab/coeff_a_%d.raw '
                        '%s/fake_ab/coeff_b_%d.raw' % (output_dir,name,
                                                       output_dir,name,
                                                       output_dir,iso_ind,
                                                       output_dir,iso_ind) + '\n')
    f.close()

def test_input():
    img_filepath = '/home/wen/env/snpe-1.54.2.2899/models/VGG/data/kitten.jpg'
    img = Image.open(img_filepath)
    src_img = img.resize((1920, 1080),resample=Image.BILINEAR)
    src_img = src_img.convert(mode='RGBA')
    img_ndarray = np.array(src_img) # read it
    img_raw = (img_ndarray - img_ndarray.min())/(img_ndarray.max()-img_ndarray.min())
    img_raw = np.transpose(img_raw, (2, 0, 1))
    img_raw = np.expand_dims(img_raw,0)
    snpe_raw = img_raw.astype('float32')
    snpe_raw.tofile('./model/test.raw')

def test_result():
    snpe_raw = '/home/wen/Documents/project/video/denoising/emvd/model/Result_0/output.raw'
    float_array = np.fromfile(snpe_raw, dtype=np.float32)
    img_raw = np.reshape(float_array,(1,4,128,128))
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(img_raw[0,i])
    print('end')

# def test_convert():
#     input = Variable(torch.randn(1, 4, 128, 128))
#     input = input.cuda()
#     model = structure.test()
#     if torch.cuda.is_available():
#         model.cuda()
#     torch.onnx.export(model,
#                       (input),
#                       "./model/test.onnx",
#                       opset_version=11,
#                       do_constant_folding=True,
#                       input_names=['input'],
#                       output_names=['output'])
#     print('end')


import config as cfg

subfolder = cfg.model_name

def prepare_input(h, w):
    device_dir = '/data/local/tmp/wen/test'
    # f = open('./%s/raw_list_%d_2.txt'%(subfolder,w), 'a+')
    f = open('./%s/raw_list_%d.txt'%(subfolder,w), 'a+')
    for i in range(100):
        # f.write('%s/test_%d_0.raw %s/test_%d_1.raw %s/coeff_a.raw %s/coeff_b.raw\n'%(device_dir,w,device_dir,w, device_dir,device_dir))
        f.write('%s/test_%d.raw %s/coeff_a.raw %s/coeff_b.raw\n' % (device_dir, w, device_dir, device_dir))
    f.close()
    data_dir = '/home/wen/Documents/project/video/denoising/emvd/data'
    # img_raw = np.random.rand(1, 4, h, w)
    # snpe_raw = img_raw.astype('float32')
    # snpe_raw.tofile('%s/test_%d_0.raw' % (data_dir, w))
    img_raw = np.random.rand(1, 8, h, w)
    snpe_raw = img_raw.astype('float32')
    snpe_raw.tofile('%s/test_%d.raw' % (data_dir, w))
    # f = open('%s/raw_list_%d_2.txt' % (data_dir,w), 'a+')
    f = open('%s/raw_list_%d.txt' % (data_dir,w), 'a+')
    for i in range(100):
        # f.write('%s/test_%d_0.raw %s/test_%d_1.raw %s/coeff_a.raw %s/coeff_b.raw\n' % (data_dir, w,data_dir,w, data_dir, data_dir))
        f.write('%s/test_%d.raw %s/coeff_a.raw %s/coeff_b.raw\n' % (data_dir, w, data_dir, data_dir))
    f.close()

def convert_onnx(model, h, w, others):
    ##########################################################
    suffix = ''
    ##########################################################
    coeff_a = others[0]
    coeff_b = others[1]

    input = Variable(torch.randn(1, 8, h, w))
    input = input.cuda()

    # input0 = Variable(torch.randn(1, 4, h, w))
    # input0 = input0.cuda()
    # input1 = Variable(torch.randn(1, 4, h, w))
    # input1 = input1.cuda()

    inputs = {# "input0": input0,
              # "input1": input1,
              "input": input,
              "coeff_a": coeff_a,
              "coeff_b": coeff_b,
              }
    torch.onnx.export(model,
                      (# inputs["input0"],
                       # inputs["input1"],
                       inputs["input"],
                       inputs["coeff_a"],
                       inputs["coeff_b"]),
                      "./%s/model.onnx" % (subfolder),
                      # "./%s/model_%d_2.onnx"%(subfolder,w),
                      # "./%s/model_%d%s.onnx"%(subfolder,w,suffix),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=[# 'input0',
                                   # 'input1',
                                   'input',
                                   'coeff_a',
                                   'coeff_b'],
                      output_names=['gamma',
                                    'fusion',
                                    'denoise',
                                    'omega',
                                    'refine']
                      )

def load_weight():
    checkpoint = torch.load('./%s/model.pth' % (subfolder))
    # checkpoint = torch.load('./model/model.pth')
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

def convert_test():
    test = 1
    shape = [[1920,1080],
             [960,544],
             [512,512],
             [256,256],
             [128,128]]
    h = shape[test][0]
    w = shape[test][1]

    ## initialize model
    model = arch.test()
    if torch.cuda.is_available():
        model.cuda()

    input = Variable(torch.randn(1, 4, h, w))
    input = input.cuda()
    inputs = {"input": input,}
    torch.onnx.export(model,
                      (inputs["input"],),
                      "./test/model.onnx",
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input',],
                      output_names=['output']
                      )
    # for i in range(len(shape)):
    #     h = shape[i][0]
    #     w = shape[i][1]
    #     cfg.image_height = h
    #     cfg.image_width = w
    #     ## initialize model
    #     model = structure.MainDenoise(cfg)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     state_dict = load_weight()
    #     model.load_state_dict(state_dict, strict=False)
    #     convert_onnx(model, h, w, others)
    # 1080 #####################################################
    # img_raw = np.random.rand(1, 8, 1920, 1080)
    # snpe_raw = img_raw.astype('float32')
    # snpe_raw.tofile('./model_1i/test_1080.raw')
    # input_1080 = Variable(torch.randn(1, 8, 1920, 1080))
    # input_1080 = input_1080.cuda()
    # inputs = {"input": input_1080,
    #           "coeff_a": coeff_a,
    #           "coeff_b": coeff_b,
    #           }
    # torch.onnx.export(model,
    #                   (inputs["input"],
    #                    inputs["coeff_a"],
    #                    inputs["coeff_b"]),
    #                   "./model_1i/model_1080.onnx",
    #                   opset_version=11,
    #                   do_constant_folding=True,
    #                   input_names=['input',
    #                                'coeff_a',
    #                                'coeff_b'],
    #                   output_names=['gamma',
    #                                 'fusion',
    #                                 'denoise',
    #                                 'omega',
    #                                 'refine']
    #                   )

    # 128 ######################################################
    # img_raw = np.random.rand(1, 8, 128, 128)
    # snpe_raw = img_raw.astype('float32')
    # snpe_raw.tofile('./model_1i/test_128.raw')
    # input_128 = Variable(torch.randn(1, 8, 128, 128))
    # input_128 = input_128.cuda()
    # inputs = {"input": input_128,
    #           "coeff_a": coeff_a,
    #           "coeff_b": coeff_b,
    #           }
    # torch.onnx.export(model,
    #                   (inputs["input"],
    #                    inputs["coeff_a"],
    #                    inputs["coeff_b"]),
    #                   "./model_1i/model_128.onnx",
    #                   opset_version=11,
    #                   do_constant_folding=True,
    #                   input_names=['input',
    #                                'coeff_a',
    #                                'coeff_b'],
    #                   output_names=['gamma',
    #                                 'fusion',
    #                                 'denoise',
    #                                 'omega',
    #                                 'refine']
    #                   )

    # 256 ######################################################

    print('end')


def convert():
    h = 544
    w = 960
    cfg.image_height = h
    cfg.image_width = w
    cfg.use_realism = False
    if cfg.use_realism:
        sub_folder = 'model_photo_real'
        model_name = 'model_real_best'
    else:
        sub_folder = 'results/a100/model_basic'
        model_name = 'model_best'

    model_qua_arch_path = '/home/wen/Documents/project/video/denoising/emvd/benchmark_output/adaround2021-11-04-21-49-43'
    # model = arch_qat.EMVD(cfg)
    # model_path = './model/model.pth'#'./test/model_best.pth'
    model_path = './%s/%s.pth'%(sub_folder, model_name)
    model = arch.EMVD(cfg)

    output_names = ['fusion',
                    'denoise',
                    'refine',
                    'omega',
                    'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])

    if torch.cuda.is_available():
        model.cuda()

    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model']
    # state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)

    data = Variable(torch.randn(1, 8, h, w))
    data = data.cuda()

    inputs = {"data": data,}
    torch.onnx.export(model,
                      (inputs["data"],),
                      "./%s/%s.onnx" % (sub_folder,model_name),
                      opset_version=11,
                      do_constant_folding=False,
                      input_names=['data',],
                      output_names=output_names
                      )


def convert_arch(cfg, subfolder, model_name):
    h = cfg.height
    w = cfg.width
    # Load model
    best_model_save_root = os.path.join(subfolder, '%s.pth'%model_name)
    print(best_model_save_root)
    checkpoint = torch.load(best_model_save_root)
    if 'iter' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model = arch_qat.EMVD(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)
    #########################################################################
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
    filters_ft = torch.cat(filters_g1, dim=0)  # .to('cuda')
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
    filters_fti = torch.cat(filters_g2, dim=0)  # .to('cuda')
    model.fti.net.weight = nn.Parameter(filters_fti)
    #########################################################################
    model.ct0.net.weight = model.ct.net.weight
    model.ct1.net.weight = model.ct.net.weight

    model.cti_fu.net.weight = model.cti.net.weight
    model.cti_de.net.weight = model.cti.net.weight
    model.cti_re.net.weight = model.cti.net.weight

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
    if cfg.use_realism:
        model.fti_realism.net.weight = model.fti.net.weight
        model.cti_realism.net.weight = model.cti.net.weight
    #########################################################################
    # model.eval()
    torch.save(model.state_dict(), "./%s/%s_qua_arch.pth" % (subfolder, model_name))
    input = Variable(torch.randn(1, 8, h, w))
    input = input.cuda()
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])
    inputs = {
        "input": input,
    }
    torch.onnx.export(model,
                      (inputs["input"],),
                      "./%s/%s_qua_arch.onnx" % (subfolder, model_name),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input',],
                      output_names=output_names
                      )

def check_onnx(subfolder, model_name):
    isp = torch.load('isp/ISP_CNN.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import onnx
    onnx_model = onnx.load("./%s/%s.onnx" % (subfolder, model_name))
    onnx.checker.check_model(onnx_model)
    # input
    folder = './%s/'% (subfolder)
    name = 'ISO1600_scene7_frame1'
    crvd_path = '/home/wen/Documents/dataset/denoising/video/CRVD_dataset/indoor_rgb/' + name + '_input_data.raw'
    input_data = np.fromfile(crvd_path, dtype=np.float32)
    input_data = np.reshape(input_data, (1, 8, 544, 960))
    ft0 = input_data[:,0:4,:,:]
    ft0 = torch.from_numpy(ft0).to(device)
    ft0_frame = tensor2numpy(isp(ft0))[0]
    cv2.imwrite(folder + name +'_ft0.png', np.uint8(ft0_frame * 255))
    ft1 = input_data[:,4:8,:,:]
    ft1 = torch.from_numpy(ft1).to(device)
    ft1_frame = tensor2numpy(isp(ft1))[0]
    cv2.imwrite(folder + name +'_ft1.png', np.uint8(ft1_frame * 255))
    input_data_torch = torch.from_numpy(input_data).to(device)
    # torch model
    # Load the pretrained model
    if cfg.use_arch_qat:
        model = arch_qat.EMVD(cfg)
    else:
        model = arch.EMVD(cfg)
    model = model.to(device)
    state_dict = torch.load("./%s/%s.pth" % (subfolder, model_name))
    if 'iter' in state_dict.keys():
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=True)
    # model.eval()
    if cfg.use_realism:
        fusion, denoise, refine, omega, gamma, real = model(input_data_torch)
        real_frame = tensor2numpy(isp(real))[0]
        cv2.imwrite(folder + name + '_real.png', np.uint8(real_frame * 255))
    else:
        fusion, denoise, refine, omega, gamma = model(input_data_torch)
    fusion_frame = tensor2numpy(isp(fusion))[0]
    cv2.imwrite(folder + name +'_fusion.png', np.uint8(fusion_frame * 255))
    denoise_frame = tensor2numpy(isp(denoise))[0]
    cv2.imwrite(folder + name + '_denoise.png', np.uint8(denoise_frame * 255))
    refine_frame = tensor2numpy(isp(refine))[0]
    cv2.imwrite(folder + name + '_refine.png', np.uint8(refine_frame * 255))
    # ONNX
    import onnxruntime
    ort_session = onnxruntime.InferenceSession("./%s/%s.onnx" % (subfolder, model_name))
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_data_torch)}
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])
        fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx, real_onnx = ort_session.run(output_names, ort_inputs)
        np.testing.assert_allclose(to_numpy(real), real_onnx, rtol=1e-03, atol=1e-05)
        real_onnx_frame = tensor2numpy(isp(torch.from_numpy(real_onnx).to(device)))[0]
        cv2.imwrite(folder + name + '_real_onnx.png', np.uint8(real_onnx_frame * 255))
    else:
        fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx = ort_session.run(output_names, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(refine), refine_onnx, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    fusion_onnx_frame = tensor2numpy(isp(torch.from_numpy(fusion_onnx).to(device)))[0]
    cv2.imwrite(folder + name + '_fusion_onnx.png', np.uint8(fusion_onnx_frame * 255))
    denoise_onnx_frame = tensor2numpy(isp(torch.from_numpy(denoise_onnx).to(device)))[0]
    cv2.imwrite(folder + name + '_denoise_onnx.png', np.uint8(denoise_onnx_frame * 255))
    refine_onnx_frame = tensor2numpy(isp(torch.from_numpy(refine_onnx).to(device)))[0]
    cv2.imwrite(folder + name + '_refine_onnx.png', np.uint8(refine_onnx_frame * 255))

def check_device():
    subfolder = 'log/models'
    model_name = 'model'
    path1 = os.path.join(subfolder, 'model.pth')
    ckpt1 = torch.load(path1)
    state_dict = ckpt1['model']
    model = architecture.EMVD(cfg)
    model = model.to(cfg.device)
    model.load_state_dict(state_dict, strict=True)

    model_reparam = architecture.test()
    model_reparam = model_reparam.to(cfg.device)

    c = 4 ** cfg.px_num * 4
    n = 4
    g = c // n
    filters = torch.zeros((g, c, 1, 1), device=model.binnings.groupn.device)
    for i in range(g):
        for j in range(n):
            filters[i, i + j * g, :, :] = model.binnings.groupn[i, j, :, :]
    model_binnings_conv_bin_weight = nn.Parameter(filters)
    model_reparam.conv1.weight.data = model_binnings_conv_bin_weight.data

    output_names = ['fusion',
                    'denoise',
                    'refine',
                    'omega',
                    'gamma']

    idx=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coeff_a = a_list[idx] / (2 ** 12 - 1 - 240)
    coeff_a = torch.tensor(coeff_a).float().to(device)
    coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1])
    coeff_b = b_list[idx] / (2 ** 12 - 1 - 240) ** 2
    coeff_b = torch.tensor(coeff_b).float().to(device)
    coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1])
    torch.save(model_reparam.state_dict(), "./%s/test/%s_qua_arch.pth" % (subfolder, model_name))
    h = cfg.height
    w = cfg.width
    n = cfg.px_num
    c = 4 ** n
    ft0 = Variable(torch.randn(1, c, h // (2 ** n), w // (2 ** n)))
    ft0 = ft0.cuda()
    ft1 = Variable(torch.randn(1, c * 4, h // (2 ** n), w // (2 ** n)))
    ft1 = ft1.cuda()

    torch.onnx.export(model_reparam,
                      (ft0,ft1,coeff_a,coeff_b),
                      "./%s/test/%s_qua_arch.onnx" % (subfolder, model_name),
                      opset_version=11,
                      do_constant_folding=False,
                      input_names=['ft0','ft1','coeff_a','coeff_b'],
                      output_names=output_names
                      )

def main():
    create_crvd_raw()
    check_device()
    convert = True
    cfg.use_arch_qat = True
    cfg.use_realism = False
    if cfg.use_realism:
        sub_folder = 'model_photo_real'
        model_name = 'model_real_best'
    else:
        sub_folder = 'model'#'results/a100/model_basic'
        model_name = 'model_best'

    if convert:
        if cfg.use_arch_qat:
            convert_arch(cfg, sub_folder, model_name)
            model_name += '_qua_arch'
        check_onnx(sub_folder, model_name)
        return
    # show_dlc_results()
    # create_crvd_raw_bhwc()
    # create_raw_list()
    # check_device_results()
    # create_crvd_raw_rgb()
    convert()
    # convert_test()
    # test_input()
    # test_convert()
    # test_result()

if __name__ == '__main__':
    main()