import glob

import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import torch.nn as nn
from torch.autograd import Variable
import onnx
import onnxruntime

import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

import cv2
import warnings

warnings.filterwarnings('ignore')
from torchstat import stat

import utils
from dataset import *
import config as cfg

from arch import architecture
from arch import architecture_qat
from arch import architecture_reparam
from dataset.dataset import preprocess, tensor2numpy, pack_gbrg_raw
from arch.modules import PixelShuffle
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


def run_model(model, ft1, coeff_a, coeff_b, conv_bin=None):
    ########################################################
    # run model
    ########################################################
    for i in range(cfg.px_num):
        ft1 = pixel_unshuffle(ft1)

    if conv_bin is not None:
        ft1 = conv_bin(ft1)

    ft0 = ft1

    fusion, denoise, refine, omega, gamma = model(ft0, ft1, coeff_a, coeff_b)
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

    fusion_np = fusion.data.cpu().numpy()  # h,w
    denoise_np = denoise.data.cpu().numpy()
    refine_np = refine.data.cpu().numpy()
    return refine_np


def check_model(name):
    device = cfg.device
    log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    # Load model
    ckpt1 = torch.load(os.path.join(log_dir, '{}.pth'.format(name)))
    state_dict1 = ckpt1['model']
    model1 = architecture.EMVD(cfg)
    model1 = model1.to(cfg.device)
    model1.load_state_dict(state_dict1, strict=True)

    state_dict2 = torch.load(os.path.join(log_dir, '{}_reparam.pth'.format(name)))
    model2 = architecture_reparam.EMVD(cfg)
    model2 = model2.to(cfg.device)
    model2.load_state_dict(state_dict2, strict=True)
    ########################################################
    # noisy
    ########################################################
    iso_ind = 3200
    noisy_level_ind = 0
    scene_ind = 1
    frame_id = 2
    noisy_frame_index_for_current = np.random.randint(0, 10)
    noisy_name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_noisy{}.npy'.format(scene_ind, iso_ind,
                                                                          iso_ind, scene_ind,
                                                                          frame_id,
                                                                          noisy_frame_index_for_current)
    input_name = os.path.join(data_dir, noisy_name)
    raw_noisy = np.load(input_name, mmap_mode='r')
    tmp = np.pad(raw_noisy, [(0, 0), (8, 8)])
    noisy = torch.from_numpy(np.expand_dims(np.expand_dims(tmp,
                                                           axis=0), axis=0)).cuda()
    ft1 = noisy
    coeff_a = a_list[noisy_level_ind]  # / (cfg.white_level - cfg.black_level)
    coeff_b = b_list[noisy_level_ind]  # / (cfg.white_level - cfg.black_level) ** 2
    print(coeff_a)
    print(coeff_b)
    print(coeff_a / (cfg.white_level - cfg.black_level))
    print(coeff_b / (cfg.white_level - cfg.black_level) ** 2)
    bin_np = np.load('{}/binning.npy'.format(log_dir)).astype(np.float32)
    bin_w = torch.from_numpy(bin_np)
    binnings = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
    binnings.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    conv_bin = binnings.to(device)

    refine1 = run_model(model1, ft1, coeff_a, coeff_b, conv_bin=None)
    refine2 = run_model(model2, ft1, coeff_a, coeff_b, conv_bin=conv_bin)
    diff = np.where(refine1 != refine2)
    if len(diff[0]) == 0:
        print('check reparam, same result.')
        gbrg_to_rgb_dispaly(os.path.join(log_dir, '{}_reparam.pth'.format(name)), refine2)
    else:
        print('check reparam, different:', diff)


def reparameters(name):
    #########################################################################
    # Load Float model
    #########################################################################
    # Load the pretrained model
    model = architecture.EMVD(cfg)
    device = cfg.device
    model = model.to(device)
    log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    ckpt = torch.load(os.path.join(log_dir, '{}.pth'.format(name)))
    state_dict = ckpt['model']
    model.load_state_dict(state_dict, strict=True)
    # model = model.eval()
    #########################################################################
    # Load Re-parameters model
    #########################################################################
    model_reparam = architecture_reparam.EMVD(cfg)
    model_reparam = model_reparam.to(device)
    model_reparam.load_state_dict(state_dict, strict=False)
    #########################################################################
    # binnings
    #########################################################################
    c = 4 ** cfg.px_num
    n = 4
    g = c // n
    filters = torch.zeros((g, c, 1, 1), device=model.binnings.groupn.device)
    for i in range(g):
        for j in range(n):
            filters[i, i // 4 * g + j * n + i % 4, :, :] = model.binnings.groupn[i, j, :, :]
    model_binnings_conv_bin_weight = nn.Parameter(filters, requires_grad=True)
    # model_reparam.binnings.conv_bin.weight = model_binnings_conv_bin_weight
    np.save('{}/binning.npy'.format(log_dir), model_binnings_conv_bin_weight.data.cpu().numpy())
    #########################################################################
    # ct
    #########################################################################
    n = 4 ** (cfg.px_num - 1)
    cfan = torch.zeros((n, n, 1, 1), device=model.ct.w.device)
    c = 4  # n // 4
    for i in range(4):
        for j in range(c):
            cfan[i * 4 + j, j, :, :] = model.ct.w[i, 0]
            cfan[i * 4 + j, j + c, :, :] = model.ct.w[i, 1]
            cfan[i * 4 + j, j + c * 2, :, :] = model.ct.w[i, 2]
            cfan[i * 4 + j, j + c * 3, :, :] = model.ct.w[i, 3]
    model_ct_net_weight = nn.Parameter(cfan, requires_grad=True)
    #########################################################################
    # cti
    #########################################################################
    n = 4 ** (cfg.px_num - 1)
    cfan_inv = torch.zeros((n, n, 1, 1), device=model.ct.w.device)
    c = 4  # n // 4
    for i in range(4):
        for j in range(c):
            cfan_inv[i * 4 + j, j, :, :] = model.cti.w[i, 0]
            cfan_inv[i * 4 + j, j + c, :, :] = model.cti.w[i, 1]
            cfan_inv[i * 4 + j, j + c * 2, :, :] = model.cti.w[i, 2]
            cfan_inv[i * 4 + j, j + c * 3, :, :] = model.cti.w[i, 3]
    model_cti_net_weight = nn.Parameter(cfan_inv, requires_grad=True)
    #########################################################################
    # ft
    #########################################################################
    h0_row = model.ft.w1
    h1_row = model.ft.w2
    h0_row_t = model.ft.w1.transpose(2, 3)
    h1_row_t = model.ft.w2.transpose(2, 3)
    h00_row = h0_row * h0_row_t
    h01_row = h0_row * h1_row_t
    h10_row = h1_row * h0_row_t
    h11_row = h1_row * h1_row_t
    filters1 = [h00_row, h01_row, h10_row, h11_row]
    n = 4 ** (cfg.px_num - 1)
    filters_ft = torch.zeros((n * 4, n, 2, 2), device=h00_row.device)
    for i in range(4):
        for j in range(n):
            filters_ft[n * i + j, j, :, :] = filters1[i][0, 0, :, :]
    model_ft_net_weight = nn.Parameter(filters_ft, requires_grad=True)
    #########################################################################
    # fti
    #########################################################################
    g0_col = model.fti.w1
    g1_col = model.fti.w2
    g0_col_t = model.fti.w1.transpose(2, 3)
    g1_col_t = model.fti.w2.transpose(2, 3)
    g00_col = g0_col * g0_col_t
    g01_col = g0_col * g1_col_t
    g10_col = g1_col * g0_col_t
    g11_col = g1_col * g1_col_t
    filters2 = [g00_col, g10_col, g01_col, g11_col]
    n = 4 ** (cfg.px_num - 1)
    filters_fti = torch.zeros((n * 4, n, 2, 2), device=g00_col.device)
    for i in range(4):
        for j in range(n):
            filters_fti[n * i + j, j, :, :] = filters2[i][0, 0, :, :]
    model_fti_net_weight = nn.Parameter(filters_fti, requires_grad=True)
    # #########################################################################
    model_reparam.ct_0.net.weight.data = model_ct_net_weight.data.clone()  # model.ct.net.weight.data
    model_reparam.ct_1.net.weight.data = model_ct_net_weight.data.clone()  # model.ct.net.weight.data

    model_reparam.cti_fu.net.weight.data = model_cti_net_weight.data.clone()  # model.cti.net.weight.data
    model_reparam.cti_de.net.weight.data = model_cti_net_weight.data.clone()  # model.cti.net.weight.data
    model_reparam.cti_re.net.weight.data = model_cti_net_weight.data.clone()  # model.cti.net.weight.data

    model_reparam.ft_00.net.weight.data = model_ft_net_weight.data.clone()  # model.ft.net.weight.data
    model_reparam.ft_10.net.weight.data = model_ft_net_weight.data.clone()  # model.ft.net.weight.data
    model_reparam.ft_01.net.weight.data = model_ft_net_weight.data.clone()  # model.ft.net.weight.data
    model_reparam.ft_11.net.weight.data = model_ft_net_weight.data.clone()  # model.ft.net.weight.data
    model_reparam.ft_02.net.weight.data = model_ft_net_weight.data.clone()  # model.ft.net.weight.data
    model_reparam.ft_12.net.weight.data = model_ft_net_weight.data.clone()  # model.ft.net.weight.data

    model_reparam.fti_d2.net.weight.data = model_fti_net_weight.data.clone()  # model.fti.net.weight.data
    model_reparam.fti_d1.net.weight.data = model_fti_net_weight.data.clone()  # model.fti.net.weight.data
    model_reparam.fti_fu.net.weight.data = model_fti_net_weight.data.clone()  # model.fti.net.weight.data
    model_reparam.fti_de.net.weight.data = model_fti_net_weight.data.clone()  # model.fti.net.weight.data
    model_reparam.fti_re.net.weight.data = model_fti_net_weight.data.clone()  # model.fti.net.weight.data
    #########################################################################

    if cfg.use_ecb:
        depth = len(model.ecb)
        for d in range(depth):
            module = model.ecb[d]
            act_type = module.act_type
            RK, RB = module.rep_params()
            model_reparam.eocb[d].conv.weight.data = RK
            model_reparam.eocb[d].conv.bias.data = RB

            if act_type == 'relu':
                pass
            elif act_type == 'linear':
                pass
            elif act_type == 'prelu':
                model_reparam.ecb[d].act.weight.data = module.act.weight.data.clone()
            else:
                raise ValueError('invalid type of activation!')
    #########################################################################
    torch.save(model_reparam.state_dict(), "{}/{}_reparam.pth".format(log_dir, name))


def check_onnx(subfolder, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    onnx_model = onnx.load("./%s/%s.onnx" % (subfolder, model_name))
    onnx.checker.check_model(onnx_model)
    # torch model
    # Load the pretrained model
    model = architecture_reparam.EMVD(cfg)  # architecture_qat.EMVD(cfg)
    model = model.to(device)
    path = "./%s/%s.pth" % (subfolder, model_name)
    state_dict = torch.load(path)
    if 'iter' in state_dict.keys():
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    # model.eval()
    ###################################
    # model_check = architecture.EMVD(cfg)
    ###################################

    # input
    path_raw = '/home/wen/Documents/project/video/denoising/emvd_bin/data/'
    names = ['ISO1600_scene7_frame1', 'ISO1600_scene8_frame3', 'ISO1600_scene9_frame1',
             'ISO3200_scene9_frame2', 'ISO3200_scene10_frame1', 'ISO3200_scene11_frame4',
             'ISO6400_scene6_frame4', 'ISO1600_scene5_frame3', 'ISO1600_scene4_frame6',
             'ISO12800_scene3_frame7', 'ISO12800_scene2_frame5', 'ISO12800_scene5_frame1',
             'ISO25600_scene4_frame3', 'ISO25600_scene8_frame4', 'ISO25600_scene11_frame5', ]
    # f = open('raw_list_crvd_local_bhwc_bin.txt', 'w')
    for name in names:
        iso = int(name.split('_')[0][3:])  # 1600
        scene_ind = int(name.split('_')[1][5:])  # 7
        frame_ind = int(name.split('_')[2][5:])  # 1
        idx = cfg.iso_list.index(iso)
        coeff_a = torch.tensor(cfg.a_list[idx] / (2 ** 12 - 1 - 240)).float().to(device)
        coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1])
        coeff_b = torch.tensor(cfg.b_list[idx] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
        coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1])

        folder = './%s/' % (subfolder)
        input_pack, img = get_input(iso, scene_ind, frame_ind)
        # noisy8 = np.concatenate([input_pack,input_pack],axis=3)
        # noisy8.tofile('%s/%s_noisy8_bhwc.raw' % (subfolder, name))
        noisy4 = np.transpose(input_pack, (0, 3, 1, 2))
        img = torch.from_numpy(img).to(device)
        img_frame = tensor2numpy(isp(img))[0]
        cv2.imwrite(folder + name + '_noisy.png', np.uint8(img_frame * 255))
        # input_data = np.concatenate([noisy4,noisy4],axis=1)
        ft0_torch = torch.from_numpy(noisy4).to(device)
        ft1_torch = torch.from_numpy(noisy4).to(device)
        for i in range(cfg.px_num):
            ft0_torch = pixel_unshuffle(ft0_torch)
            ft1_torch = pixel_unshuffle(ft1_torch)
        ft0_torch = conv_bin(ft0_torch)
        ft1_torch = conv_bin(ft1_torch)
        ft0_torch.permute(0, 2, 3, 1).data.cpu().numpy().tofile('%s/%s_ft0_bin_bhwc.raw' % (subfolder, name))
        ft1_torch.permute(0, 2, 3, 1).data.cpu().numpy().tofile('%s/%s_ft1_bin_bhwc.raw' % (subfolder, name))
        # path_ft0 = path_raw + '%s_ft0_bin_bhwc.raw' % (name)
        # path_ft1 = path_raw + '%s_ft1_bin_bhwc.raw' % (name)
        # content = '%s %s %s %s\n'%(path_ft0,path_ft1,path_coeff_a,path_coeff_b)
        # f.write(content)
        if cfg.use_realism:
            fusion, denoise, refine, omega, gamma, real = model(ft0_torch, ft1_torch, coeff_a, coeff_b)
            for i in range(cfg.px_num - 1 - 1):
                real = pixel_shuffle(real)
            real_frame = tensor2numpy(isp(real))[0]
            cv2.imwrite(folder + name + '_real.png', np.uint8(real_frame * 255))
        else:
            fusion, denoise, refine, omega, gamma = model(ft0_torch, ft1_torch, coeff_a, coeff_b)
        for i in range(cfg.px_num - 1):
            fusion = pixel_shuffle(fusion)
            denoise = pixel_shuffle(denoise)
            refine = pixel_shuffle(refine)
            omega = pixel_shuffle(omega)
            gamma = pixel_shuffle(gamma)

        def gbrg2rgbg(data):
            data = torch.cat([data[:, 2:3, :, :],
                              data[:, 3:4, :, :],
                              data[:, 1:2, :, :],
                              data[:, 0:1, :, :]], dim=1)
            return data

        fusion_ = gbrg2rgbg(fusion)
        denoise_ = gbrg2rgbg(denoise)
        refine_ = gbrg2rgbg(refine)

        fusion_frame = tensor2numpy(isp(fusion_))[0]
        cv2.imwrite(folder + name + '_fusion.png', np.uint8(fusion_frame * 255))
        denoise_frame = tensor2numpy(isp(denoise_))[0]
        cv2.imwrite(folder + name + '_denoise.png', np.uint8(denoise_frame * 255))
        refine_frame = tensor2numpy(isp(refine_))[0]
        cv2.imwrite(folder + name + '_refine.png', np.uint8(refine_frame * 255))
        # ONNX
        ort_session = onnxruntime.InferenceSession("./%s/%s.onnx" % (subfolder, model_name))

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(ft0_torch),
                      ort_session.get_inputs()[1].name: to_numpy(ft1_torch),
                      ort_session.get_inputs()[2].name: to_numpy(coeff_a),
                      ort_session.get_inputs()[3].name: to_numpy(coeff_b)}
        output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
        if cfg.use_realism:
            output_names.extend(['real'])
            fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx, real_onnx = ort_session.run(output_names,
                                                                                                        ort_inputs)
            for i in range(cfg.px_num - 1):
                real_onnx = pixel_shuffle(torch.from_numpy(real_onnx)).cpu().numpy()
            np.testing.assert_allclose(to_numpy(real), real_onnx, rtol=1e-03, atol=1e-05)
            real_onnx_frame = tensor2numpy(isp(torch.from_numpy(real_onnx).to(device)))[0]
            cv2.imwrite(folder + name + '_real_onnx.png', np.uint8(real_onnx_frame * 255))
        else:
            fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx = ort_session.run(output_names, ort_inputs)
        for i in range(cfg.px_num - 1):
            fusion_onnx = pixel_shuffle(torch.from_numpy(fusion_onnx)).cpu().numpy()
            denoise_onnx = pixel_shuffle(torch.from_numpy(denoise_onnx)).cpu().numpy()
            refine_onnx = pixel_shuffle(torch.from_numpy(refine_onnx)).cpu().numpy()
            omega_onnx = pixel_shuffle(torch.from_numpy(omega_onnx)).cpu().numpy()
            gamma_onnx = pixel_shuffle(torch.from_numpy(gamma_onnx)).cpu().numpy()
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(refine), refine_onnx, rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        # fusion_onnx_frame = tensor2numpy(isp(torch.from_numpy(fusion_onnx).to(device)))[0]
        # cv2.imwrite(folder + name + '_fusion_onnx.png', np.uint8(fusion_onnx_frame * 255))
        # denoise_onnx_frame = tensor2numpy(isp(torch.from_numpy(denoise_onnx).to(device)))[0]
        # cv2.imwrite(folder + name + '_denoise_onnx.png', np.uint8(denoise_onnx_frame * 255))
        refine_ = np.concatenate([refine_onnx[:, 2:3, :, :],
                                  refine_onnx[:, 3:4, :, :],
                                  refine_onnx[:, 1:2, :, :],
                                  refine_onnx[:, 0:1, :, :]], axis=1)
        refine_onnx_frame = tensor2numpy(isp(torch.from_numpy(refine_).to(device)))[0]
        cv2.imwrite(folder + name + '_refine_onnx.png', np.uint8(refine_onnx_frame * 255))
    # f.close()


def convert_arch_qat(model_name):
    # Load the pretrained model
    model = architecture_qat.EMVD(cfg)
    device = cfg.device
    model = model.to(device)
    log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    ckpt = torch.load(os.path.join(log_dir, '{}.pth'.format(model_name)))
    state_dict = ckpt['model']
    model.load_state_dict(state_dict, strict=False)
    # model = model.eval()

    model.binnings_0.groupn.data = model.binnings.groupn.data.clone()
    model.binnings_1.groupn.data = model.binnings.groupn.data.clone()

    model.ct_0.w.data = model.ct.w.data.clone()
    model.ct_1.w.data = model.ct.w.data.clone()

    model.cti_fu.w.data = model.cti.w.data.clone()
    model.cti_de.w.data = model.cti.w.data.clone()
    model.cti_re.w.data = model.cti.w.data.clone()

    w1 = model.ft.w1.data
    w2 = model.ft.w2.data
    h0_row = w1
    h1_row = w2
    h0_row_t = w1.transpose(2, 3)
    h1_row_t = w2.transpose(2, 3)
    h00_row = h0_row * h0_row_t
    h01_row = h0_row * h1_row_t
    h10_row = h1_row * h0_row_t
    h11_row = h1_row * h1_row_t

    model.ft_00.h00_row.data = h00_row.clone()
    model.ft_00.h01_row.data = h01_row.clone()
    model.ft_00.h10_row.data = h10_row.clone()
    model.ft_00.h11_row.data = h11_row.clone()

    model.ft_10.h00_row.data = h00_row.clone()
    model.ft_10.h01_row.data = h01_row.clone()
    model.ft_10.h10_row.data = h10_row.clone()
    model.ft_10.h11_row.data = h11_row.clone()

    model.ft_01.h00_row.data = h00_row.clone()
    model.ft_01.h01_row.data = h01_row.clone()
    model.ft_01.h10_row.data = h10_row.clone()
    model.ft_01.h11_row.data = h11_row.clone()

    model.ft_11.h00_row.data = h00_row.clone()
    model.ft_11.h01_row.data = h01_row.clone()
    model.ft_11.h10_row.data = h10_row.clone()
    model.ft_11.h11_row.data = h11_row.clone()

    model.ft_02.h00_row.data = h00_row.clone()
    model.ft_02.h01_row.data = h01_row.clone()
    model.ft_02.h10_row.data = h10_row.clone()
    model.ft_02.h11_row.data = h11_row.clone()

    model.ft_12.h00_row.data = h00_row.clone()
    model.ft_12.h01_row.data = h01_row.clone()
    model.ft_12.h10_row.data = h10_row.clone()
    model.ft_12.h11_row.data = h11_row.clone()

    w1 = model.fti.w1.data
    w2 = model.fti.w2.data
    g0_col = w1
    g1_col = w2
    g0_col_t = g0_col.transpose(2, 3)
    g1_col_t = g1_col.transpose(2, 3)
    g00_col = g0_col * g0_col_t
    g01_col = g0_col * g1_col_t
    g10_col = g1_col * g0_col_t
    g11_col = g1_col * g1_col_t

    model.fti_d2.g00_col.data = g00_col.clone()
    model.fti_d2.g01_col.data = g01_col.clone()
    model.fti_d2.g10_col.data = g10_col.clone()
    model.fti_d2.g11_col.data = g11_col.clone()

    model.fti_d1.g00_col.data = g00_col.clone()
    model.fti_d1.g01_col.data = g01_col.clone()
    model.fti_d1.g10_col.data = g10_col.clone()
    model.fti_d1.g11_col.data = g11_col.clone()

    model.fti_fu.g00_col.data = g00_col.clone()
    model.fti_fu.g01_col.data = g01_col.clone()
    model.fti_fu.g10_col.data = g10_col.clone()
    model.fti_fu.g11_col.data = g11_col.clone()

    model.fti_de.g00_col.data = g00_col.clone()
    model.fti_de.g01_col.data = g01_col.clone()
    model.fti_de.g10_col.data = g10_col.clone()
    model.fti_de.g11_col.data = g11_col.clone()

    model.fti_re.g00_col.data = g00_col.clone()
    model.fti_re.g01_col.data = g01_col.clone()
    model.fti_re.g10_col.data = g10_col.clone()
    model.fti_re.g11_col.data = g11_col.clone()

    torch.save(model.state_dict(), "{}/{}_qat.pth".format(log_dir, model_name))


def to_onnx(name):
    # Load the pretrained model
    model = architecture_reparam.EMVD(cfg)
    model = model.to(cfg.device)
    log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    state_dict = torch.load(os.path.join(log_dir, '{}_reparam.pth'.format(name)))
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()
    ch = 16
    height = 3072
    width = 4096
    n = 3
    ft0 = Variable(torch.rand(1, ch, height // (2 ** n), width // (2 ** n))).to(cfg.device)
    ft1 = Variable(torch.rand(1, ch, height // (2 ** n), width // (2 ** n))).to(cfg.device)
    coeff_a = Variable(torch.rand(1, 1, 1, 1)).to(cfg.device)
    coeff_b = Variable(torch.rand(1, 1, 1, 1)).to(cfg.device)
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    torch.onnx.export(model,
                      (ft0, ft1, coeff_a, coeff_b),
                      os.path.join(log_dir, '{}_reparam.onnx'.format(name)),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['ft0', 'ft1', 'coeff_a', 'coeff_b'],
                      output_names=output_names)


def main():
    name = 'model_best'
    # reparameters(name)
    to_onnx(name)
    # check_model(name)


if __name__ == '__main__':
    main()

'''
snpe-onnx-to-dlc --input_network model_qua_arch.onnx --output_path model_qua_arch.dlc 
2021-12-09 19:40:58,559 - 199 - INFO - INFO_DLC_SAVE_LOCATION: Saving model at model_qua_arch.dlc
/home/wen/env/snpe-1.54.2.2899/lib/python/qti/aisw/converters/backend/ir_to_dlc.py:308: RuntimeWarning: info_code=802; message=Layer parameter value is invalid in GPU. Layer Concat_206 : output width = 240, depth = 400 width * depth (packed) = 24000 exceeds maximum image width 16384 for Adreno A650; component=GPU Runtime; line_no=875; thread_id=140571610281792
  node.op.axis)
/home/wen/env/snpe-1.54.2.2899/lib/python/qti/aisw/converters/backend/ir_to_dlc.py:292: RuntimeWarning: info_code=802; message=Layer parameter value is invalid in GPU. Layer Conv_207 : input width = 240, depth = 400 width * depth (packed) = 24000 exceeds maximum image width 16384 for Adreno A650; component=GPU Runtime; line_no=875; thread_id=140571610281792
  node.op.groups)
/home/wen/env/snpe-1.54.2.2899/lib/python/qti/aisw/converters/backend/ir_to_dlc.py:308: RuntimeWarning: info_code=802; message=Layer parameter value is invalid in GPU. Layer Concat_212 : output width = 240, depth = 528 width * depth (packed) = 31680 exceeds maximum image width 16384 for Adreno A650; component=GPU Runtime; line_no=875; thread_id=140571610281792
  node.op.axis)
/home/wen/env/snpe-1.54.2.2899/lib/python/qti/aisw/converters/backend/ir_to_dlc.py:292: RuntimeWarning: info_code=802; message=Layer parameter value is invalid in GPU. Layer Conv_213 : input width = 240, depth = 528 width * depth (packed) = 31680 exceeds maximum image width 16384 for Adreno A650; component=GPU Runtime; line_no=875; thread_id=140571610281792
  node.op.groups)
/home/wen/env/snpe-1.54.2.2899/lib/python/qti/aisw/converters/backend/ir_to_dlc.py:292: RuntimeWarning: info_code=802; message=Layer parameter value is invalid in GPU. Layer Conv_225 : input width = 240, depth = 528 width * depth (packed) = 31680 exceeds maximum image width 16384 for Adreno A650; component=GPU Runtime; line_no=875; thread_id=140571610281792
  node.op.groups)
2021-12-09 19:40:58,798 - 199 - INFO - INFO_CONVERSION_SUCCESS: Conversion completed successfully

'''

'''
[WARNING] Converting TF quantized 8->32 bit. Consider quantizing directly from float for the best accuracy.
[WARNING] Converting TF quantized 8->32 bit. Consider quantizing directly from float for the best accuracy.
/local/mnt/workspace/mlg_user_admin/docker.ci.tmp_htp/build/x86_64-linux-clang/SecondParty/QNN/src/qnn-htp-emu/DSP/HTP/src/hexagon/src/graph.cc:295:ERROR:couldn't create op: q::Padzap.tcm
/local/mnt/workspace/mlg_user_admin/docker.ci.tmp_htp/build/x86_64-linux-clang/SecondParty/QNN/src/qnn-htp-emu/DSP/HTP/src/hexagon/src/graph.cc:566:ERROR:Op 88c000000094 preparation failed with err:-1
[1] QnnDsp graph prepare failed 12
[ERROR] SNPE HTP Offline Prepare: Failed to generate QNN HTP graph cache.
[INFO] SNPE HTP Offline Prepare: Done creating QNN HTP graph cache for Vtcm size 4 MB.
[INFO] Successfully compiled HTP metadata into DLC.
[INFO] DebugLog shutting down.
'''
