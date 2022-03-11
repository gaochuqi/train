import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torchvision
from torch.autograd import Variable

import shutil
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

import arch_qat
import arch
import architecture
import architecture_qat

import netloss as netloss
from load_data import *
import time

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
import collections
#####################################################################################
# imports for AIMET
#####################################################################################
# import aimet_common
# from aimet_torch import bias_correction
# from aimet_torch.cross_layer_equalization import equalize_model
# from aimet_torch.quantsim import QuantParams, QuantizationSimModel
from typing import Tuple
from functools import partial
import logging
from datetime import datetime

logger = logging.getLogger('TorchQAT')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)
#####################################################################################

iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
pixel_shuffle = architecture.PixelShuffle(2)
pixel_unshuffle = architecture.PixelShuffle(0.5)

def initialize():
    """
    # clear some dir if necessary
    make some dir if necessary
    make sure training from scratch
    :return:
    """
    ##
    if not os.path.exists(cfg.model_name):
        os.mkdir(cfg.model_name)

    if not os.path.exists(cfg.debug_dir):
        os.mkdir(cfg.debug_dir)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    # if cfg.checkpoint == None:
    #     s = input('Are you sure training the model from scratch? y/n \n')
    #     if not (s == 'y'):
    #         return

def duplicate_output_to_log(name):
    tee = utils.Tee(name)
    return tee

mse_criterion = torch.nn.MSELoss(reduction='mean')

def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1 / len(features)] * len(features)
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += mse_criterion(f, t) * w
    return content_loss

def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

def loss_color(model, device):       # Color Transform
    '''
    :param model:
    :param layers: layer name we want to use orthogonal regularization
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        # k = '.'.join(name.split('.')[1:])
        # params[k] = param
        params[name] = param
    # for k,v in params.items():
    # print(k,v.shape)
    if cfg.ngpu > 1:
        ct = params['module.ct.net1.weight'].squeeze()
        cti = params['module.cti.net1.weight'].squeeze()
    else:
        ct = params['ct.net1.weight'].squeeze()
        cti = params['cti.net1.weight'].squeeze()
    weight_squared = torch.matmul(ct, cti)
    diag = torch.eye(weight_squared.shape[0], dtype=torch.float32, device=device)
    loss = ((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

def loss_wavelet(model, device): # Frequency Transform
    '''
    :param model:
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        #k = '.'.join(name.split('.')[1:])
        #params[k] = param
        params[name] = param
    if cfg.ngpu > 1:
        ft = torch.cat([params['module.ft.w1'], params['module.ft.w2']], dim=0).squeeze()
        fti = torch.cat([params['module.fti.w1'],params['module.fti.w2']],dim= 0).squeeze()
    else:
        ft = torch.cat([params['ft.w1'], params['ft.w2']], dim=0).squeeze()
        fti = torch.cat([params['fti.w1'], params['fti.w2']], dim=0).squeeze()

    weight_squared = torch.matmul(ft, fti)
    diag = torch.eye(weight_squared.shape[1], dtype=torch.float32, device=device)
    loss=((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

def train(in_data, gt_raw_data, noisy_level, model, loss, device, optimizer, loss_network=None):
    l1loss_list = []
    l1loss_total = 0
    content_loss_total = 0
    tv_loss_total = 0
    coeff_a = (noisy_level[0] / (2 ** 12 - 1 - 240)).float().to(device)
    coeff_a = coeff_a[:, None, None, None]
    coeff_b = (noisy_level[1] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
    coeff_b = coeff_b[:, None, None, None]
    for time_ind in range(cfg.frame_num):
        ft1 = in_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]  # the t-th input frame
        fgt = gt_raw_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]  # the t-th gt frame
        if time_ind == 0:
            ft0_fusion = ft1
        else:
            ft0_fusion = ft0_fusion_data  # the t-1 fusion frame

        input = torch.cat([ft0_fusion, ft1], dim=1)

        model.train()
        fusion_out, denoise_out, refine_out, omega, gamma = model(input, coeff_a, coeff_b)
        loss_refine = loss(refine_out, fgt)
        loss_fusion = loss(fusion_out, fgt)
        loss_denoise = loss(denoise_out, fgt)

        l1loss = loss_refine

        l1loss_list.append(l1loss)
        l1loss_total += l1loss

        ft0_fusion_data = fusion_out

        rgb_gt = torch.cat([fgt[:, 0:1, :, :],
                            (fgt[:, 1:2, :, :] + fgt[:, 3:4, :, :]) / 2,
                            fgt[:, 2:3, :, :]], 1)
        rgb_pred = torch.cat([refine_out[:, 0:1, :, :],
                              (refine_out[:, 1:2, :, :] + refine_out[:, 3:4, :, :]) / 2,
                              refine_out[:, 2:3, :, :]], 1)

        if cfg.use_perceptual_loss and loss_network is not None:
            target_content_features = extract_features(loss_network, rgb_gt, cfg.content_layers)
            output_content_features = extract_features(loss_network, rgb_pred, cfg.content_layers)
            content_loss = calc_Content_Loss(output_content_features, target_content_features)
            content_loss_total += content_loss
        else:
            content_loss_total = torch.tensor(0., dtype=l1loss.dtype, device=device)
        if cfg.use_tvr:
            tv_loss = calc_TV_Loss(refine_out)
            tv_loss_total += tv_loss
        else:
            tv_loss_total = torch.tensor(0., dtype=l1loss.dtype, device=device)

    # loss_ct = netloss.loss_color(model, device)
    loss_ct = torch.tensor(0., dtype=l1loss.dtype, device=device)

    # loss_ft = netloss.loss_wavelet(model, device)
    loss_ft = torch.tensor(0., dtype=l1loss.dtype, device=device)

    loss_content = content_loss_total / (cfg.frame_num)
    loss_tv = tv_loss_total / (cfg.frame_num)

    loss_l1 = l1loss_total / (cfg.frame_num)
    total_loss = loss_l1 + loss_ct + loss_ft \
                 + loss_content \
                 + loss_tv

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
    optimizer.step()

    print('Loss                 |   ',
          ('%.8f' % total_loss.item()),
          ('%.8f' % loss_l1.item()),
          ('%.10f' % loss_ct.item()),
          ('%.10f' % loss_ft.item()),
          ('%.10f' % loss_content.item()),
          ('%.10f' % loss_tv.item()))

    del in_data, gt_raw_data
    return ft1, fgt, refine_out, fusion_out, denoise_out, omega, gamma, \
           total_loss, loss_ct, loss_ft, loss_fusion, loss_denoise, \
            loss_content, loss_tv

def evaluate(model, psnr, writer=None, iter=None):
    print('Evaluate...')
    cnt = 0
    total_psnr = 0
    total_psnr_raw = 0
    model.eval()
    with torch.no_grad():
        for scene_ind in range(7, 12):
            for noisy_level in range(0, 5):
                in_data, gt_raw_data = load_eval_data(noisy_level, scene_ind)
                frame_psnr = 0
                frame_psnr_raw = 0
                for time_ind in range(cfg.frame_num):
                    ft1 = in_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]
                    fgt = gt_raw_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]
                    if time_ind == 0:
                        ft0_fusion = ft1
                    else:
                        ft0_fusion = ft0_fusion_data

                    coeff_a = a_list[noisy_level] / (2 ** 12 - 1 - 240)
                    coeff_b = b_list[noisy_level] / (2 ** 12 - 1 - 240) ** 2
                    input = torch.cat([ft0_fusion, ft1], dim=1)

                    fusion_out, denoise_out, refine_out, omega, gamma = model(input, coeff_a, coeff_b)

                    ft0_fusion_data = fusion_out

                    frame_psnr += psnr(refine_out, fgt)
                    frame_psnr_raw += psnr(ft1, fgt)

                frame_psnr = frame_psnr / (cfg.frame_num)
                frame_psnr_raw = frame_psnr_raw / (cfg.frame_num)
                print('---------')
                print('Scene: ', ('%02d' % scene_ind), 'Noisy_level: ', ('%02d' % noisy_level), 'PSNR: ',
                      '%.8f' % frame_psnr.item())
                total_psnr += frame_psnr
                total_psnr_raw += frame_psnr_raw
                cnt += 1
                del in_data, gt_raw_data
        total_psnr = total_psnr / cnt
        total_psnr_raw = total_psnr_raw / cnt
    print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
    if writer is not None:
        writer.add_scalar('PSNR', total_psnr.item(), iter)
        writer.add_scalar('PSNR_RAW', total_psnr_raw.item(), iter)
        writer.add_scalar('PSNR_IMP', total_psnr.item() - total_psnr_raw.item(), iter)
        torch.cuda.empty_cache()
        return total_psnr, total_psnr_raw
    else:
        torch.cuda.empty_cache()
        return total_psnr

def extract_features(model, x, layers):
    features = []
    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features


def convert_arch(cfg, subfolder, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coeff_a = torch.tensor(a_list[0] / (2 ** 12 - 1 - 240)).float().to(device)
    coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1])
    coeff_b = torch.tensor(b_list[1] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
    coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1])

    h = cfg.height
    w = cfg.width
    # Load model
    # best_model_save_root = os.path.join(subfolder, '%s.pth'%model_name)
    best_model_save_root = cfg.best_model_save_root
    print(best_model_save_root)
    checkpoint = torch.load(best_model_save_root)
    state_dict = checkpoint['model']
    if cfg.use_pixel_shuffle:
        model = architecture_qat.EMVD(cfg)
    else:
        model = arch_qat.EMVD(cfg)
    model = model.to(device)
    model.load_state_dict(state_dict, strict=False)

    #########################################################################
    # ct
    n = 16
    cfan = torch.zeros((n, n, 1, 1), device=model.ct.w.device)
    c = 4  # n // 4
    for i in range(4):
        for j in range(c):
            cfan[i * 4 + j, j, :, :] = model.ct.w[i, 0]
            cfan[i * 4 + j, j + c, :, :] = model.ct.w[i, 1]
            cfan[i * 4 + j, j + c * 2, :, :] = model.ct.w[i, 2]
            cfan[i * 4 + j, j + c * 3, :, :] = model.ct.w[i, 3]
    model.ct.net.weight.data = nn.Parameter(cfan, requires_grad=True)
    model_ct_net_weight = nn.Parameter(cfan, requires_grad=True)
    #########################################################################
    # cti
    #########################################################################
    n = 16
    cfan_inv = torch.zeros((n, n, 1, 1), device=model.ct.w.device)
    c = 4  # n // 4
    for i in range(4):
        for j in range(c):
            cfan_inv[i * 4 + j, j, :, :] = model.cti.w[i, 0]
            cfan_inv[i * 4 + j, j + c, :, :] = model.cti.w[i, 1]
            cfan_inv[i * 4 + j, j + c * 2, :, :] = model.cti.w[i, 2]
            cfan_inv[i * 4 + j, j + c * 3, :, :] = model.cti.w[i, 3]
    model.cti.net.weight.data = nn.Parameter(cfan_inv, requires_grad=True)
    model_cti_net_weight = nn.Parameter(cfan_inv, requires_grad=True)
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
    if cfg.use_pixel_shuffle:
        filters_ft = torch.zeros((64, 16, 2, 2), device=h00_row.device)
        for i in range(4):
            for j in range(16):
                filters_ft[i * 16 + j, j, :, :] = filters1[i][0, 0, :, :]
    else:
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
    # #######################
    # for name, params in model.named_parameters():
    #     print(name)
    # #######################
    model.ft.net.weight.data = nn.Parameter(filters_ft)
    model_ft_net_weight = nn.Parameter(filters_ft, requires_grad=True)
    ##########################################################################
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
    if cfg.use_pixel_shuffle:
        filters_fti = torch.zeros((64, 16, 2, 2), device=g00_col.device)
        for i in range(4):
            for j in range(16):
                filters_fti[i * 16 + j, j, :, :] = filters2[i][0, 0, :, :]
    else:
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
    model.fti.net.weight.data = nn.Parameter(filters_fti)
    model_fti_net_weight = nn.Parameter(filters_fti, requires_grad=True)
    #########################################################################
    model.ct0.net.weight.data = model.ct.net.weight.data
    model.ct1.net.weight.data = model.ct.net.weight.data

    model.cti_fu.net.weight.data = model.cti.net.weight.data
    model.cti_de.net.weight.data = model.cti.net.weight.data
    model.cti_re.net.weight.data = model.cti.net.weight.data

    model.ft_00.net.weight.data = model.ft.net.weight.data
    model.ft_10.net.weight.data = model.ft.net.weight.data
    model.ft_01.net.weight.data = model.ft.net.weight.data
    model.ft_11.net.weight.data = model.ft.net.weight.data
    model.ft_02.net.weight.data = model.ft.net.weight.data
    model.ft_12.net.weight.data = model.ft.net.weight.data

    model.fti_d2.net.weight.data = model.fti.net.weight.data
    model.fti_d1.net.weight.data = model.fti.net.weight.data
    model.fti_fu.net.weight.data = model.fti.net.weight.data
    model.fti_de.net.weight.data = model.fti.net.weight.data
    model.fti_re.net.weight.data = model.fti.net.weight.data

    #########################################################################
    # filters = torch.zeros((16, 64, 1, 1)).to(device)
    # for i in range(16):
    #     for j in range(4):
    #         filters[i, i + j * 16, :, :] = model.groupn[i, j, :, :]
    # model.bin.weight = nn.Parameter(filters)
    #########################################################################

    # model.eval()
    # torch.save(model.state_dict(), "./%s/%s_qua_arch.pth" % (subfolder, model_name))
    if cfg.use_pixel_shuffle:
        # input = Variable(torch.randn(1, 128, h//4, w//4))
        h = cfg.height
        w = cfg.width
        ft0 = Variable(torch.randn(1, 16, h // 2, w // 2))
        ft0 = ft0.cuda()
        # ft1 = Variable(torch.randn(1, c*4, h // (2 ** n), w // (2 ** n)))
        ft1 = Variable(torch.randn(1, 16, h // 2, w // 2))
        ft1 = ft1.cuda()
        output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    else:
        input = Variable(torch.randn(1, 8, h, w))
        input = input.cuda()
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])
    # inputs = {
    #     "ft0": ft0,
    #     "ft1": ft1,
    #     "coeff_a": coeff_a,
    #     "coeff_b": coeff_b,
    # }
    # torch.onnx.export(model,
    #                   (inputs["input"],coeff_a,coeff_b),
    #                   "./%s/%s_qua_arch.onnx" % (subfolder, model_name),
    #                   opset_version=11,
    #                   do_constant_folding=True,
    #                   input_names=['input','coeff_a','coeff_b'],
    #                   output_names=output_names
    #                   )
    torch.save(model.state_dict(), "./%s/%s_qua_arch.pth" % (subfolder, model_name))
    inputs = {
        "ft0": ft0,
        "ft1": ft1,
        "coeff_a": coeff_a,
        "coeff_b": coeff_b,
    }
    torch.onnx.export(model,
                      (ft0, ft1, coeff_a, coeff_b),
                      "./%s/%s_qua_arch.onnx" % (subfolder, model_name),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['ft0', 'ft1', 'coeff_a', 'coeff_b'],
                      output_names=output_names)


def check_onnx(subfolder, model_name):
    isp = torch.load('isp/ISP_CNN.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import onnx
    onnx_model = onnx.load("%s/%s.onnx" % (subfolder, model_name))
    onnx.checker.check_model(onnx_model)
    # input
    folder = './%s/' % (subfolder)
    name = 'ISO1600_scene7_frame1'
    crvd_path = '/home/pb/yushu/CRVD_dataset/indoor_rgb/' + name + '_input_data.raw'
    coeff_a = torch.tensor(a_list[0] / (2 ** 12 - 1 - 240)).float().to(device)
    coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1])
    coeff_b = torch.tensor(b_list[1] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
    coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1])

    ##########################################################
    bin_np = np.fromfile('./bin.raw', dtype=np.float32)
    bin_np = np.reshape(bin_np, (16, 64, 1, 1))
    bin_w = torch.from_numpy(bin_np).to(device)
    conv_bin = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False).to(device)
    conv_bin.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    ##########################################################

    input_data = np.fromfile(crvd_path, dtype=np.float32)
    input_data = np.reshape(input_data, (1, 8, 544, 960))
    ft0 = input_data[:, 0:4, :, :]
    ft0 = torch.from_numpy(ft0).to(device)
    ft0_frame = tensor2numpy(isp(ft0))[0]
    cv2.imwrite(folder + name + '_ft0.png', np.uint8(ft0_frame * 255))
    ##################################################
    if cfg.use_pixel_shuffle:
        for i in range(cfg.px_num - 1 - 1):
            ft0 = pixel_unshuffle(ft0)
    ft0.permute(0, 2, 3, 1).cpu().detach().numpy().tofile('./ISO1600_scene7_frame1_ft0_bin_bhwc.raw')
    ##################################################
    ft1 = input_data[:, 4:8, :, :]
    ft1 = torch.from_numpy(ft1).to(device)
    ft1_frame = tensor2numpy(isp(ft1))[0]
    cv2.imwrite(folder + name + '_ft1.png', np.uint8(ft1_frame * 255))
    ##################################################
    if cfg.use_pixel_shuffle:
        for i in range(cfg.px_num - 1 - 1):
            ft1 = pixel_unshuffle(ft1)
    print('conv bin', ft1.shape)
    ft1.permute(0, 2, 3, 1).cpu().detach().numpy().tofile('./ISO1600_scene7_frame1_ft1_bin_bhwc.raw')
    ##################################################
    # input_data_torch = torch.from_numpy(input_data).to(device)
    # # print('intput:', input_data_torch.shape)
    # if cfg.use_pixel_shuffle:
    #     # input_data_torch = pixel_unshuffle(input_data_torch)
    #     for i in range(cfg.px_num - 1):
    #         input_data_torch = pixel_unshuffle(input_data_torch)
    # # print('intput after ps:', input_data_torch.shape)
    # torch model
    # Load the pretrained model
    if cfg.use_arch_qat:
        if cfg.use_pixel_shuffle:
            model = architecture_qat.EMVD(cfg)
        else:
            model = arch_qat.EMVD(cfg)
    else:
        if cfg.use_pixel_shuffle:
            # model = arch_qat.EMVD(cfg)
            ###############################################
            model = architecture_qat.EMVD(cfg)
            ###############################################
        else:
            model = arch.EMVD(cfg)
    model = model.to(device)
    state_dict = torch.load("./%s/%s.pth" % (subfolder, model_name))
    if 'iter' in state_dict.keys():
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=True)
    # model.eval()
    if cfg.use_realism:
        fusion, denoise, refine, omega, gamma, real = model(ft0, ft1, coeff_a, coeff_b)
        if cfg.use_pixel_shuffle:
            real = pixel_shuffle(real)
        real_frame = tensor2numpy(isp(real))[0]
        cv2.imwrite(folder + name + '_real.png', np.uint8(real_frame * 255))
    else:
        fusion, denoise, refine, omega, gamma = model(ft0, ft1, coeff_a, coeff_b)
    if cfg.use_pixel_shuffle:
        fusion = pixel_shuffle(fusion)
        denoise = pixel_shuffle(denoise)
        refine = pixel_shuffle(refine)
        omega = pixel_shuffle(omega)
        gamma = pixel_shuffle(gamma)
    fusion_frame = tensor2numpy(isp(fusion))[0]
    cv2.imwrite(folder + name + '_fusion.png', np.uint8(fusion_frame * 255))
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
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(ft0),
                  ort_session.get_inputs()[1].name: to_numpy(ft1),
                  ort_session.get_inputs()[2].name: to_numpy(coeff_a),
                  ort_session.get_inputs()[3].name: to_numpy(coeff_b)}
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])
        fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx, real_onnx = ort_session.run(output_names,
                                                                                                    ort_inputs)
        if cfg.use_pixel_shuffle:
            real_onnx = pixel_shuffle(torch.from_numpy(real_onnx)).cpu().numpy()
        np.testing.assert_allclose(to_numpy(real), real_onnx, rtol=1e-03, atol=1e-05)
        real_onnx_frame = tensor2numpy(isp(torch.from_numpy(real_onnx).to(device)))[0]
        cv2.imwrite(folder + name + '_real_onnx.png', np.uint8(real_onnx_frame * 255))
    else:
        fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx = ort_session.run(output_names, ort_inputs)
    if cfg.use_pixel_shuffle:
        fusion_onnx = pixel_shuffle(torch.from_numpy(fusion_onnx)).cpu().numpy()
        denoise_onnx = pixel_shuffle(torch.from_numpy(denoise_onnx)).cpu().numpy()
        refine_onnx = pixel_shuffle(torch.from_numpy(refine_onnx)).cpu().numpy()
        omega_onnx = pixel_shuffle(torch.from_numpy(omega_onnx)).cpu().numpy()
        gamma_onnx = pixel_shuffle(torch.from_numpy(gamma_onnx)).cpu().numpy()
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(refine), refine_onnx, rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    fusion_onnx_frame = tensor2numpy(isp(torch.from_numpy(fusion_onnx).to(device)))[0]
    cv2.imwrite(folder + name + '_fusion_onnx.png', np.uint8(fusion_onnx_frame * 255))
    denoise_onnx_frame = tensor2numpy(isp(torch.from_numpy(denoise_onnx).to(device)))[0]
    cv2.imwrite(folder + name + '_denoise_onnx.png', np.uint8(denoise_onnx_frame * 255))
    refine_onnx_frame = tensor2numpy(isp(torch.from_numpy(refine_onnx).to(device)))[0]
    cv2.imwrite(folder + name + '_refine_onnx.png', np.uint8(refine_onnx_frame * 255))


def main():
    convert = True
    cfg.use_arch_qat = True
    cfg.use_realism = False
    if cfg.use_realism:
        sub_folder = 'model_photo_real'
        model_name = 'model_real_best'
    else:
        sub_folder = './log/a100/model_px_ecb_No' # 'model_px'
        model_name = 'model' # 'model_best'
    #######################################################
    sub_folder = './model'
    model_name = cfg.model_name
    #######################################################

    if convert:
        if cfg.use_arch_qat:
            convert_arch(cfg, sub_folder, model_name)
            model_name += '_qua_arch'
        check_onnx(sub_folder, model_name)
        return


if __name__ == '__main__':
    initialize()
    main()