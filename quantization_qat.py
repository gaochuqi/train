import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torchvision
from torch.autograd import Variable
import onnx
import onnxruntime

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
import architecture_reparam

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
import aimet_common
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel
from typing import Tuple
from functools import partial
import logging
from datetime import datetime

logger = logging.getLogger('TorchQAT')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)
#####################################################################################
cfg.model_name = 'qat'
print(cfg.model_name)

isp = torch.load('isp/ISP_CNN.pth')
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

def calculate_quantsim_accuracy(model: torch.nn.Module,
                                evaluator: aimet_common.defs.EvalFunction,
                                psnr,
                                use_cuda: bool = False,
                                logdir: str = '') -> Tuple[torch.nn.Module, float]:
    """
    Calculates model accuracy on quantized simulator and returns quantized model with accuracy.

    :param model: the loaded model
    :param evaluator: the Eval function to use for evaluation
    :param iterations: No of batches to use in computing encodings.
                       Not used in image net dataset
    :param num_val_samples_per_class: No of samples to use from every class in
                                      computing encodings. Not used in pascal voc
                                      dataset
    :param use_cuda: the cuda device.
    :return: a tuple of quantsim and accuracy of model on this quantsim
    """

    input_shape = (1, cfg.image_channels,
                   cfg.height,
                   cfg.height,)
    if use_cuda:
        model.to(torch.device('cuda'))
        dummy_input = torch.rand(input_shape).cuda()
    else:
        dummy_input = torch.rand(input_shape)
    quant_sim = QuantizationSimModel(prepared_model, dummy_input=dummy_input,
                                     quant_scheme=QuantScheme.post_training_tf_enhanced,
                                     default_param_bw=8, default_output_bw=8,
                                     config_file='../../TrainingExtensions/common/src/python/aimet_common/quantsim_config/'
                                                 'default_config.json')
    quantsim = QuantizationSimModel(model=model, quant_scheme='tf_enhanced',
                                    dummy_input=dummy_input, rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8, in_place=False)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator),
                               forward_pass_callback_args=psnr)

    quantsim.export(path=logdir, filename_prefix='model_encodings', dummy_input=dummy_input.cpu())
    accuracy = evaluator(quantsim.model, psnr)

    return quantsim, accuracy

def apply_cross_layer_equalization(model: torch.nn.Module, input_shape: tuple):
    """
    Applies CLE on the model and calculates model accuracy on quantized simulator
    Applying CLE on the model inplace consists of:
        Batch Norm Folding
        Cross Layer Scaling
        High Bias Fold
    Converts any ReLU6 into ReLU.

    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """

    equalize_model(model, input_shape)

def check_model(cfg, subfolder, model, model_qua):
    device = cfg.device
    # Load model
    path1 = os.path.join(subfolder, '%s.pth' % model)
    ckpt1 = torch.load(path1)
    state_dict = ckpt1['model']
    model1 = architecture.EMVD(cfg)
    model1 = model1.to(cfg.device)
    model1.load_state_dict(state_dict, strict=True)

    path2 = os.path.join(subfolder, '%s.pth' % model_qua)
    state_dict = torch.load(path2)
    model2 = architecture_reparam.EMVD(cfg) # architecture_qat.EMVD(cfg)
    model2 = model2.to(cfg.device)
    model2.load_state_dict(state_dict, strict=False)
    # input
    name = 'ISO1600_scene7_frame1'
    idx = 0
    iso = 1600
    scene_ind = 7
    frame_ind = 1
    coeff_a = torch.tensor(a_list[idx] / (2 ** 12 - 1 - 240)).float().to(device)
    coeff_a = torch.reshape(coeff_a, [1, 1, 1, 1])
    coeff_b = torch.tensor(b_list[idx] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
    coeff_b = torch.reshape(coeff_b, [1, 1, 1, 1])
    folder = './%s/' % (subfolder)
    input_pack, img = get_input(iso, scene_ind, frame_ind)
    # noisy8 = np.concatenate([input_pack, input_pack], axis=3)
    # noisy8.tofile('%s/%s_noisy8_bhwc.raw' % (subfolder, name))
    noisy4 = np.transpose(input_pack, (0, 3, 1, 2))
    img = torch.from_numpy(img).to(device)
    img_frame = tensor2numpy(isp(img))[0]
    cv2.imwrite(folder + name + '_noisy.png', np.uint8(img_frame * 255))
    # input_data = np.concatenate([noisy4, noisy4], axis=1)
    ft0_torch = torch.from_numpy(noisy4).to(device)
    ft1_torch = torch.from_numpy(noisy4).to(device)
    for i in range(cfg.px_num):
        ft0_torch = pixel_unshuffle(ft0_torch)
        ft1_torch = pixel_unshuffle(ft1_torch)

    bin_np = np.fromfile('%s/bin.raw' % (subfolder), dtype=np.float32)
    bin_np = np.reshape(bin_np, (16, 64, 1, 1))
    bin_w = torch.from_numpy(bin_np).to(device)
    conv_bin = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False).to(device)
    conv_bin.weight = torch.nn.Parameter(bin_w, requires_grad=False)

    ft0 = conv_bin(ft0_torch) # model1.binnings(ft0_torch)
    ft1 = conv_bin(ft1_torch)
    output1 = model1(ft0, ft1, coeff_a, coeff_b)
    output2 = model2(ft0, ft1, coeff_a, coeff_b)
    fusion1, denoise1, refine1, omega1, gamma1 = output1
    fusion2, denoise2, refine2, omega2, gamma2 = output2
    # for i in range(len(output2)):
    # for k,v in output1.items():
    #     err = torch.where(output1[k]!=output2[k])
    #     if err[0].shape[0] > 0:
    #         print(k) # ,output1[k],output2[k])
    print(refine1==refine2)
    print('end')

def check_result_raw():
    path = '/home/wen/Documents/project/video/denoising/emvd_bin/log/model/results/Result_4/refine.raw'
    input_data = np.fromfile(path, dtype=np.float32)
    input_data = np.reshape(input_data, (1, 272, 480, 16))
    input_data_bhwc = np.transpose(input_data, (0, 3, 1, 2))
    data = torch.from_numpy(input_data_bhwc)
    for i in range(cfg.px_num - 1):
        data = pixel_shuffle(data)

    def gbrg2rgbg(data):
        data = torch.cat([data[:, 2:3, :, :],
                          data[:, 3:4, :, :],
                          data[:, 1:2, :, :],
                          data[:, 0:1, :, :]], dim=1)
        return data

    refine = gbrg2rgbg(data).to(cfg.device)
    refine_frame = tensor2numpy(isp(refine))[0]
    cv2.imwrite('./log/model/snpe_dlc_refine.png', np.uint8(refine_frame * 255))


def reparameters(cfg, subfolder, model_name):
    # input
    name = 'ISO1600_scene7_frame1'
    idx = 0
    iso = 1600
    scene_ind = 7
    frame_ind = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coeff_a = a_list[idx] / (2 ** 12 - 1 - 240)
    coeff_a = torch.tensor(coeff_a).float().to(device)
    coeff_a = torch.reshape(coeff_a,[1,1,1,1])
    coeff_b = b_list[idx] / (2 ** 12 - 1 - 240) ** 2
    coeff_b = torch.tensor(coeff_b).float().to(device)
    coeff_b = torch.reshape(coeff_b,[1,1,1,1])
    #########################################################################
    # Load Float model
    #########################################################################
    best_model_save_root = os.path.join(subfolder, '%s.pth'%model_name)
    print(best_model_save_root)
    checkpoint = torch.load(best_model_save_root)
    state_dict = checkpoint['model']
    model = architecture.EMVD(cfg)
    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    # model.eval()
    #########################################################################
    # Load Re-parameters model
    #########################################################################
    model_reparam = architecture_reparam.EMVD(cfg)
    model_reparam = model_reparam.to(device)
    model_reparam.load_state_dict(state_dict, strict=False)
    # model_reparam.eval()
    # err = torch.ones((16, 16, 1, 1), device=model.ct.w.device)
    # model_reparam.err.weight = nn.Parameter(err)
    # #########################################################################
    # # binning
    # #########################################################################
    # # n = 4 ** cfg.px_num
    # # filter_bin = torch.zeros((16, 64, 1, 1), device=cfg.device)
    # # for i in range(4):
    # #     for j in range(4):
    # #         filter_bin[i, j * n + i, :, :] = model.binning.bin_gb[0, j, :, :]
    # #         filter_bin[4 + i, j * n + i + 4, :, :] = model.binning.bin_b[0, j, :, :]
    # #         filter_bin[8 + i, j * n + i + 8, :, :] = model.binning.bin_r[0, j, :, :]
    # #         filter_bin[12 + i, j * n + i + 12, :, :] = model.binning.bin_gr[0, j, :, :]
    # # model_binning_conv_bin_weight = nn.Parameter(filter_bin)
    # # model_reparam.bin.conv_bin.weight.data = model_binning_conv_bin_weight.data
    #########################################################################
    c = 4 ** cfg.px_num * 4
    n = 4
    g = c // n
    filters = torch.zeros((g, c, 1, 1), device=model.binnings.groupn.device)
    for i in range(g):
        for j in range(n):
            filters[i, i + j * g, :, :] = model.binnings.groupn[i, j, :, :]
    model_binnings_conv_bin_weight = nn.Parameter(filters, requires_grad=True)
    model_binnings_conv_bin_weight.data.cpu().numpy().tofile('%s/bin.raw' % (subfolder))
    # model_reparam.bin.conv_bin.weight = model_binnings_conv_bin_weight #.clone()
    #########################################################################
    #########################################################################
    # ct
    #########################################################################
    n = 4 ** cfg.px_num
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
    n = 4 ** cfg.px_num
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
    n = 4 ** cfg.px_num
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
    n = 4 ** cfg.px_num
    filters_fti = torch.zeros((n * 4, n, 2, 2), device=g00_col.device)
    for i in range(4):
        for j in range(n):
            filters_fti[n * i + j, j, :, :] = filters2[i][0, 0, :, :]
    model_fti_net_weight = nn.Parameter(filters_fti, requires_grad=True)
    # #########################################################################
    # model_reparam.ct.net.weight = model_ct_net_weight #.clone() # model.ct.net.weight.data
    model_reparam.ct0.net.weight.data = model_ct_net_weight.data # .clone() # model.ct.net.weight.data
    model_reparam.ct1.net.weight.data = model_ct_net_weight.data # .clone() # model.ct.net.weight.data

    # model_reparam.cti.net.weight = model_cti_net_weight #.clone() # model.cti.net.weight.data
    model_reparam.cti_fu.net.weight.data = model_cti_net_weight.data # .clone() # model.cti.net.weight.data
    model_reparam.cti_de.net.weight.data = model_cti_net_weight.data # .clone() # model.cti.net.weight.data
    model_reparam.cti_re.net.weight.data = model_cti_net_weight.data # .clone() # model.cti.net.weight.data

    # model_reparam.ft.net.weight = model_ft_net_weight #.clone() # model.ft.net.weight.data
    model_reparam.ft_00.net.weight.data = model_ft_net_weight.data #.clone() # model.ft.net.weight.data
    model_reparam.ft_10.net.weight.data = model_ft_net_weight.data #.clone() # model.ft.net.weight.data
    model_reparam.ft_01.net.weight.data = model_ft_net_weight.data #.clone() # model.ft.net.weight.data
    model_reparam.ft_11.net.weight.data = model_ft_net_weight.data #.clone() # model.ft.net.weight.data
    model_reparam.ft_02.net.weight.data = model_ft_net_weight.data #.clone() # model.ft.net.weight.data
    model_reparam.ft_12.net.weight.data = model_ft_net_weight.data #.clone() # model.ft.net.weight.data

    # model_reparam.fti.net.weight = model_fti_net_weight #.clone() # model.fti.net.weight.data
    model_reparam.fti_d2.net.weight.data = model_fti_net_weight.data #.clone() # model.fti.net.weight.data
    model_reparam.fti_d1.net.weight.data = model_fti_net_weight.data #.clone() # model.fti.net.weight.data
    model_reparam.fti_fu.net.weight.data = model_fti_net_weight.data #.clone() # model.fti.net.weight.data
    model_reparam.fti_de.net.weight.data = model_fti_net_weight.data #.clone() # model.fti.net.weight.data
    model_reparam.fti_re.net.weight.data = model_fti_net_weight.data #.clone() # model.fti.net.weight.data
    # #########################################################################
    #########################################################################

    if cfg.use_realism:
        model_reparam.fti_realism.net.weight.data = model_fti_realism_net_weight.data # model.fti.net.weight.data
        model_reparam.cti_realism.net.weight.data = model_cti_realism_net_weight.data # model.cti.net.weight.data
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
    h = cfg.height
    w = cfg.width
    n = cfg.px_num
    c = 4 ** n
    ft0 = Variable(torch.randn(1, c, h // (2**n), w // (2**n)))
    ft0 = ft0.cuda()
    # ft1 = Variable(torch.randn(1, c*4, h // (2 ** n), w // (2 ** n)))
    ft1 = Variable(torch.randn(1, c, h // (2**n), w // (2**n)))
    ft1 = ft1.cuda()
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])
    # model_reparam.eval()
    # with torch.no_grad():
    #     outputs = model_reparam(ft0,ft1, coeff_a, coeff_b)
    torch.save(model_reparam.state_dict(), "./%s/%s_qua_arch.pth" % (subfolder, model_name))
    inputs = {
        "ft0": ft0,
        "ft1": ft1,
        "coeff_a": coeff_a,
        "coeff_b": coeff_b,
    }
    torch.onnx.export(model_reparam,
                      (ft0,ft1,coeff_a,coeff_b),
                      "./%s/%s_qua_arch.onnx" % (subfolder, model_name),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['ft0','ft1','coeff_a','coeff_b'],
                      output_names=output_names)

def convert_arch(cfg, subfolder, model_name):
    # input
    name = 'ISO1600_scene7_frame1'
    idx = 0
    iso = 1600
    scene_ind = 7
    frame_ind = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coeff_a = a_list[idx] / (2 ** 12 - 1 - 240)
    coeff_a = torch.tensor(coeff_a).float().to(device)
    coeff_a = torch.reshape(coeff_a,[1,1,1,1])
    coeff_b = b_list[idx] / (2 ** 12 - 1 - 240) ** 2
    coeff_b = torch.tensor(coeff_b).float().to(device)
    coeff_b = torch.reshape(coeff_b,[1,1,1,1])


    # Load model
    best_model_save_root = os.path.join(subfolder, '%s.pth'%model_name)
    print(best_model_save_root)
    # checkpoint = torch.load(best_model_save_root)
    # state_dict = checkpoint['model']
    state_dict = torch.load(best_model_save_root)
    if cfg.use_pixel_shuffle:
        model = architecture_reparam.EMVD(cfg) # architecture_qat.EMVD(cfg)
    else:
        model = arch_qat.EMVD(cfg)
    model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    # #########################################################################
    # # ct
    # #########################################################################
    # n = 4 ** cfg.px_num
    # cfan = torch.zeros((n, n, 1, 1), device=model.ct.w.device)
    # c = 4  # n // 4
    # for i in range(4):
    #     for j in range(c):
    #         cfan[i * 4 + j, j, :, :] = model.ct.w[i, 0]
    #         cfan[i * 4 + j, j + c, :, :] = model.ct.w[i, 1]
    #         cfan[i * 4 + j, j + c * 2, :, :] = model.ct.w[i, 2]
    #         cfan[i * 4 + j, j + c * 3, :, :] = model.ct.w[i, 3]
    # model.ct.net.weight = nn.Parameter(cfan)
    # #########################################################################
    # # cti
    # #########################################################################
    # n = 4 ** cfg.px_num
    # cfan_inv = torch.zeros((n, n, 1, 1), device=model.ct.w.device)
    # c = 4  # n // 4
    # for i in range(4):
    #     for j in range(c):
    #         cfan_inv[i * 4 + j, j, :, :] = model.cti.w[i, 0]
    #         cfan_inv[i * 4 + j, j + c, :, :] = model.cti.w[i, 1]
    #         cfan_inv[i * 4 + j, j + c * 2, :, :] = model.cti.w[i, 2]
    #         cfan_inv[i * 4 + j, j + c * 3, :, :] = model.cti.w[i, 3]
    # model.cti.net.weight = nn.Parameter(cfan_inv)
    # #########################################################################
    # # ft
    # #########################################################################
    # h0_row = model.ft.w1
    # h1_row = model.ft.w2
    # h0_row_t = model.ft.w1.transpose(2, 3)
    # h1_row_t = model.ft.w2.transpose(2, 3)
    # h00_row = h0_row * h0_row_t
    # h01_row = h0_row * h1_row_t
    # h10_row = h1_row * h0_row_t
    # h11_row = h1_row * h1_row_t
    # filters1 = [h00_row, h01_row, h10_row, h11_row]
    # n = 4 ** cfg.px_num
    # filters_ft = torch.zeros((n * 4, n, 2, 2), device=h00_row.device)
    # for i in range(4):
    #     for j in range(n):
    #         filters_ft[n * i + j, j, :, :] = filters1[i][0, 0, :, :]
    # model.ft.net.weight = nn.Parameter(filters_ft)
    # #########################################################################
    # # fti
    # #########################################################################
    # g0_col = model.fti.w1
    # g1_col = model.fti.w2
    # g0_col_t = model.fti.w1.transpose(2, 3)
    # g1_col_t = model.fti.w2.transpose(2, 3)
    # g00_col = g0_col * g0_col_t
    # g01_col = g0_col * g1_col_t
    # g10_col = g1_col * g0_col_t
    # g11_col = g1_col * g1_col_t
    # filters2 = [g00_col, g10_col, g01_col, g11_col]
    # n = 4 ** cfg.px_num
    # filters_fti = torch.zeros((n * 4, n, 2, 2), device=g00_col.device)
    # for i in range(4):
    #     for j in range(n):
    #         filters_fti[n * i + j, j, :, :] = filters2[i][0, 0, :, :]
    # model.fti.net.weight = nn.Parameter(filters_fti)
    # #########################################################################
    # # for i in range(4):
    # #     for j in range(4):
    # #         print(i, j * n + i)
    # #         print(4 + i, j * n + i + 4)
    # #         print(8 + i, j * n + i + 8)
    # #         print(12 + i, j * n + i + 12)
    # #########################################################################
    # n = 4 ** cfg.px_num
    # filter_bin = torch.zeros((16, 64, 1, 1), device=cfg.device)
    # for i in range(4):
    #     for j in range(4):
    #         filter_bin[i, j * n + i, :, :] = model.binning.bin_gb[0, j, :, :]
    #         filter_bin[4 + i, j * n + i + 4, :, :] = model.binning.bin_b[0, j, :, :]
    #         filter_bin[8 + i, j * n + i + 8, :, :] = model.binning.bin_r[0, j, :, :]
    #         filter_bin[12 + i, j * n + i + 12, :, :] = model.binning.bin_gr[0, j, :, :]
    # model.binning.conv_bin.weight = nn.Parameter(filter_bin)
    # # model.bin.conv_bin.weight.data = model.binning.conv_bin.weight.data
    # # #########################################################################
    # # c = 4 ** cfg.px_num * 4
    # # n = 4
    # # g = c // n
    # # filters = torch.zeros((g, c, 1, 1)).to(device)
    # # for i in range(g):
    # #     for j in range(n):
    # #         filters[i, i + j * g, :, :] = model.binnings.groupn[i, j, :, :]
    # # model.binnings.conv_bin.weight = nn.Parameter(filters)
    # # #########################################################################
    # model.ct0.net.weight.data = model.ct.net.weight.data.clone()
    model.ct1.net.weight.data = model.ct0.net.weight.data.clone()
    #
    # model.cti_fu.net.weight.data = model.cti.net.weight.data.clone()
    model.cti_de.net.weight.data = model.cti_fu.net.weight.data.clone()
    model.cti_re.net.weight.data = model.cti_fu.net.weight.data.clone()
    #
    # model.ft_00.net.weight.data = model.ft.net.weight.data.clone()
    model.ft_10.net.weight.data = model.ft_00.net.weight.data.clone()
    model.ft_01.net.weight.data = model.ft_00.net.weight.data.clone()
    model.ft_11.net.weight.data = model.ft_00.net.weight.data.clone()
    model.ft_02.net.weight.data = model.ft_00.net.weight.data.clone()
    model.ft_12.net.weight.data = model.ft_00.net.weight.data.clone()
    #
    # model.fti_d2.net.weight.data = model.fti.net.weight.data.clone()
    model.fti_d1.net.weight.data = model.fti_d2.net.weight.data.clone()
    model.fti_fu.net.weight.data = model.fti_d2.net.weight.data.clone()
    model.fti_de.net.weight.data = model.fti_d2.net.weight.data.clone()
    model.fti_re.net.weight.data = model.fti_d2.net.weight.data.clone()
    # #########################################################################
    # if cfg.use_realism:
    #     model.fti_realism.net.weight.data = model.fti.net.weight.data
    #     model.cti_realism.net.weight.data = model.cti.net.weight.data
    # if cfg.use_ecb:
    #     depth = len(model.ecb)
    #     for d in range(depth):
    #         module = model.ecb[d]
    #         act_type = module.act_type
    #         RK, RB = module.rep_params()
    #         model.eocb[d].conv.weight.data = RK
    #         model.eocb[d].conv.bias.data = RB
    #
    #         if act_type == 'relu':
    #             pass
    #         elif act_type == 'linear':
    #             pass
    #         elif act_type == 'prelu':
    #             model.ecb[d].act.weight.data = module.act.weight.data
    #         else:
    #             raise ValueError('invalid type of activation!')
    # #########################################################################
    h = cfg.height
    w = cfg.width
    n = cfg.px_num
    c = 4 ** n
    ft0 = Variable(torch.randn(1, c, h // (2**n), w // (2**n)))
    ft0 = ft0.cuda()
    ft1 = Variable(torch.randn(1, c*4, h // (2 ** n), w // (2 ** n)))
    ft1 = ft1.cuda()
    output_names = ['fusion', 'denoise', 'refine', 'omega', 'gamma']
    if cfg.use_realism:
        output_names.extend(['real'])
    # model.eval()
    # outputs = model(ft0,ft1, coeff_a, coeff_b)
    torch.save(model.state_dict(), "./%s/%s_000.pth" % (subfolder, model_name))
    inputs = {
        "ft0": ft0,
        "ft1": ft1,
        "coeff_a": coeff_a,
        "coeff_b": coeff_b,
    }
    torch.onnx.export(model,
                      (ft0,ft1,coeff_a,coeff_b),
                      "./%s/%s_000.onnx" % (subfolder, model_name),
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['ft0','ft1','coeff_a','coeff_b'],
                      output_names=output_names)

def get_input(iso, scene_ind, frame_ind):
    # frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
    input_pack_list = []
    noisy_frame_index_for_current = np.random.choice(10, 4, replace=False)
    for i in noisy_frame_index_for_current:
        input_name = os.path.join(cfg.data_root[1],
                                  'indoor_raw_noisy/indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(
                                      scene_ind, scene_ind, iso,
                                      frame_ind, i))
        noisy_raw = cv2.imread(input_name, -1)
        input_pack = np.expand_dims(pack_gbrg_raw(noisy_raw), axis=0)
        input_pack = np.pad(input_pack, [(0, 0), (4, 4), (0, 0), (0, 0)], mode='constant')
        input_pack_list.append(input_pack.astype('float32'))
    im = input_pack_list[0][0]
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    img = np.concatenate((im[1:H:2, 0:W:2, :],          # r
                          im[1:H:2, 1:W:2, :],          # gr
                          im[0:H:2, 1:W:2, :],          # b
                          im[0:H:2, 0:W:2, :]), axis=2) # gb
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img,(0, 3, 1, 2)) # 1,4,h,w
    input_pack = np.concatenate(input_pack_list, axis=-1)
    return input_pack, img

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
    bin_np = np.fromfile('%s/bin.raw' % (subfolder), dtype=np.float32)
    bin_np = np.reshape(bin_np, (16, 64, 1, 1))
    bin_w = torch.from_numpy(bin_np).to(device)
    conv_bin = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False).to(device)
    conv_bin.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    # input
    path_raw = '/home/wen/Documents/project/video/denoising/emvd_bin/data/'
    names = ['ISO1600_scene7_frame1','ISO1600_scene8_frame3','ISO1600_scene9_frame1',
             'ISO3200_scene9_frame2','ISO3200_scene10_frame1','ISO3200_scene11_frame4',
             'ISO6400_scene6_frame4','ISO1600_scene5_frame3','ISO1600_scene4_frame6',
             'ISO12800_scene3_frame7','ISO12800_scene2_frame5','ISO12800_scene5_frame1',
             'ISO25600_scene4_frame3','ISO25600_scene8_frame4','ISO25600_scene11_frame5',]
    # f = open('raw_list_crvd_local_bhwc_bin.txt', 'w')
    for name in names:
        iso = int(name.split('_')[0][3:]) # 1600
        scene_ind = int(name.split('_')[1][5:]) # 7
        frame_ind = int(name.split('_')[2][5:]) # 1
        idx = iso_list.index(iso)
        coeff_a = torch.tensor(a_list[idx] / (2 ** 12 - 1 - 240)).float().to(device)
        coeff_a = torch.reshape(coeff_a,[1,1,1,1])
        coeff_b = torch.tensor(b_list[idx] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
        coeff_b = torch.reshape(coeff_b,[1,1,1,1])
        # coeff_b.data.cpu().numpy().tofile('%s/coeff_b_%d.raw' % (subfolder,iso))
        # coeff_a.data.cpu().numpy().tofile('%s/coeff_a_%d.raw' % (subfolder,iso))
        # path_coeff_a = path_raw+'coeff_a_%d.raw'% (iso)
        # path_coeff_b = path_raw + 'coeff_b_%d.raw' % (iso)

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
        ft0_torch = conv_bin(ft0_torch) # model.binning(ft0_torch)
        ft1_torch = conv_bin(ft1_torch)
        ft0_torch.permute(0,2,3,1).data.cpu().numpy().tofile('%s/%s_ft0_bin_bhwc.raw' % (subfolder, name))
        ft1_torch.permute(0,2,3,1).data.cpu().numpy().tofile('%s/%s_ft1_bin_bhwc.raw' % (subfolder, name))
        # path_ft0 = path_raw + '%s_ft0_bin_bhwc.raw' % (name)
        # path_ft1 = path_raw + '%s_ft1_bin_bhwc.raw' % (name)
        # content = '%s %s %s %s\n'%(path_ft0,path_ft1,path_coeff_a,path_coeff_b)
        # f.write(content)
        if cfg.use_realism:
            fusion, denoise, refine, omega, gamma, real = model(ft0_torch,ft1_torch,coeff_a,coeff_b)
            for i in range(cfg.px_num - 1 - 1):
                real = pixel_shuffle(real)
            real_frame = tensor2numpy(isp(real))[0]
            cv2.imwrite(folder + name + '_real.png', np.uint8(real_frame * 255))
        else:
            fusion, denoise, refine, omega, gamma = model(ft0_torch,ft1_torch,coeff_a,coeff_b)
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
        cv2.imwrite(folder + name +'_fusion.png', np.uint8(fusion_frame * 255))
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
            fusion_onnx, denoise_onnx, refine_onnx, omega_onnx, gamma_onnx, real_onnx = ort_session.run(output_names, ort_inputs)
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

def onnx_remove_bin_node(subfolder, model_name):
    model = onnx.load("./%s/%s.onnx" % (subfolder, model_name))
    model.graph.input[1].type.tensor_type.shape.dim[1].dim_value = 16
    old_nodes = model.graph.node
    new_nodes = old_nodes[1:]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    nodes = model.graph.node
    for i in range(len(nodes)):
        node = nodes[i]
        for key in node.input:
            if "ct1.net.weight" in key:
                node.input[0] = 'ft1'
    onnx.save_model(model,"./%s/%s.onnx" % (subfolder, "model") )

def main():
    # check_result_raw()
    convert = True
    cfg.use_arch_qat = True
    cfg.use_realism = False
    cfg.use_pixel_shuffle = True
    cfg.use_ecb = False
    cfg.use_attFeatFusion = False
    if cfg.use_realism:
        sub_folder = 'model_photo_real'
        model_name = 'model_real_best'
    else:
        sub_folder = './log_bk/models/' # './log/model/yushu/'
        model_name = 'model' # 'model_best'

    if convert:
        if cfg.use_arch_qat:
            reparameters(cfg, sub_folder, model_name)
            model_name += '_qua_arch'
            # onnx_remove_bin_node(sub_folder, model_name)
            # convert_arch(cfg, sub_folder, model_name)
        check_model(cfg, sub_folder, 'model', 'model_qua_arch')
        check_onnx(sub_folder, 'model_qua_arch')
        return


    # checkpoint = cfg.checkpoint
    state_dict_path = cfg.model_qua_arch_root
    start_epoch = cfg.start_epoch
    start_iter = cfg.start_iter
    best_psnr = 0

    ## use gpu
    device = cfg.device
    ngpu = cfg.ngpu
    cudnn.benchmark = True

    ## tensorboard --logdir runs
    writer = SummaryWriter(os.path.join(cfg.model_name, 'log'))

    ## initialize model
    # model = structure.MainDenoise(cfg)
    model = arch_qat.EMVD(cfg)

    ## compute GFLOPs
    # stat(model, (8,512,512))

    model = model.to(device)
    loss = netloss.L1Loss().to(device)
    psnr = netloss.PSNR().to(device)

    learning_rate = cfg.learning_rate
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    loss_network = None
    if cfg.use_perceptual_loss:
        # Loss network
        loss_network = torchvision.models.__dict__[cfg.vgg_flag](pretrained=False)
        state_dict = torch.load(cfg.vgg_path)
        loss_network.load_state_dict(state_dict)
        loss_network.training = False
        for param in loss_network.parameters():
            param.requires_grad = False
        loss_network = loss_network.features.to(device)

    ## load pretrained model
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    iter = start_iter

    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    shutil.copy('config.py', os.path.join(cfg.model_name))
    shutil.copy('arch_qat.py', os.path.join(cfg.model_name))
    shutil.copy('qat.py', os.path.join(cfg.model_name))
    shutil.copy('netloss.py', os.path.join(cfg.model_name))


    train_data_name_queue = generate_file_list(['1', '2', '3', '4', '5', '6'])
    train_dataset = loadImgs(train_data_name_queue)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    ##################################################################
    # quantization aware training
    ##################################################################
    '''
    1. Instantiates Data Pipeline for evaluation
    2. Loads the pretrained resnet18 Pytorch model
    3. Calculates Model accuracy
        3.1. Calculates floating point accuracy
        3.2. Calculates Quant Simulator accuracy
    4. Applies AIMET CLE and BC
        4.1. Applies AIMET CLE and calculates QuantSim accuracy
        4.2. Applies AIMET BC and calculates QuantSim accuracy

    '''
    ##################################################################
    default_logdir = os.path.join(cfg.model_name, "benchmark_output", "QAT" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if not os.path.exists(default_logdir):
        os.makedirs(default_logdir)
    use_cuda = True if torch.cuda.is_available() else False
    model = model.eval()
    # Calculate FP32 accuracy
    eval_psnr, eval_psnr_raw = evaluate(model, psnr, writer, iter)
    logger.info("Original Model PSNR = %.2f", eval_psnr)
    logger.info("Starting Model Quantization")
    # Quantize the model using AIMET QAT (quantization aware training) and calculate accuracy on Quant Simulator
    quantsim, accuracy = calculate_quantsim_accuracy(model=model,
                                                     evaluator=evaluate,
                                                     psnr=psnr,
                                                     use_cuda=use_cuda,
                                                     logdir=default_logdir)

    logger.info("Quantized Model PSNR = %.2f", accuracy)
    input_shape = (1, cfg.image_channels,
                   cfg.height,
                   cfg.width)
    # For good initialization apply, apply Post Training Quantization (PTQ) methods
    # such as Cross Layer Equalization (CLE) and Bias Correction (BC) (optional)
    apply_cross_layer_equalization(model=model, input_shape=input_shape)
    # apply_bias_correction(model=model, data_loader=data_loader) # No bias in Model
    quantsim, _ = calculate_quantsim_accuracy(model=model,
                                              evaluator=evaluate,
                                              psnr=psnr,
                                              use_cuda=use_cuda,
                                              logdir=default_logdir)
    logger.info("Post Training Quantization (PTQ) Complete")
    ##################################################################
    # Finetune the quantized model
    ##################################################################
    logger.info("Starting Model Finetuning")
    model = quantsim.model
    ##################################################################
    # https://developer.qualcomm.com/sites/default/files/docs/snpe/quantized_models.html
    '''Adjusted Weights Quantization Mode
        This mode is used only for quantizing weights to 8 bit fixed point
        (invoked by using the "use_adjusted_weights_quantizer" parameter to snpe-dlc-quantize), 
        which uses adjusted min or max of the data being quantized other than 
        true min/max or the min/max that exclude the long tail. 
        This has been verified to be able to provide accuracy benefit for denoise model specifically. 
        Using this quantizer, the max will be expanded or the min will be decreased if necessary.
        Adjusted weights quantizer still enforces a minimum range and ensures 0.0 is exactly quantizable.
    '''
    ##################################################################
    dummy_input = torch.rand(input_shape)
    finetune_epochs = 30
    for epoch in range(start_epoch, finetune_epochs):
        print('------------------------------------------------')
        print('Epoch                |   ', ('%08d' % epoch))
        for i, (input, label, noisy_level) in enumerate(train_loader):
            print('------------------------------------------------')
            print('Iter                 |   ', ('%08d' % iter))
            in_data = input.permute(0, 3, 1, 2).to(device)
            gt_raw_data = label.permute(0, 3, 1, 2).to(device)

            ft1, fgt, refine_out, fusion_out, denoise_out, omega, gamma, \
            total_loss, loss_ct, loss_ft, loss_fusion, loss_denoise,\
                loss_content, loss_tv = train(in_data,
                                                gt_raw_data,
                                                noisy_level,
                                                model,
                                                loss,
                                                device,
                                                optimizer,
                                                loss_network)
            iter = iter + 1
            if iter % cfg.log_step == 0:
                input_gray = torch.mean(ft1, 1, True)
                label_gray = torch.mean(fgt, 1, True)
                predict_gray = torch.mean(refine_out, 1, True)
                fusion_gray = torch.mean(fusion_out, 1, True)
                denoise_gray = torch.mean(denoise_out, 1, True)
                # gamma_gray = torch.mean(gamma[:, 0:1, :, :], 1, True)
                # omega_gray = torch.mean(omega[:, 0:1, :, :], 1, True)
                gamma_gray = torch.mean(gamma[:, 0:16, :, :], 1, True)
                omega_gray = torch.mean(omega[:, 0:16, :, :], 1, True)

                writer.add_image('input', make_grid(input_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('fusion_out', make_grid(fusion_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('denoise_out', make_grid(denoise_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('refine_out', make_grid(predict_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('label', make_grid(label_gray.cpu(), nrow=4, normalize=True), iter)

                writer.add_image('gamma', make_grid(gamma_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('omega', make_grid(omega_gray.cpu(), nrow=4, normalize=True), iter)

                writer.add_scalar('L1Loss', total_loss.item(), iter)
                writer.add_scalar('L1Color', loss_ct.item(), iter)
                writer.add_scalar('L1Wavelet', loss_ft.item(), iter)
                writer.add_scalar('L1Denoise', loss_denoise.item(), iter)
                writer.add_scalar('L1Fusion', loss_fusion.item(), iter)
                if cfg.use_perceptual_loss:
                    writer.add_scalar('Perceptual_Losses', loss_content.item(), iter)
                    writer.add_scalar('total_variation_regularizer', loss_tv.item(), iter)

                torch.save({
                    'epoch': epoch,
                    'iter': iter,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(cfg.model_name, 'model_qat.pth'))

            if iter % cfg.valid_step == 0:
                eval_psnr, eval_psnr_raw = evaluate(model, psnr, writer, iter)
                if eval_psnr > best_psnr:
                    best_psnr = eval_psnr
                    torch.save({
                        'epoch': epoch,
                        'iter': iter,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_psnr': best_psnr},
                        os.path.join(cfg.model_name, 'model_qat_best.pth'))
                    quantsim.export(path=cfg.model_name,
                                    filename_prefix='QAT_model_best',
                                    dummy_input=dummy_input.cpu())

    ##################################################################
    # Calculate and log the accuracy of quantized-finetuned model
    eval_psnr, eval_psnr_raw = evaluate(model, psnr, writer, iter)
    logger.info("After Quantization Aware Training, PSNR = %.2f", eval_psnr)
    ##################################################################
    logger.info("Quantization Aware Training Complete")
    ##################################################################

    # Save the quantized model
    quantsim.export(path=cfg.model_name,
                    filename_prefix='QAT_model',
                    dummy_input=dummy_input.cpu())

    writer.close()



if __name__ == '__main__':
    initialize()
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