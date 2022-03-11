import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torchvision
import shutil
from PIL import ImageFile
import functools
import pytorch_msssim
from torch.optim import lr_scheduler
import torch.nn as nn
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import time
import cv2
import warnings
import torch
import random
import numpy as np
import collections

warnings.filterwarnings('ignore')
from torchstat import stat

import utils
# from dataset import *
import config as cfg
from arch import architecture
# import arch
from utils import netloss #  as netloss
from dataset.load_data import generate_file_list, load_eval_data, loadImgs
from utils import VGG_Model
from utils.ContextualLoss import Contextual_Loss
from utils import contextual_loss as cl
from utils.models import models
from generate.tools import setup_seed, tensor2numpy
from arch.modules import PixelShuffle


def duplicate_output_to_log(name):
    tee = utils.Tee(name)
    return tee


def define_D():
    # netD = arch.Discriminator(in_nc=4, base_nf=64)
    netD = architecture.Discriminator(in_nc=4, base_nf=64)
    utils.init_weights(netD, init_type='kaiming', scale=1)
    return netD


def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss

def show_rgb(ft,isp,name):
    # gb,b,r,gr => r,gr,b,gb
    ftt = torch.cat([ft[:, 2:3, :, :],
                      ft[:, 3:4, :, :],
                      ft[:, 1:2, :, :],
                      ft[:, 0:1, :, :]], dim=1)
    rgb_ft = isp(ftt)
    rgb_ft = tensor2numpy(rgb_ft)[0]
    cv2.imwrite(os.path.join('log', cfg.model_name, '%s.png'%name), np.uint8(rgb_ft * 255))

def train(in_data, gt_raw_data, noisy_level,
          model, loss, device, optimizer,
          loss_network=None,
          isp=None,
          cri_msssim=None,
          cri_ml1=None,
          cri_gan=None,
          discriminator=None,
          optimizer_D=None,
          optimizer_G=None,
          cri_edge=None,
          pixel_shuffle = None,
          pixel_unshuffle = None,
          ):
    total_loss = 0
    loss_dict = {}
    loss_l1_total = 0
    loss_fusion_total = 0
    loss_denoise_total = 0
    loss_content_total = 0
    loss_tvr_total = 0
    loss_real_total = 0
    loss_gan_total = 0
    loss_ml1_total = 0
    loss_msssim_total = 0
    loss_edge_total = 0
    fake = []
    ftgt = []
    real = None
    ###########################################################################
    # coeff_a = (noisy_level[0] / (cfg.white_level - cfg.black_level)).float().to(device)
    coeff_a = noisy_level[0].float().to(device)
    coeff_a = coeff_a[:, None, None, None]
    # coeff_b = (noisy_level[1] / (cfg.white_level - cfg.black_level) ** 2).float().to(device)
    coeff_b = noisy_level[1].float().to(device)
    coeff_b = coeff_b[:, None, None, None]
    ###########################################################################
    for time_ind in range(cfg.frame_num):
        ft1 = in_data[:, time_ind: (time_ind + 1), :, :]  # the t-th input frame
        fgt = gt_raw_data[:, time_ind: (time_ind + 1), :, :]  # the t-th gt frame, (b,1,h,w)
        fgt = pixel_unshuffle(fgt) # (b,4,h/2,w/2)

        for i in range(cfg.px_num):
            ft1 = pixel_unshuffle(ft1)
        #######################################################################
        if time_ind == 0:
            ft0_fusion = ft1
        else:
            ft0_fusion = ft0_fusion_data  # the t-1 fusion frame

        model.train()
        outputs = model(ft0_fusion, ft1, coeff_a, coeff_b)
        #######################################################################
        if cfg.use_realism:
            if cfg.use_gan_loss:
                for p in discriminator.parameters():
                    p.requires_grad = False
            optimizer_G.zero_grad()
            fusion_out, denoise_out, refine_out, \
            omega, gamma, \
            real = outputs
            for i in range(cfg.px_num - 1):
                real = pixel_shuffle(real)
            ftgt.append(fgt)
            fake.append(real)
        else:
            optimizer.zero_grad()
            fusion_out, denoise_out, refine_out, \
            omega, gamma = outputs
        #######################################################################
        ft0_fusion_data = fusion_out
        #######################################################################
        # for i in range(cfg.px_num-1):
        fusion_out = pixel_shuffle(fusion_out)
        denoise_out = pixel_shuffle(denoise_out)
        refine_out = pixel_shuffle(refine_out)
        omega = pixel_shuffle(omega)
        gamma = pixel_shuffle(gamma)
        for i in range(cfg.px_num-1):
            ft1 = pixel_shuffle(ft1)
        #######################################################################
        test_show = False
        if test_show:
            show_rgb(fgt, isp, 'gt')
            show_rgb(ft1, isp, 'ft1')
            show_rgb(fusion_out, isp, 'fusion')
            show_rgb(denoise_out, isp, 'denoise')
            show_rgb(refine_out, isp, 'refine')
            cv2.imwrite(os.path.join('log', cfg.model_name, '{}.png'.format('omega')),
                        np.uint8(tensor2numpy(omega)[0] * 255))
            cv2.imwrite(os.path.join('log', cfg.model_name, '{}.png'.format('gamma')),
                        np.uint8(tensor2numpy(gamma)[0] * 255))

        #######################################################################
        # loss          emvd
        #######################################################################
        if not cfg.freeze_emvd:
            loss_refine = loss(refine_out, fgt)
            loss_fusion = loss(fusion_out, fgt)
            loss_denoise = loss(denoise_out, fgt)
            loss_l1_total += loss_refine
            loss_fusion_total += loss_fusion
            loss_denoise_total += loss_denoise
        #######################################################################
        # loss          Edge loss Sobel
        #######################################################################
        if cfg.use_edge_loss:
            loss_edge = cri_edge(refine_out, fgt)
            loss_edge_total += loss_edge
        #######################################################################
        # loss          Structure - GAN
        #######################################################################
        if cfg.use_realism:
            if cfg.use_real_L1:
                loss_real = loss(real, fgt)
                loss_real_total += loss_real  # .item()
            if cfg.use_gan_loss and cri_gan is not None:
                pred_g_fake = discriminator(real)
                pred_g_real = discriminator(fgt)
                pred_g_real.detach_()
                loss_gan = cfg.lambda_gan * (cri_gan(pred_g_real - torch.mean(pred_g_fake), False) +
                                             cri_gan(pred_g_fake - torch.mean(pred_g_real), True)) / 2
                loss_gan_total += loss_gan  # .item()
            if cfg.use_structure_loss and cri_ml1 is not None:
                loss_ml1 = cri_ml1(real, fgt)
                loss_ml1_total += loss_ml1  # .item()
            if cfg.use_structure_loss and cri_msssim is not None:
                loss_msssim = 1.0 - cri_msssim(real, fgt)
                loss_msssim_total += loss_msssim  # .item()
        #######################################################################
        # loss          Perceptual & Contextual - VGG
        #######################################################################
        if cfg.use_perceptual_loss or cfg.use_contextual_loss and loss_network is not None:
            rg1b_gt = torch.cat([fgt[:, 0:1, :, :],
                                 fgt[:, 1:2, :, :],
                                 fgt[:, 2:3, :, :]], 1)
            rg1b_pred = torch.cat([refine_out[:, 0:1, :, :],
                                   refine_out[:, 1:2, :, :],
                                   refine_out[:, 2:3, :, :]], 1)
            rg2b_gt = torch.cat([fgt[:, 0:1, :, :],
                                 fgt[:, 3:4, :, :],
                                 fgt[:, 2:3, :, :]], 1)
            rg2b_pred = torch.cat([refine_out[:, 0:1, :, :],
                                   refine_out[:, 3:4, :, :],
                                   refine_out[:, 2:3, :, :]], 1)
            if cfg.use_isp:
                rgb_gt = isp(fgt)
                rgb_pred = isp(refine_out)
                target_content_features = extract_features(loss_network, rgb_gt, cfg.content_layers)
                output_content_features = extract_features(loss_network, rgb_pred, cfg.content_layers)
            else:
                target_content_features = extract_features(loss_network, rg1b_gt, cfg.content_layers)
                output_content_features = extract_features(loss_network, rg1b_pred, cfg.content_layers)
                target_content_features.extend(extract_features(loss_network, rg2b_gt, cfg.content_layers))
                output_content_features.extend(extract_features(loss_network, rg2b_pred, cfg.content_layers))
            if cfg.use_perceptual_loss:
                content_loss = netloss.calc_Content_Loss(output_content_features,
                                                         target_content_features,
                                                         ltype='l1')
            elif cfg.use_contextual_loss:
                layers = {}
                if 'vgg19' in cfg.vgg_flag:
                    vgg = VGG_Model.vgg19_layer_inv
                if 'vgg16' in cfg.vgg_flag:
                    vgg = VGG_Model.vgg16_layer_inv
                for i in cfg.content_layers:
                    layers[vgg[i]] = 1.0
                max_1d_size = cfg.max_1d_size
                contex_loss = Contextual_Loss(layers, max_1d_size=max_1d_size).cuda()
                # content_loss_total = contex_loss.calculate_CX_Loss(output_content_features[0], target_content_features[0])
                for i in range(len(target_content_features)):
                    N, C, H, W = output_content_features[i].size()
                    if H * W > max_1d_size ** 2:
                        output_content_features[i] = contex_loss._random_pooling(output_content_features[i],
                                                                                 output_1d_size=max_1d_size)
                        target_content_features[i] = contex_loss._random_pooling(target_content_features[i],
                                                                                 output_1d_size=max_1d_size)
                    content_loss += contex_loss.calculate_CX_Loss(output_content_features[i],
                                                                  target_content_features[i])
            loss_content_total += content_loss
            del target_content_features
            del output_content_features
        # total variation regularizer
        if cfg.use_tvr:
            loss_tv = calc_TV_Loss(refine_out)
            loss_tvr_total += loss_tv
    ###########################################################################
    # loss          EMVD
    ###########################################################################
    if not cfg.freeze_emvd:
        if cfg.ngpu > 1:
            loss_ct = netloss.loss_color(model, ['module.ct.net.weight', 'module.cti.net.weight'], device)
        else:
            loss_ct = netloss.loss_color(model, ['ct.net.weight', 'cti.net.weight'], device)
        loss_ft = netloss.loss_wavelet(model, device)
        loss_l1_per = loss_l1_total / (cfg.frame_num)
        loss_l1_fu_per = loss_fusion_total / (cfg.frame_num)
        loss_l1_de_per = loss_denoise_total / (cfg.frame_num)
        total_loss = loss_l1_per + loss_ct + loss_ft
        if cfg.use_fusion_loss:
            total_loss += loss_l1_fu_per
        if cfg.use_denoise_loss:
            total_loss += loss_l1_de_per
        loss_dict['L1_Refine'] = loss_l1_per
        loss_dict['L1_Color'] = loss_ct
        loss_dict['L1_Wavelet'] = loss_ft
        loss_dict['L1_Fusion'] = loss_l1_fu_per
        loss_dict['L1_Denoise'] = loss_l1_de_per
    ###########################################################################
    if cfg.use_perceptual_loss or cfg.use_contextual_loss and loss_network is not None:
        loss_content_per = loss_content_total / (cfg.frame_num) * cfg.lambda_perceptual  # loss weight
        total_loss += loss_content_per
        if cfg.use_perceptual_loss:
            loss_dict['Perceptual'] = loss_content_per
        if cfg.use_contextual_loss:
            loss_dict['Contextual'] = loss_content_per
    if cfg.use_tvr:
        loss_tv_per = loss_tvr_total / (cfg.frame_num)
        total_loss += loss_tv_per
        loss_dict['Total_Variation'] = loss_tv_per
    if cfg.use_realism:
        if cfg.use_real_L1:
            loss_real_per = loss_real_total / (cfg.frame_num)
            total_loss += loss_real_per
            loss_dict['L1_Real'] = loss_real_per
        if cfg.use_gan_loss and cri_gan is not None:
            loss_gan_per = loss_gan_total / (cfg.frame_num)
            total_loss += loss_gan_per
            loss_dict['GAN_Loss'] = loss_gan_per
        if cfg.use_structure_loss and cri_msssim is not None:
            loss_msssim_per = loss_msssim_total / (cfg.frame_num)
            total_loss += loss_msssim_per
            loss_dict['MSSSIM_Loss'] = loss_msssim_per
        if cfg.use_structure_loss and cri_ml1 is not None:
            loss_ml1_per = loss_ml1_total / (cfg.frame_num)
            loss_dict['ML1_Loss'] = loss_ml1_per
    if cfg.use_edge_loss:
        loss_edge_per = loss_edge_total / (cfg.frame_num)
        loss_dict['Sobel_Loss'] = loss_edge_per
        total_loss += loss_edge_per
    loss_dict['Total_Loss'] = total_loss
    # optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
    if cfg.use_realism:
        optimizer_G.step()
        if cfg.use_gan_loss:
            l_d_total = 0
            l_d_real_total = 0
            l_d_fake_total = 0
            pred_d_real_total = 0
            pred_d_fake_total = 0
            discriminator.train()
            for p in discriminator.parameters():
                p.requires_grad = True
            optimizer_D.zero_grad()
            for time_ind in range(cfg.frame_num):
                pred_d_real = discriminator(ftgt[time_ind])
                pred_d_fake = discriminator(fake[time_ind].detach())  # detach to avoid BP to G
                l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                l_d_total += (l_d_real + l_d_fake) / 2
                l_d_real_total += l_d_real  # .item()
                l_d_fake_total += l_d_fake  # .item()
                pred_d_real_total += torch.mean(pred_d_real.detach())
                pred_d_fake_total += torch.mean(pred_d_fake.detach())
            l_d_total /= cfg.frame_num
            l_d_real_total /= cfg.frame_num
            l_d_fake_total /= cfg.frame_num
            l_d_total.backward()
            optimizer_D.step()
            loss_dict['D_Loss_Real'] = l_d_real_total
            loss_dict['D_Loss_Fake'] = l_d_fake_total
            loss_dict['D_real'] = pred_d_real_total.detach() / cfg.frame_num
            loss_dict['D_fake'] = pred_d_fake_total.detach() / cfg.frame_num
    else:
        optimizer.step()

    text = 'Loss | '
    for k, v in loss_dict.items():
        text += '%s %.10f | ' % (k, v.item())
    print(text)

    del in_data, gt_raw_data
    if cfg.use_realism:
        return ft1, fgt, \
               refine_out, fusion_out, denoise_out, \
               omega, gamma, \
               real, loss_dict
    else:
        return ft1, fgt, \
               refine_out, fusion_out, denoise_out, \
               omega, gamma, \
               loss_dict


def evaluate(model, psnr, writer, iter):
    print('Evaluate...')
    cnt = 0
    total_psnr = 0
    total_psnr_raw = 0
    total_psnr_real = 0
    model.eval()
    pixel_shuffle = PixelShuffle(2)
    pixel_unshuffle = PixelShuffle(0.5)
    avg_pool = torch.nn.AvgPool2d(2, 2)

    with torch.no_grad():
        for scene_ind in cfg.val_list:
            # scene_ind = '{:0>4d}'.format(scene_ind)
            for noisy_level in range(0, len(cfg.iso_list)):
                in_data, gt_raw_data = load_eval_data(noisy_level, scene_ind)
                frame_psnr = 0
                frame_psnr_raw = 0
                frame_psnr_real = 0
                for time_ind in range(cfg.frame_num):
                    ft1 = in_data[:, time_ind: (time_ind + 1), :, :]
                    fgt = gt_raw_data[:, time_ind : (time_ind + 1), :, :]
                    fgt = pixel_unshuffle(fgt)  # (b,4,h/2,w/2)
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)
                    if time_ind == 0:
                        ft0_fusion = ft1
                    else:
                        ft0_fusion = ft0_fusion_data

                    # coeff_a = cfg.a_list[noisy_level] / (cfg.white_level - cfg.black_level)
                    coeff_a = cfg.a_list[noisy_level] #  / (cfg.white_level - cfg.black_level)
                    # coeff_b = cfg.b_list[noisy_level] / (cfg.white_level - cfg.black_level) ** 2
                    coeff_b = cfg.b_list[noisy_level] #  / (cfg.white_level - cfg.black_level) ** 2

                    if cfg.use_realism:
                        fusion_out, denoise_out, refine_out, omega, gamma, real = model(ft0_fusion, ft1, coeff_a, coeff_b)
                        for i in range(cfg.px_num - 1):
                            real = pixel_shuffle(real)
                        frame_psnr_real += psnr(real, fgt)
                    else:
                        fusion_out, denoise_out, refine_out, omega, gamma = model(ft0_fusion, ft1, coeff_a, coeff_b)
                    ft0_fusion_data = fusion_out

                    fusion_out = pixel_shuffle(fusion_out)
                    denoise_out = pixel_shuffle(denoise_out)
                    refine_out = pixel_shuffle(refine_out)
                    omega = pixel_shuffle(omega)
                    gamma = pixel_shuffle(gamma)

                    frame_psnr += psnr(refine_out, fgt)

                    tmp = ft1
                    for i in range(cfg.px_num-1):
                        tmp = pixel_shuffle(tmp)
                    tmp = avg_pool(tmp)
                    frame_psnr_raw += psnr(tmp, fgt)

                frame_psnr = frame_psnr / (cfg.frame_num)
                frame_psnr_raw = frame_psnr_raw / (cfg.frame_num)

                print('---------')
                print('Scene: ', scene_ind, # ('%02d' % scene_ind),
                      'Noisy_level: ', ('%02d' % noisy_level),
                      'PSNR: ', ('%.8f' % frame_psnr.item()),
                      'Raw PSNR: ',('%.8f' % frame_psnr_raw.item()))
                if cfg.use_realism:
                    frame_psnr_real /= (cfg.frame_num)
                    print('Real PSNR: ', ('%.8f' % frame_psnr_real.item()))
                    total_psnr_real += frame_psnr_real
                total_psnr += frame_psnr
                total_psnr_raw += frame_psnr_raw

                cnt += 1
                del in_data, gt_raw_data
        total_psnr = total_psnr / cnt
        total_psnr_raw = total_psnr_raw / cnt
        if cfg.use_realism:
            total_psnr_real /= cnt
    print('Eval  |  Refine PSNR  |   ', ('%.8f' % total_psnr.item()))
    writer.add_scalar('PSNR', total_psnr.item(), iter)
    writer.add_scalar('PSNR_RAW', total_psnr_raw.item(), iter)
    writer.add_scalar('PSNR_IMP', total_psnr.item() - total_psnr_raw.item(), iter)

    if cfg.use_realism:
        print('Photo Real PSNR  | ', ('%.8f' % total_psnr_real.item()))
        writer.add_scalar('PSNR_Photo_Real', total_psnr_real.item(), iter)
        torch.cuda.empty_cache()
        return total_psnr, total_psnr_raw, total_psnr_real
    else:
        torch.cuda.empty_cache()
        return total_psnr, total_psnr_raw


def extract_features(model, x, layers):
    features = []
    if torch.cuda.is_available() and cfg.ngpu > 1:
        model = model.module

    for index, layer in enumerate(model):
        x = layer(x)
        if index in layers:
            features.append(x)
    return features

def main():
    """
    Train, Valid, Write Log, Write Predict ,etc
    :return:
    """
    setup_seed(666)
    ##########################################################
    ## use gpu
    device = cfg.device
    ngpu = cfg.ngpu
    cudnn.benchmark = True
    ## tensorboard --logdir runs
    log_dir = cfg.log_dir
    writer = SummaryWriter(log_dir)
    ##########################################################
    checkpoint = cfg.checkpoint
    start_epoch = cfg.start_epoch
    start_iter = cfg.start_iter
    best_psnr = 0

    ## initialize model
    model = architecture.EMVD(cfg)
    
    ## compute GFLOPs
    # stat(model, (8,512,512))

    model = model.to(device)
    loss = netloss.L1Loss().to(device)
    L1Loss = nn.L1Loss().cuda()
    L2Loss = nn.MSELoss().cuda()
    psnr = netloss.PSNR().to(device)

    learning_rate = cfg.learning_rate
    optimizer = None
    ##########################################################
    pixel_shuffle = None
    pixel_unshuffle = None
    if cfg.use_pixel_shuffle:
        pixel_shuffle = PixelShuffle(2)
        pixel_unshuffle = PixelShuffle(0.5)
    ##########################################################
    cri_edge = None
    if cfg.use_edge_loss:
        cri_edge = netloss.EdgeLoss(kernel_type='sobel', weight=0.1, c=4).cuda()
    ##########################################################
    cri_msssim = None
    cri_ml1 = None
    cri_gan = None
    discriminator = None
    optimizer_D = None
    scheduler_D = None
    optimizer_G = None
    scheduler_G = None
    if cfg.freeze_emvd:
        for name, child in model.named_children():
            if 'realism' in name:
                continue
            for param in child.parameters():
                param.requires_grad = False
    if cfg.use_realism:
        if cfg.use_structure_loss:
            cri_msssim = pytorch_msssim.MS_SSIM(channel=4, data_range=1.0).cuda()
            cri_ml1 = netloss.MultiscaleL1Loss().cuda()
        if cfg.use_gan_loss:
            cri_gan = netloss.GANLoss('gan', 1.0, 0.0).cuda()
            discriminator = define_D().cuda()
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_D)
            scheduler_D = lr_scheduler.MultiStepLR(optimizer_D, cfg.lr_steps, cfg.lr_gamma)

        optimizer_G = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                     model.parameters()), lr=cfg.lr_G)
        scheduler_G = lr_scheduler.MultiStepLR(optimizer_G, cfg.lr_steps, cfg.lr_gamma)
    ##########################################################
    else:
        if cfg.use_regularization:
            optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad,
                                                        model.parameters()),
                                          lr=learning_rate,
                                          weight_decay=cfg.weight_decay)
        else:
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                                       model.parameters()),
                                         lr=learning_rate)

    loss_network = None
    isp = None
    if cfg.use_perceptual_loss or cfg.use_contextual_loss:
        # Loss network
        loss_network = torchvision.models.__dict__[cfg.vgg_flag](pretrained=False)
        state_dict = torch.load(cfg.vgg_path)
        loss_network.load_state_dict(state_dict)
        loss_network.training = False
        for param in loss_network.parameters():
            param.requires_grad = False
        loss_network = loss_network.features.to(device)
    if cfg.use_isp:
        isp = models.ISP()
        isp_state_dict = torch.load('utils/models/ISP.pth')
        isp.load_state_dict(isp_state_dict)

        isp.training = False
        for param in isp.parameters():
            param.requires_grad = False
        isp = isp.to(device)

    ## load pretrained model
    if checkpoint is not None:
        print('--- Loading Pretrained Model ---')
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        state_dict = checkpoint['model']
        if ngpu > 1:
            temp = collections.OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    n = '.'.join(k.split('.')[1:])
                    temp[n] = v
            state_dict = temp
        if cfg.use_realism:
            if 'optimizer_G' in checkpoint.keys():
                optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                model.load_state_dict(state_dict)
            if cfg.use_gan_loss and 'optimizer_D' in checkpoint.keys():
                optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            else:
                model.load_state_dict(state_dict, strict=False)
                start_iter = 0
                start_epoch = 0
        else:
            model.load_state_dict(state_dict, strict=True)

        optimizer.load_state_dict(checkpoint['optimizer'])

    iter = start_iter

    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))
        if cfg.use_perceptual_loss or cfg.use_contextual_loss:
            loss_network = nn.DataParallel(loss_network, device_ids=list(range(ngpu)))
        if cfg.use_isp:
            isp = nn.DataParallel(isp, device_ids=list(range(ngpu)))

    shutil.copy('config.py', log_dir)
    shutil.copy('./arch/architecture.py', log_dir)
    shutil.copy('train.py', log_dir)
    shutil.copy('./utils/netloss.py', log_dir)
    shutil.copy('./dataset/load_data.py', log_dir)

    train_data_name_queue = generate_file_list(cfg.train_list)
    train_dataset = loadImgs(train_data_name_queue)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    for epoch in range(start_epoch, cfg.epoch):
        print('------------------------------------------------')
        print('Epoch                |   ', ('%08d' % epoch))
        start = time.time()
        for i, (input, label, noisy_level) in enumerate(train_loader):
            print('------------------------------------------------')
            print('Iter                 |   ', ('%08d' % iter))
                  # ,'             data loader time, ',time.time() - start)
            start = time.time()
            in_data = input.to(device) # .permute(0, 3, 1, 2).to(device)
            gt_raw_data = label.to(device) # .permute(0, 3, 1, 2).to(device)
            outputs = train(in_data,
                            gt_raw_data,
                            noisy_level,
                            model,
                            loss,
                            device,
                            optimizer,
                            loss_network=loss_network,
                            isp=isp,
                            cri_msssim=cri_msssim,
                            cri_ml1=cri_ml1,
                            cri_gan=cri_gan,
                            discriminator=discriminator,
                            optimizer_D=optimizer_D,
                            optimizer_G=optimizer_G,
                            cri_edge=cri_edge,
                            pixel_shuffle=pixel_shuffle,
                            pixel_unshuffle=pixel_unshuffle
                            )
            if cfg.use_realism:
                ft1, fgt, refine_out, fusion_out, denoise_out, omega, gamma, \
                real, loss_dict = outputs
            else:
                ft1, fgt, refine_out, fusion_out, denoise_out, omega, gamma, \
                loss_dict = outputs
            iter = iter + 1
            if iter % cfg.log_step == 0:
                input_gray = torch.mean(ft1, 1, True)
                label_gray = torch.mean(fgt, 1, True)
                predict_gray = torch.mean(refine_out, 1, True)

                writer.add_image('input', make_grid(input_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('label', make_grid(label_gray.cpu(), nrow=4, normalize=True), iter)
                writer.add_image('refine_out', make_grid(predict_gray.cpu(), nrow=4, normalize=True), iter)
                if cfg.use_realism:
                    real_gray = torch.mean(real, 1, True)
                    writer.add_image('real', make_grid(real_gray.cpu(), nrow=4, normalize=True), iter)
                else:
                    fusion_gray = torch.mean(fusion_out, 1, True)
                    denoise_gray = torch.mean(denoise_out, 1, True)
                    writer.add_image('fusion_out', make_grid(fusion_gray.cpu(), nrow=4, normalize=True), iter)
                    writer.add_image('denoise_out', make_grid(denoise_gray.cpu(), nrow=4, normalize=True), iter)
                    gamma_gray = torch.mean(gamma, 1, True)
                    omega_gray = torch.mean(omega, 1, True)
                    writer.add_image('gamma', make_grid(gamma_gray.cpu(), nrow=4, normalize=True), iter)
                    writer.add_image('omega', make_grid(omega_gray.cpu(), nrow=4, normalize=True), iter)
                for k, v in loss_dict.items():
                    writer.add_scalar(k, v.item(), iter)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                obj = {'epoch': epoch,
                       'iter': iter,
                       'model': model.state_dict()}
                if cfg.use_realism:
                    obj['optimizer_G'] = optimizer_G.state_dict()
                    if cfg.use_gan_loss:
                        obj['optimizer_D'] = optimizer_D.state_dict()
                else:
                    obj['optimizer'] = optimizer.state_dict()
                torch.save(obj, cfg.model_save_root)

            if iter % cfg.valid_step == 0 and iter > cfg.valid_start_iter:
                if cfg.use_realism:
                    eval_psnr, eval_psnr_raw, eval_psnr_real = evaluate(model, psnr, writer, iter)
                    eval_psnr = eval_psnr_real
                else:
                    eval_psnr, eval_psnr_raw = evaluate(model, psnr, writer, iter)
                if eval_psnr > best_psnr:
                    best_psnr = eval_psnr
                    obj = {'epoch': epoch,
                           'iter': iter,
                           'model': model.state_dict(),
                           'best_psnr': best_psnr}
                    if cfg.use_realism:
                        obj['optimizer_G'] = optimizer_G.state_dict()
                        if cfg.use_gan_loss:
                            obj['optimizer_D'] = optimizer_D.state_dict()
                    else:
                        obj['optimizer'] = optimizer.state_dict()
                    torch.save(obj, cfg.best_model_save_root)
            # print('model time,  ', time.time() - start)
            start = time.time()

        if cfg.use_realism:
            scheduler_G.step()
            if cfg.use_gan_loss:
                scheduler_D.step()
    writer.close()


if __name__ == '__main__':
    # initialize()
    main()
