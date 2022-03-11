# =============================================================================
#
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
#
# =============================================================================
"""
This file demonstrates the use of quantization using AIMET Adaround
technique.
"""
import sys
import argparse
import copy
import logging
import os
from datetime import datetime
from functools import partial
from typing import Tuple
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import Dataset
from torch.autograd import Variable
import collections
import glob
import numpy as np
import cv2
import time
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import shutil


# imports for AIMET
import aimet_common
from aimet_common.defs import QuantScheme
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.quantsim import QuantizationSimModel

# imports for data pipelines
from aimet.common import image_net_config
from aimet.torch.utils.image_net_data_loader import ImageNetDataLoader
from aimet.torch.utils.image_net_evaluator import ImageNetEvaluator

import config as cfg
from arch import architecture_reparam
from arch.modules import PixelShuffle
from generate.tools import binning_raw
from reparam import reparameters

logger = logging.getLogger('TorchAdaround')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)

###
# This script utilizes AIMET to apply Adaround on a resnet18 pretrained model with
# the ImageNet data set. It should re-create the same performance numbers as
# published in the AIMET release for the particular scenario as described below.

# Scenario parameters:
#    - AIMET quantization accuracy using simulation model
#       - Quant Scheme: 'tf_enhanced'
#       - rounding_mode: 'nearest'
#       - default_output_bw: 8, default_param_bw: 8
#       - Encoding compution with or without encodings file
#       - Encoding computation using 5 batches of data
#    - AIMET Adaround
#       - num of batches for adarounding: 5
#       - bitwidth for quantizing layer parameters: 4
#       - Quant Scheme: 'tf_enhanced'
#       - Remaining Parameters: default
#    - Input shape: [1, 3, 224, 224]
###

device = cfg.device
default_logdir = os.path.join(cfg.log_dir, 'log/{}/'.format(cfg.model_name),
                              "benchmark_output",
                              "adaround_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
if not os.path.exists(default_logdir):
    os.makedirs(default_logdir)
shutil.copy('quantization_adaRound.py', default_logdir)
shutil.copy('./arch/architecture_reparam.py', default_logdir)

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
scene_list = [1]
frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3]
ch = 16  # 64
input_shape = (1, ch,
               cfg.test_height,
               cfg.test_width)
log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)


def evaluate(model=None, eval_loader=None):
    cfg.model_name = 'model_fuLoss_deLoss'  # 'model_deLoss' # 'model' #
    bin_np = np.load('{}/binning.npy'.format(log_dir)).astype(np.float32)
    bin_w = torch.from_numpy(bin_np)
    binnings = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
    binnings.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    conv_bin = binnings.to(device)
    # scene_list.sort(reverse=True)
    scene_avg_raw_psnr = 0
    scene_avg_raw_ssim = 0
    log_path = default_logdir

    f = open(log_path + '/{}_test_psnr_and_ssim_{}.txt'.format('adaRound',
                                                               time.strftime("%Y-%m-%d_%H:%M:%S",
                                                                             time.localtime())), 'w')
    with torch.no_grad():
        for scene_ind in scene_list:
            iso_average_raw_psnr = 0
            iso_average_raw_ssim = 0
            for noisy_level_ind, iso_ind in enumerate(iso_list):
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
                    fgt = pixel_unshuffle(torch.from_numpy(
                        np.expand_dims(np.expand_dims(raw_gt_bin, axis=0), axis=0)))  # (b,4,h/2,w/2)
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
                    ########################################################
                    # run model
                    ########################################################
                    tmp = np.pad(raw_noisy, [(0, 0), (8, 8)])
                    noisy = torch.from_numpy(np.expand_dims(np.expand_dims(tmp,
                                                                           axis=0), axis=0)).cuda()
                    ft1 = noisy
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)

                    if idx == 1:
                        ft0 = ft1

                    coeff_a = a_list[noisy_level_ind] / (cfg.white_level - cfg.black_level)
                    coeff_b = b_list[noisy_level_ind] / (cfg.white_level - cfg.black_level) ** 2

                    if ft0.shape == ft1.shape:
                        ft0 = conv_bin(ft0)
                    ft1 = conv_bin(ft1)

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

                    fusion_np = fusion.data.cpu().numpy()  # h,w
                    denoise_np = denoise.data.cpu().numpy()
                    refine_np = refine.data.cpu().numpy()

                    fout = pixel_unshuffle(
                        torch.from_numpy(np.expand_dims(np.expand_dims(refine_np, axis=0), axis=0)))  # (b,4,h/2,w/2)

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
                    test_raw_ssim /= 4
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
    total_psnr = scene_avg_raw_psnr
    print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr))
    torch.cuda.empty_cache()
    return total_psnr


def apply_adaround_and_find_quantized_accuracy(model: torch.nn.Module,
                                               evaluator: aimet_common.defs.EvalFunction,
                                               data_loader: torch_data.DataLoader,
                                               use_cuda: bool = False,
                                               logdir: str = '') -> Tuple[torch.nn.Module, float]:
    """
    Quantizes the model using AIMET's adaround feature, and saves the model.

    :param model: The loaded model
    :param evaluator: The Eval function to use for evaluation
    :param dataloader: The dataloader to be passed into the AdaroundParameters api
    :param num_val_samples_per_class: No of samples to use from every class in
                                      computing encodings. Not used in pascal voc
                                      dataset
    :param use_cuda: The cuda device.
    :param logdir: Path to a directory for logging.
    :return: A tuple of quantized model and accuracy of model on this quantsim
    """
    # bn_folded_model = copy.deepcopy(model)

    from dill import dumps, loads
    bn_folded_model = loads(dumps(model))

    _ = fold_all_batch_norms(bn_folded_model, input_shapes=[input_shape,
                                                            input_shape,
                                                            1, 1])

    if use_cuda:
        ft0 = torch.rand(input_shape).cuda()
        ft1 = torch.rand(input_shape).cuda()
        a = torch.rand((1, 1, 1, 1)).cuda()
        b = torch.rand((1, 1, 1, 1)).cuda()
    else:
        ft0 = torch.rand(input_shape)
        ft1 = torch.rand(input_shape)
        a = torch.rand((1, 1, 1, 1))
        b = torch.rand((1, 1, 1, 1))
    ######################################################################
    # QuantizationSimModel
    ######################################################################
    quantsim = QuantizationSimModel(model=bn_folded_model, dummy_input=(ft0, ft1, a, b),
                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest',
                                    default_output_bw=8,
                                    default_param_bw=8,
                                    in_place=False)

    # Set and freeze parameter encodings. These encodings are associated with the Adarounded parameters.
    # This will make sure compute_encodings() doesn't alter the parameter encodings.
    quantsim.compute_encodings(forward_pass_callback=partial(evaluator),
                               forward_pass_callback_args=None)
    quantsim.export(path=logdir, filename_prefix='adaround_model', dummy_input=(ft0.cpu(), ft1.cpu(),
                                                                                a.cpu(), b.cpu()))
    print('evaluate quantsim model ---------------------------------------------')
    psnr = evaluator(quantsim.model)
    logger.info("Before applying Adaround, PSNR = %.8f", psnr)
    ######################################################################
    # Adaround QuantizationSimModel
    ######################################################################
    params = AdaroundParameters(data_loader=data_loader, num_batches=len(frame_list))  # 5)
    ada_model = Adaround.apply_adaround(bn_folded_model,
                                        dummy_input=(ft0, ft1, a, b),
                                        params=params,
                                        path=logdir,
                                        filename_prefix='adaround',
                                        default_param_bw=8,
                                        default_quant_scheme=QuantScheme.post_training_tf_enhanced)

    quantsim = QuantizationSimModel(model=ada_model, dummy_input=(ft0, ft1, a, b),
                                    quant_scheme=QuantScheme.post_training_tf_enhanced,
                                    rounding_mode='nearest',
                                    default_output_bw=8,
                                    default_param_bw=8,
                                    in_place=False)

    # Set and freeze parameter encodings. These encodings are associated with the Adarounded parameters.
    # This will make sure compute_encodings() doesn't alter the parameter encodings.
    quantsim.set_and_freeze_param_encodings(encoding_path=os.path.join(logdir, 'adaround.encodings'))
    quantsim.compute_encodings(forward_pass_callback=partial(evaluator),
                               forward_pass_callback_args=None)
    quantsim.export(path=logdir, filename_prefix='adaround_model', dummy_input=(ft0.cpu(), ft1.cpu(),
                                                                                a.cpu(), b.cpu()))
    print('evaluate adaround quantsim model ---------------------------------------------')
    psnr = evaluator(quantsim.model)

    return psnr


def decode_data(data_name):
    # data_name : 'videoType,{},scene,{},frame,{},iso,{}'
    _, video_type, _, scene_ind, _, frame_ind, _, iso_ind = data_name.split(',')
    scene_ind = int(scene_ind)
    noisy_level_ind = iso_list.index(int(iso_ind))

    gt_name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_gt{}.npy'.format(scene_ind, iso_ind,
                                                                    iso_ind, scene_ind,
                                                                    frame_ind, 400)
    gt_name = os.path.join(data_dir, gt_name)
    raw_gt = np.load(gt_name, mmap_mode='r')
    raw_gt = np.pad(raw_gt, [(0, 0), (8, 8)])
    raw_gt = raw_gt.astype(np.float32)
    fgt = binning_raw(raw_gt)

    noisy_frame_index_for_current = np.random.randint(0, 10)
    name = 'scene{}/ISO{}/ISO{}_scene{}_frame{}_noisy{}.npy'.format(scene_ind, iso_ind,
                                                                    iso_ind, scene_ind,
                                                                    frame_ind,
                                                                    noisy_frame_index_for_current)
    noisy_name = os.path.join(data_dir, name)
    raw_noisy = np.load(noisy_name, mmap_mode='r')
    raw_noisy = np.pad(raw_noisy, [(0, 0), (8, 8)])
    raw_noisy = raw_noisy.astype(np.float32)

    a = a_list[noisy_level_ind]
    b = b_list[noisy_level_ind]
    a = a / (cfg.white_level - cfg.black_level)
    b = b / (cfg.white_level - cfg.black_level) ** 2
    a = np.reshape(np.array(a, dtype=np.float32), (1, 1, 1))
    b = np.reshape(np.array(b, dtype=np.float32), (1, 1, 1))

    test_noisy_psnr = compare_psnr(raw_gt, raw_noisy, data_range=1.0)
    # print(name, test_noisy_psnr)
    return raw_noisy, fgt, a, b


def generate_file_list(scene_list):
    file_num = 0
    data_name = []
    for scene_ind in scene_list:
        video_type = 'None'
        for iso in iso_list:
            for frame_ind in frame_list:
                gt_name = 'videoType,{},scene,{},frame,{},iso,{}'.format(video_type, scene_ind, frame_ind, iso)
                data_name.append(gt_name)
                file_num += 1
    return data_name
    # random_index = np.random.permutation(file_num)
    # data_random_list = []
    # for i, idx in enumerate(random_index):
    #     data_random_list.append(data_name[idx])
    # return data_random_list


class loadImgs(Dataset):
    def __init__(self, filelist, model_name):
        self.filelist = filelist
        # Load the pretrained model
        model = architecture_reparam.EMVD(cfg)
        state_dict = torch.load(os.path.join(log_dir, '{}.pth'.format(model_name + '_reparam')))
        model.load_state_dict(state_dict, strict=True)
        self.model = model.eval()
        bin_np = np.load('{}/binning.npy'.format(log_dir)).astype(np.float32)
        bin_w = torch.from_numpy(bin_np)
        self.binnings = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.binnings.weight = torch.nn.Parameter(bin_w, requires_grad=False)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        self.data_name = self.filelist[item]
        ft1, fgt, a, b = decode_data(self.data_name)
        ft1 = np.expand_dims(np.expand_dims(ft1, axis=0), axis=0)
        ft1 = torch.from_numpy(ft1)
        for i in range(cfg.px_num):
            ft1 = pixel_unshuffle(ft1)
        self.ft1 = self.binnings(ft1)
        fgt = np.expand_dims(np.expand_dims(fgt, axis=0), axis=0)
        fgt = torch.from_numpy(fgt)
        for i in range(cfg.px_num-1):
            fgt = pixel_unshuffle(fgt)
        self.fgt = fgt
        self.a = a
        self.b = b
        # print(item, self.data_name)
        return (self.ft1.detach().numpy()[0], self.ft1.detach().numpy()[0], self.a, self.b), self.fgt

        # if item % len(frame_list) == 0:
        #     self.ft0 = self.ft1
        #
        # fusion, _, refine, _, _ = self.model(self.ft0,
        #                                      self.ft1,
        #                                      torch.from_numpy(self.a),
        #                                      torch.from_numpy(self.b))
        # # refine = pixel_shuffle(refine)
        # # refine = pixel_shuffle(refine)
        # test_noisy_psnr = compare_psnr(fgt.data.cpu().numpy(),
        #                                refine.data.cpu().numpy(), # [0, 0].data.cpu().numpy(),
        #                                data_range=1.0)
        # print(item, test_noisy_psnr)
        # self.ft0 = fusion
        # return (self.ft0.detach().numpy()[0],
        #         self.ft1.detach().numpy()[0],
        #         self.a, self.b), \
        #        self.fgt


def adaround_example(config: argparse.Namespace):
    """
    1. Instantiates Data Pipeline for evaluation
    2. Loads the pretrained resnet18 Pytorch model
    3. Calculates Model accuracy
        3.1. Calculates floating point accuracy
        3.2. Calculates Quant Simulator accuracy
    4. Applies AIMET CLE and BC
        4.1. Applies AIMET CLE and calculates QuantSim accuracy
        4.2. Applies AIMET BC and calculates QuantSim accuracy

    :param config: This argparse.Namespace config expects following parameters:
                   tfrecord_dir: Path to a directory containing ImageNet TFRecords.
                                This folder should conatin files starting with:
                                'train*': for training records and 'validation*': for validation records
                   use_cuda: A boolean var to indicate to run the test on GPU.
                   logdir: Path to a directory for logging.
    """

    # Instantiate Data Pipeline for evaluation and training
    # data_pipeline = ImageNetDataPipeline(config)

    model_name = 'model'
    suffix = '_reparam'
    # convert_arch_qat(model_name)
    reparameters(model_name)
    # Load the pretrained model
    model = architecture_reparam.EMVD(cfg)
    model = model.to(device)
    # log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    state_dict = torch.load(os.path.join(log_dir, '{}.pth'.format(model_name + suffix)))
    model.load_state_dict(state_dict, strict=True)
    #########################################################################
    model = model.eval()
    #########################################################################
    from aimet_torch.model_validator.model_validator import ModelValidator
    # Output of ModelValidator.validate_model will be True if model is valid, False otherwise
    result = ModelValidator.validate_model(model, model_input=(torch.rand(input_shape).to(device),
                                                               torch.rand(input_shape).to(device),
                                                               torch.rand(1, 1, 1, 1).to(device),
                                                               torch.rand(1, 1, 1, 1).to(device)))
    print('# ModelValidator.validate_model result:  ', result)
    #########################################################################
    eval_data_name_queue = generate_file_list(scene_list)
    eval_dataset = loadImgs(eval_data_name_queue, model_name)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=1,  # cfg.batch_size,
                                              num_workers=1,  # cfg.num_workers,
                                              shuffle=False,
                                              pin_memory=True)
    #########################################################################
    # # Calculate FP32 accuracy
    # eval_psnr = evaluate(model)
    # logger.info("Original Model PSNR = %.8f", eval_psnr)
    # logger.info("Applying Adaround")

    # Applying Adaround
    # Optimally rounds the parameters of the model
    # data_loader = ImageNetDataLoader(is_training=False, images_dir=config.dataset_dir,
    #                                  image_size=image_net_config.dataset['image_size']).data_loader

    psnr = apply_adaround_and_find_quantized_accuracy(model=model,
                                                      evaluator=evaluate,
                                                      data_loader=eval_loader,
                                                      use_cuda=config.use_cuda,
                                                      logdir=config.logdir)

    logger.info("After applying Adaround, PSNR = %.8f", psnr)
    logger.info("Adaround Complete")


def test_data_loader():
    model_name = 'model'
    suffix = '_reparam'
    reparameters(model_name)
    # Load the pretrained model
    model = architecture_reparam.EMVD(cfg)
    model = model.to(device)
    # log_dir = os.path.join(cfg.log_dir, 'log', cfg.model_name)
    state_dict = torch.load(os.path.join(log_dir, '{}.pth'.format(model_name + suffix)))
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()
    eval_data_name_queue = generate_file_list(scene_list)
    eval_dataset = loadImgs(eval_data_name_queue, model_name)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=1,  # cfg.batch_size,
                                              num_workers=1,  # cfg.num_workers,
                                              shuffle=False,
                                              pin_memory=True)
    for i, (input, label) in enumerate(eval_loader):
        ft0, ft1, a, b = input
        fusion_out, denoise_out, refine_out, omega, gamma = model(ft0.to(device),
                                                                  ft1.to(device),
                                                                  a.to(device),
                                                                  b.to(device))
        refine = pixel_shuffle(refine_out)
        refine = pixel_shuffle(refine)
        test_noisy_psnr = compare_psnr(label[0].data.cpu().numpy(),
                                       refine[0, 0].data.cpu().numpy(),
                                       data_range=1.0)
        print(i, test_noisy_psnr)


def main():
    # test_data_loader()
    # return

    parser = argparse.ArgumentParser(
        description='Apply Adaround on pretrained ResNet18 model and evaluate on ImageNet dataset')

    # parser.add_argument('--dataset_dir', type=str,
    #                     required=True,
    #                     help="Path to a directory containing ImageNet dataset.\n\
    #                           This folder should conatin at least 2 subfolders:\n\
    #                           'train': for training dataset and 'val': for validation dataset")

    parser.add_argument('--use_cuda', action='store_true',
                        # required=True,
                        default=True,
                        help='Add this flag to run the test on GPU.')

    parser.add_argument('--logdir', type=str,
                        default=default_logdir,
                        help="Path to a directory for logging.\
                                  Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

    _config = parser.parse_args()

    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    adaround_example(_config)


if __name__ == '__main__':
    main()

'''
/home/wen/anaconda3/lib/python3.6/site-packages/aimet_common/x86_64-linux-gnu
2021-11-09 16:57:10,777 - root - INFO - AIMET
2021-11-09 16:57:12,814 - Utils - INFO - Running validator check <function validate_for_reused_modules at 0x7f94a2499268>
2021-11-09 16:57:12,827 - Utils - INFO - Running validator check <function validate_for_missing_modules at 0x7f94a2499378>
2021-11-09 16:57:12,943 - Utils - WARNING - Ops with missing modules: ['abs_41', 'Concat_42', 'rsub_49', 'Mul_50', 'Mul_51', 'Add_52', 'Concat_53', 'upsample_nearest2d_60', 'abs_70', 'Concat_71', 'rsub_78', 'Mul_79', 'Mul_80', 'Add_81', 'Concat_82', 'upsample_nearest2d_89', 'abs_99', 'Concat_100', 'rsub_107', 'Mul_108', 'Mul_109', 'Add_110', 'Concat_111', 'Concat_117', 'rsub_124', 'Mul_125', 'Mul_126', 'Add_127']
This can be due to several reasons:
1. There is no mapping for the op in ConnectedGraph.op_type_map. Add a mapping for ConnectedGraph to recognize and be able to map the op.
2. The op is defined as a functional in the forward function, instead of as a class module. Redefine the op as a class module if possible. Else, check 3.
3. This op is one that cannot be defined as a class module, but has not been added to ConnectedGraph.functional_ops. Add to continue.
2021-11-09 16:57:12,943 - Utils - INFO - The following validator checks failed:
2021-11-09 16:57:12,943 - Utils - INFO - 	<function validate_for_missing_modules at 0x7f94a2499378>
ModelValidator.validate_model result False

'''
