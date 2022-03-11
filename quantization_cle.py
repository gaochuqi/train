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
This file demonstrates the use of quantization using AIMET Cross Layer Equalization (CLE)
and Bias Correction (BC) technique.
"""

import argparse
import logging
import os
from datetime import datetime
from functools import partial
import torch
import torch.utils.data as torch_data
from torchvision import models
import collections

# imports for AIMET
import aimet_common
from aimet_torch import bias_correction
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantParams, QuantizationSimModel

# imports for data pipelines
import config as cfg
import architecture_reparam
from load_data import *
import netloss

from Examples.common import image_net_config
from Examples.torch.utils.image_net_data_loader import ImageNetDataLoader
from Examples.torch.utils.image_net_evaluator import ImageNetEvaluator

logger = logging.getLogger('TorchCLE-BC')
formatter = logging.Formatter('%(asctime)s : %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(format=formatter)


###
# This script utilizes AIMET to apply Cross Layer Equalization and Bias Correction on a resnet18
# pretrained model with the ImageNet data set. It should re-create the same performance numbers
# as published in the AIMET release for the particular scenario as described below.

# Scenario parameters:
#    - AIMET quantization accuracy using simulation model
#       - Quant Scheme: 'tf_enhanced'
#       - rounding_mode: 'nearest'
#       - default_output_bw: 8, default_param_bw: 8
#       - Encoding computation using 5 batches of data
#    - AIMET Bias Correction
#       - Quant Scheme: 'tf_enhanced'
#       - rounding_mode: 'nearest'
#       - num_quant_samples: 16
#       - num_bias_correct_samples: 16
#       - ops_to_ignore: None
#    - Input shape: [1, 3, 224, 224]
###

def calculate_quantsim_accuracy(model: torch.nn.Module, evaluator: aimet_common.defs.EvalFunction,
                                use_cuda: bool = False) -> float:
    """
    Calculates quantized model accuracy (INT8) using AIMET QuantizationSim

    :param model: the loaded model
    :param evaluator: the Eval function to use for evaluation
    :param use_cuda: True, if model is placed on GPU
    :return: quantized accuracy of model
    """
    input_shape = (1, 16,
                   cfg.image_height,
                   cfg.image_width,)
    if use_cuda:
        ft0 = torch.rand(input_shape).cuda() /4
        ft1 = torch.rand(input_shape).cuda() /4
        a = torch.rand((1,1,1,1)).cuda()
        b = torch.rand((1,1,1,1)).cuda()
    else:
        ft0 = torch.rand(input_shape) /4
        ft1 = torch.rand(input_shape) /4
        a = torch.rand((1,1,1,1))
        b = torch.rand((1,1,1,1))

    quantsim = QuantizationSimModel(model=model, quant_scheme='tf_enhanced',
                                    dummy_input=(ft0,ft1,a,b), rounding_mode='nearest',
                                    default_output_bw=8, default_param_bw=8, in_place=True)

    quantsim.compute_encodings(forward_pass_callback=partial(evaluator),
                               forward_pass_callback_args=None)

    accuracy = evaluator(quantsim.model, use_cuda=use_cuda)

    return accuracy


def apply_cross_layer_equalization(model: torch.nn.Module, input_shape: tuple) -> object:
    """
    Applies CLE on the model and calculates model accuracy on quantized simulator
    Applying CLE on the model inplace consists of:
        - Batch Norm Folding
        - Converts any ReLU6 layers to ReLU layers
        - Cross Layer Scaling
        - High Bias Fold
        - Converts any ReLU6 into ReLU

    :param model: the loaded model
    :param input_shape: the shape of the input to the model
    :return:
    """
    equalize_model(model, input_shape)


def apply_bias_correction(model: torch.nn.Module, data_loader: torch_data.DataLoader):
    """
    Applies Bias-Correction on the model.
    :param model: The model to quantize
    :param evaluator: Evaluator used during quantization
    :param dataloader: DataLoader used during quantization
    :param logdir: Log directory used for storing log files
    :return: None
    """
    # Rounding mode can be 'nearest' or 'stochastic'
    rounding_mode = 'nearest'

    # Number of samples used during quantization
    num_quant_samples = 16

    # Number of samples used for bias correction
    num_bias_correct_samples = 16

    params = QuantParams(weight_bw=8,
                         act_bw=8,
                         round_mode=rounding_mode,
                         quant_scheme='tf_enhanced')

    # Perform Bias Correction
    bias_correction.correct_bias(model.to(device="cuda"),
                                 params,
                                 num_quant_samples=num_quant_samples,
                                 data_loader=data_loader,
                                 num_bias_correct_samples=num_bias_correct_samples)

def load_weight():
    checkpoint = torch.load('./%s/model.pth' % (cfg.model_name))
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

def evaluate(model, use_cuda=None):
    psnr = netloss.PSNR().to(cfg.device)
    pixel_shuffle = architecture_reparam.PixelShuffle(2)
    pixel_unshuffle = architecture_reparam.PixelShuffle(0.5)
    subfolder = os.path.join('log', cfg.model_name)
    bin_np = np.fromfile('%s/bin.raw' % (subfolder), dtype=np.float32)
    bin_np = np.reshape(bin_np, (16, 64, 1, 1))
    bin_w = torch.from_numpy(bin_np).to(cfg.device)
    conv_bin = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False).to(cfg.device)
    conv_bin.weight = torch.nn.Parameter(bin_w, requires_grad=False)
    print('Evaluate...')
    cnt = 0
    total_psnr = 0
    model.eval()
    with torch.no_grad():
        for scene_ind in range(7, 11+1):
            for noisy_level in range(0, 5):
                in_data, gt_raw_data = load_eval_data(noisy_level, scene_ind)
                frame_psnr = 0
                for time_ind in range(cfg.frame_num):
                    ft1 = in_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]
                    fgt = gt_raw_data[:, time_ind: (time_ind + 1), :, :]
                    fgt = pixel_unshuffle(fgt)
                    for i in range(cfg.px_num):
                        ft1 = pixel_unshuffle(ft1)
                    ft1 = conv_bin(ft1)
                    if time_ind == 0:
                        ft0_fusion = ft1
                    else:
                        ft0_fusion = ft0_fusion_data

                    coeff_a = a_list[noisy_level] / (2 ** 12 - 1 - 240)
                    coeff_b = b_list[noisy_level] / (2 ** 12 - 1 - 240) ** 2

                    fusion_out, denoise_out, refine_out, omega, gamma = model(ft0_fusion, ft1, coeff_a, coeff_b)

                    ft0_fusion_data = fusion_out
                    for i in range(cfg.px_num - 1):
                        fusion_out = pixel_shuffle(fusion_out)
                        denoise_out = pixel_shuffle(denoise_out)
                        refine_out = pixel_shuffle(refine_out)
                        omega = pixel_shuffle(omega)
                        gamma = pixel_shuffle(gamma)

                    frame_psnr += psnr(refine_out, fgt)

                frame_psnr = frame_psnr / (cfg.frame_num)
                # print('---------')
                # print('Scene: ', ('%02d' % scene_ind), 'Noisy_level: ', ('%02d' % noisy_level), 'PSNR: ',
                #       '%.8f' % frame_psnr.item())
                total_psnr += frame_psnr
                cnt += 1
                del in_data, gt_raw_data
        total_psnr = total_psnr / cnt
    print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
    torch.cuda.empty_cache()
    return total_psnr

def cle_bc_example(config: argparse.Namespace):
    """
    Example code that shows the following
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
    eval_data_name_queue = generate_file_list(['7', '8', '9', '10', '11'])
    eval_dataset = loadImgs(eval_data_name_queue)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.num_workers,
                                              shuffle=True,
                                              pin_memory=True)

    # Load the pretrained model
    model = architecture_reparam.EMVD(cfg)
    state_dict = load_weight()
    model.load_state_dict(state_dict)
    if config.use_cuda:
        model.to(torch.device('cuda'))
    model = model.eval()

    # Calculate FP32 accuracy
    eval_psnr = evaluate(model)
    logger.info("Original Model PSNR = %.8f", eval_psnr)
    input_shape = (1, 16,
                   cfg.test_height,
                   cfg.test_width)
    # Applying cross-layer equalization (CLE)
    # Note that this API will equalize the model in-place
    apply_cross_layer_equalization(model=model, input_shape=(input_shape,
                                                             input_shape,
                                                             1,1))

    # Calculate quantized (INT8) accuracy after CLE
    accuracy = calculate_quantsim_accuracy(model=model, evaluator=evaluate, use_cuda=config.use_cuda)
    logger.info("Quantized (INT8) Model PSNR After CLE = %.8f", accuracy)

    # Save the quantized model
    torch.save(model.state_dict(), os.path.join(cfg.model_name, 'model_cle.pth'))

    logger.info("Cross Layer Equalization (CLE) complete")

def main():
    default_logdir = os.path.join("benchmark_output", "CLE" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    parser = argparse.ArgumentParser(description='Apply Cross Layer Equalization on pretrained '
                                                 'model and evaluate on dataset')

    # parser.add_argument('--dataset_dir', type=str,
    #                     # required=True,
    #                     default=cfg.crvd,
    #                     help="Path to a directory containing dataset.\n\
    #                           This folder should conatin at least 2 subfolders:\n\
    #                           'train': for training dataset and 'val': for validation dataset")

    parser.add_argument('--use_cuda', action='store_true',
                        # required=True,
                        default=True,
                        help='Add this flag to run the test on GPU.')

    parser.add_argument('--logdir', type=str,
                        default=default_logdir,
                        help="Path to a directory for logging. "
                             "Default value is 'benchmark_output/weight_svd_<Y-m-d-H-M-S>'")

    _config = parser.parse_args()

    os.makedirs(_config.logdir, exist_ok=True)

    fileHandler = logging.FileHandler(os.path.join(_config.logdir, "test.log"))
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    if _config.use_cuda and not torch.cuda.is_available():
        logger.error('use_cuda is selected but no cuda device found.')
        raise RuntimeError("Found no CUDA Device while use_cuda is selected")

    cle_bc_example(_config)


if __name__ == '__main__':
    main()
