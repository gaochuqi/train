import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
print('torch version,           ', torch.__version__)
print('torch cuda is available, ', torch.cuda.is_available())
print('torch cuda device count, ', torch.cuda.device_count())

device = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda')

# gpu
ngpu = 1
batch_size = 16 # 32 # 64
num_workers = 4 # 8 # 16
image_height = 128  # 256 #
image_width = 128  # 256 #
start_epoch = 0
start_iter = 0
# parameter of train
learning_rate = 1e-4  # 0.0001
epoch = int(1.5e4)
##################################################################################
version = os.getcwd().split('/')[-1] # 'vd_v000015'
model_name = 'model' # 'test' #
use_fusion_loss = True
use_denoise_loss = True
if use_fusion_loss:
    model_name += '_fuLoss'
if use_denoise_loss:
    model_name += '_deLoss'
use_sigma = True
use_pixel_shuffle = True
px_num = 3 if use_pixel_shuffle else 1
use_regularization = True
weight_decay = 0.3 # 0.01

##################################################################################
# log
log_dir = '/home/work/ssd1/proj/denoise/emvd/log/' \
          '{}/{}'.format(version, model_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_step = 10  # save the train log per log_step
# model store
model_save_root = os.path.join(log_dir, 'model.pth')
best_model_save_root = os.path.join(log_dir, 'model_best.pth')
# model_qua_arch_root = os.path.join(model_name, 'model_qua_arch.pth')
# pretrained model path
checkpoint = None if not os.path.exists(model_save_root) else model_save_root
# validation
valid_start_iter = 500
valid_step = 50
log_step = 10  # save the train log per log_step
vis_data = 1  # whether to visualize noisy and gt data

# generate data
data_root='/home/work/ssd1/dataset/DRV1to50/trainval_v3'
# data_root = '/home/wen/Documents/dataset/DRV/trainval_v3'
height = 3672
width = 5496

bayer_pattern = 'GBRG'  # 'RGGB'  #  'GRBG' # 'BGGR'
black_level = 64  # 800
white_level = 2 ** 10 - 1  # 16380
##################################################################################
# Mi11Ultra calibrate
a_list = [0.02886277, 0.05565189, 0.109696, 0.21690887, 0.43752982, 0.89052248, 1.88346616]
b_list = [0.31526701, 0.49276135, 0.30756074, 0.50924099, 1.1088187, 3.12988574, 8.53779323]
iso_list = [50, 100, 200, 400, 800, 1600, 3200]
# # CRVD fit curve # get_CRVD_coef_a_b
# iso_list = [50, 100, 200, 400, 800, 1600, 3200]
# a_list = [0.2879409, 0.3933852, 0.6042123, 1.0256209, 1.8674616, 3.5472803, 6.8918157]
# b_list = [1.3714607, 1.5073164, 1.8218268, 2.621904, 4.9051676, 12.195202, 37.59787]
# # 2022.02.22 tang
# ISO [50, 100, 200, 400, 800, 1600, 3200]
# a [0.029, 0.056, 0.14, 0.24, 0.47, 0.99, 1.91]
# b [0.74, 0.82, 0.71, 0.61, 0.98, 1.73, 3.12]
# # CRVD
# iso_list = [1600, 3200, 6400, 12800, 25600]
# a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
# b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
##################################################################################
frame_list = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1,
              0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1,
              0, 1, 2, 3, 4, 5, 6, 7]
frame_list_eval = [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1,
                   0, 1, 2, 3]
frame_num = len(frame_list) - 8 # 25
##################################################################################
# DRV 0001 - 0050
train_list = [1, 2, 3, 6, 7, 9, 12, 13, 14, 15,
              16, 17, 18, 19, 20, 21, 22, 24, 25, 27,
              28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
              39, 41, 42, 43, 45, 46, 47, 48, 49, 50]
val_list = [4, 5, 8, 10, 11, 23, 26, 38, 40, 44]
noisy_num = 3
obj_motion = [1, 2, 4, 5, 7, 8, 9, 11, 16, 17, 18, 19, 27, 28, 30, 33, 36, 39, 40, 41, 42, 44]
# # just camera motion
# train_list = [3, 6, 12, 13, 14, 15, 20, 21, 22, 23, 25, 29, 31, 32, 34, 35, 37, 43, 45, 46, 47, 48, 49, 50]
# val_list = [10, 24, 26, 38]
'''
camera_motion,   rotate       ['0014', '0023', '0026', '0037', '0047']
camera_motion,   translate    ['0022']
camera_motion,   scale        ['0021', '0029', '0043']
camera_motion,   perspective  ['0003', '0006', '0015', '0024', '0025', '0035', '0038', '0048', '0050']
camera_motion,   all          ['0010', '0012', '0013', '0020', '0031', '0032', '0034', '0045', '0046', '0049']
'''
# # 2022.02.18
# # minimal parts
# train_list = [1, 2, 3, 6, 7, 8, 9, 11]
# val_list = [4, 5, 8, 10]
# just object motion
# train_list = [1,  2,  4,  7,  9, 16, 17, 18, 19, 27, 28, 30, 33, 36, 39, 41, 42]
# val_list = [5, 8, 11, 40, 44]
##################################################################################
##################################################################################
use_isp = False
use_realism = False
freeze_emvd = False
use_gan_loss = False
use_edge_loss = False
use_ecb = False
use_attFeatFusion = False
######################################################################
# VGG
######################################################################
# vgg_flag = 'vgg19'
vgg_flag = 'vgg16'
# vgg_path = 'vgg19-dcbb9e9d.pth'
vgg_path = './vgg16-397923af.pth'
# [3, 8, 15, 22]
# content_layers = [15]  # 细节保留好，但稍稍颜色失真
# content_layers = [12] # [8] 
content_layers = [28]
max_1d_size = 48

lr_G = 2e-4
lr_D = 2e-4
lr_scheme = 'MultiStepLR'
lr_steps = [200, 300, 400, 500]
lr_gamma = 0.5
lambda_gan = 5e-3
######################################################################
# Perceptual Losses
######################################################################
use_perceptual_loss = False
lambda_perceptual = 0.01
# total variation regularizer
use_tvr = False
######################################################################
# Contextual Loss
######################################################################
use_contextual_loss = False
######################################################################
qat_test = False
######################################################################
use_real_L1 = False
use_structure_loss = False

test_height = 384
test_width = 512
image_channels = 4  # 8

channels = 4

VALID_VALUE = 959


# clip threshold
image_min_value = 0
image_max_value = 1
image_norm_value = 1

label_min_value = 0
label_max_value = 255
label_norm_value = 1

'''
# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
'''
'''
We train the networks using Adam optimizer [24] with 
batch size 16 and 
initial learning rate 1e-4. 
We apply a piece-wise constant decay which reduces
the learning rate by a factor of 10 every 100000 iterations.
All models are trained for an initial 300000 iterations on the
CRVD and SRVD dataset, and then fine-tuned for an addi-
tional 300000 iterations on CRVD only.
'''
'''
Channel Attention Is All You Need for Video Frame Interpolation 
	- ResGroup
		CA module
	- L = λ 1 L r + λ 2 L p	其中 λ 1 = 0.9 and λ 2 = 0.005
		where φ(·) is a function to extract conv5 4 features from the VGG-19 model pretrained on ImageNet dataset (Simonyan and Zisserman 2014b). 
		As noted in (Zhang et al. 2018a), the perceptual loss greatly helps in synthesizing realistic frames.

'''
