from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
# os.environ["CUDA_VISIBLE_DEVICE"]="4,5,6,7"
import cv2
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import collections

from arch import arch
from arch import arch_qat
from arch import architecture
from utils import netloss
from dataset.dataset import preprocess, tensor2numpy, pack_gbrg_raw, depack_gbrg_raw
import config as cfg

def test_big_size_raw(input_data,
					  block_size,
					  denoiser,
					  a, b):
	stack_image = input_data
	hgt = np.shape(stack_image)[1]
	wid = np.shape(stack_image)[2]

	border = 32

	expand_raw = np.zeros(shape=[1, int(hgt * 2.5), int(wid * 2.5), 4], dtype=np.float)


	expand_raw[:,0: hgt, 0:wid,:] = stack_image
	expand_raw[:,0: hgt, border * 2: wid + border * 2,:] = stack_image
	expand_raw[:,border * 2:hgt + border * 2, 0:0 + wid + 0,:] = stack_image
	expand_raw[:,border * 2:hgt + border * 2, 0 + border * 2:0 + wid + border * 2,:] = stack_image

	expand_raw[:,0 + border:0 + border + hgt, 0 + border:0 + border + wid,:] = stack_image

	expand_res = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)
	expand_fusion = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)
	expand_denoise = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)
	expand_gamma = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),1], dtype=np.float)
	expand_omega = np.zeros([1, int(hgt * 2.5), int(wid * 2.5), 1], dtype=np.float)
	if cfg.use_realism:
		expand_real = np.zeros([1,int(hgt * 2.5), int(wid * 2.5),4], dtype=np.float)

	'''process'''
	for i in range(0 + border, hgt + border, int(block_size)):
		index = '%.2f' % (float(i) / float(hgt + border) * 100)
		print('run model : ', index, '%')
		for j in range(0 + border, wid + border, int(block_size)):
			block = expand_raw[:,i - border:i + block_size + border, j - border:j + block_size + border,:]   # t frame input
			block = preprocess(block).float()
			input = block
			###############################################################################
			ft0_fusion = input # [:, 0:4, :, :]
			ft1 = input # [:, 4:8, :, :]

			###############################################################################
			with torch.no_grad():
				for i in range(cfg.px_num):
					ft1 = pixel_unshuffle(ft1)
				if cfg.use_realism:
					fusion_out, denoise_out, refine_out, \
					omega, gamma, \
					real_out  = denoiser(ft0_fusion, ft1, a, b)
				else:
					fusion_out, denoise_out, refine_out, omega, gamma = denoiser(ft0_fusion, ft1, a, b)
				for i in range(cfg.px_num-1):
					fusion_out = pixel_shuffle(fusion_out)
					denoise_out = pixel_shuffle(denoise_out)
					refine_out = pixel_shuffle(refine_out)
					omega = pixel_shuffle(omega)
					gamma = pixel_shuffle(gamma)
				if cfg.use_realism:
					for i in range(cfg.px_num):
						real_out = pixel_shuffle(real_out)
					real_out = tensor2numpy(real_out)
					expand_real[:, i:i + block_size, j:j + block_size, :] = real_out[:, border:-border,
																			border:-border, :]
					real_result = expand_real[:, border:hgt + border, border:wid + border, :]

				fusion_out = tensor2numpy(fusion_out)
				refine_out = tensor2numpy(refine_out)
				denoise_out = tensor2numpy(denoise_out)
				gamma_gray = torch.mean(gamma[:, 0:16, :, :], 1, True)
				omega_gray = torch.mean(omega[:, 0:16, :, :], 1, True)
				gamma = tensor2numpy(F.upsample(gamma_gray, scale_factor=2))
				omega = tensor2numpy(F.upsample(omega_gray, scale_factor=2))
				expand_res[:,i:i + block_size, j:j + block_size,:] = refine_out[:,border:-border, border:-border,:]
				expand_fusion[:,i:i + block_size, j:j + block_size,:] = fusion_out[:,border:-border, border:-border,:]
				expand_denoise[:,i:i + block_size, j:j + block_size,:] = denoise_out[:,border:-border, border:-border,:]
				expand_gamma[:,i:i + block_size, j:j + block_size,:] = gamma[:,border:-border, border:-border,0:1]
				expand_omega[:, i:i + block_size, j:j + block_size, :] = omega[:, border:-border, border:-border, 0:1]

	refine_result = expand_res[:,border:hgt + border, border:wid + border,:]
	fusion_result = expand_fusion[:,border:hgt + border, border:wid + border,:]
	denoise_result = expand_denoise[:,border:hgt + border, border:wid + border,:]
	gamma_result = expand_gamma[:,border:hgt + border, border:wid + border,:]
	omega_result = expand_omega[:, border:hgt + border, border:wid + border, :]
	print('------------- Run Model Successfully -------------')
	if cfg.use_realism:
		return refine_result, fusion_result, denoise_result, omega_result, gamma_result, real_result
	else:
		return refine_result, fusion_result, denoise_result, omega_result, gamma_result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ngpu = cfg.ngpu
cudnn.benchmark = True # False #

'''network'''
#####################################################################
# quantized
#####################################################################
# model_qua_arch_path = './benchmark_output/adaround2021-11-04-21-49-43'
# model_path = os.path.join(model_qua_arch_path, 'adaround_model.pth')
# state_dict = torch.load(model_path)

#####################################################################
# A100
#####################################################################
PROJ = '/home/wen/Documents/project/video/denoising/emvd'
a100_model = 'results/a100/stage2/'
sub_dir = ''# a100_model
# sub_folder = 'model_photo_real'
# model_name = 'model_real_best.pth' # 'model_best.pth'
cfg.use_arch_qat = False
cfg.use_realism = False
if cfg.use_realism:
	name = 'model_photo_real_gan_noL1_8r_vgg19_34_'# 'model_photo_real'
	sub_folder = a100_model+name
	model_name = 'model_real' # 'model_real_best' #
else:
	name = 'model_px_ecb_bz32'# 'model_px_ecb_edge'#'model_edge_loss_w0.6' #'model_px'# 'model_edge_loss' # 'model_basic'
	sub_folder =  'log/a100/'+ name
	model_name = 'model_best'
if cfg.use_arch_qat:
	model_name += '_qua_arch'
# 'model_sigma_1_perceptual_lambda0.01'
# 'model_sigma_1'
# 'model_CX_VGG19_L0_L5'
# 'model_CX_VGG19_L12' 		# 造成拖影
# 'model_CX_VGG19_L0_L12' 	# 比 L5 保留细节好
# 'model_CX_VGG19_L5'  		# 纹理细节差,过度平滑?
# 'model_CX_VGG19_L5_L12' 	# 比 L5 保留细节好
cfg.use_pixel_shuffle = True
pixel_shuffle = architecture.PixelShuffle(2)
pixel_unshuffle = architecture.PixelShuffle(0.5)
cfg.use_sigma_0 = False # True
cfg.use_sigma_1 = False
cfg.use_isp = False
model_path = os.path.join(PROJ,sub_dir,sub_folder, model_name)+'.pth'
checkpoint = torch.load(model_path)
print(model_path)
if 'iter' in checkpoint.keys():
	state_dict = checkpoint['model']
else:
	state_dict = checkpoint
#####################################################################
# local
#####################################################################
# model_path = cfg.best_model_save_root
# state_dict = torch.load(model_path)


# output_dir = cfg.output_root
output_dir = os.path.join(PROJ,sub_dir,sub_folder,'outputs')
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
output_dir += '/'

#####################################################################
# Model Structure ORIGIN
#####################################################################
# model = structure.MainDenoise(cfg)
#####################################################################
# Model Structure MODIFY
#####################################################################
# model = arch.EMVD(cfg)
if cfg.use_arch_qat:
	model = arch_qat.EMVD(cfg)
elif cfg.use_pixel_shuffle:
	model = architecture.EMVD(cfg)
else:
	model = arch.EMVD(cfg)

#####################################################################
# Model Structure QAT
#####################################################################
# model = arch_qat.EMVD(cfg)
#####################################################################

model = model.cuda() # .to(device)

if ngpu > 1:
	temp=collections.OrderedDict()
	for k,v in state_dict.items():
		if 'module' in k:
			n='.'.join(k.split('.')[1:])
			temp[n]=v
		else:
			temp=state_dict
	model.load_state_dict(temp)
else:
	# model.load_state_dict(checkpoint['model'])
	model.load_state_dict(state_dict)

# multi gpu test
if torch.cuda.is_available() and ngpu > 1:
	model = nn.DataParallel(model, device_ids=list(range(ngpu)))

model.eval()

iso_list = [1600, 3200, 6400, 12800, 25600]

isp = torch.load('isp/ISP_CNN.pth')

iso_average_raw_psnr = 0
iso_average_raw_ssim = 0

iso_average_real_psnr = 0
iso_average_real_ssim = 0

psnr = netloss.PSNR().cuda() #.to(cfg.device)

cfa = None
haar = None
if cfg.use_sigma_0 or cfg.use_sigma_1:
	cfa = arch.Cfa().cuda() # .to(device)
	haar = arch.Haar().cuda() # .to(device)

# for iso_ind, iso in enumerate(iso_list):
for iso_ind in range(0,len(iso_list)):
	iso = iso_list[iso_ind]
	print('processing iso={}'.format(iso))

	if not os.path.isdir(output_dir + 'ISO{}'.format(iso)):
		os.makedirs(output_dir + 'ISO{}'.format(iso))
	subs = ['gt','noisy','fusion','denoise','refine','gamma','omega']
	if cfg.use_realism:
		subs.extend(['real'])
	for sub in subs:
		if not os.path.isdir(output_dir + 'ISO{}'.format(iso) + '/{}'.format(sub)):
			os.makedirs(output_dir + 'ISO{}'.format(iso) + '/{}'.format(sub))

	f = open(output_dir+'denoise_model_test_psnr_and_ssim_on_iso{}.txt'.format(iso), 'w')

	context = 'ISO{}'.format(iso) + '\n'
	f.write(context)

	scene_avg_raw_psnr = 0
	scene_avg_raw_ssim = 0
	frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
	a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
	b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

	scene_avg_real_psnr = 0
	scene_avg_real_ssim = 0
	for scene_id in range(7,11+1):
		context = 'scene{}'.format(scene_id) + '\n'
		f.write(context)

		frame_avg_raw_psnr = 0
		frame_avg_raw_ssim = 0
		block_size = 512
		ft0_fusion_data = np.zeros([1, 540, 960, 4 * 7])
		gt_fusion_data = np.zeros([1, 540, 960, 4 * 7])
		frame_psnr = 0

		frame_real_psnr = 0
		frame_avg_real_psnr = 0
		frame_avg_real_ssim = 0
		for time_ind in range(0,7):
			input_pack_list = []
			noisy_frame_index_for_current = np.random.choice(10, 4, replace=False)
			for i in noisy_frame_index_for_current:
				input_name = os.path.join(cfg.data_root[1],
										'indoor_raw_noisy/indoor_raw_noisy_scene{}/'
										'scene{}/ISO{}/frame{}_noisy0.tiff'.format(scene_id,
																				   scene_id,
																				   iso,
																				   frame_list[time_ind],
																				   i))
				noisy_raw = cv2.imread(input_name, -1)
				input_pack = np.expand_dims(pack_gbrg_raw(noisy_raw), axis=0)
				input_pack_list.append(input_pack)
			input_pack = np.concatenate(input_pack_list, axis=-1)

			gt_raw = cv2.imread(os.path.join(cfg.data_root[1],
												  'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
													  scene_id,scene_id, iso, frame_list[time_ind])), -1).astype(np.float32)
			fgt = np.expand_dims(pack_gbrg_raw(gt_raw), axis=0)

			# if time_ind == 0:
			# 	ft0_fusion = input_pack  # 1 * 512 * 512 * 4
			# else:
			# 	ft0_fusion = ft0_fusion_data[:, :, :,  (time_ind-1) * 4: (time_ind) * 4]  # 1 * 512 * 512 * 4

			input_data = input_pack # np.concatenate([ft0_fusion, input_full], axis=3)
			coeff_a = a_list[iso_ind] / (2 ** 12 - 1 - 240)
			coeff_b = b_list[iso_ind] / (2 ** 12 - 1 - 240) ** 2
			if cfg.use_realism:
				refine_out, fusion_out, denoise_out, omega_out, gamma_out, real_out = test_big_size_raw(input_data,
																							  block_size,
																							  model,
																							  coeff_a, coeff_b)
			else:
				refine_out, fusion_out, denoise_out, omega_out, gamma_out = test_big_size_raw(input_data,
																							block_size,
																							model,
																							coeff_a,
																							coeff_b)
			ft0_fusion_data[:, :, :,  time_ind * 4: (time_ind+1) * 4] = fusion_out
			if cfg.use_realism:
				frame_real_psnr += psnr(real_out, fgt)
				test_real = depack_gbrg_raw(real_out)
				real = test_real * (2 ** 12 - 1 - 240) + 240

			frame_psnr += psnr(refine_out, fgt) # torch.from_numpy()
			test_result = depack_gbrg_raw(refine_out)
			test_fusion = depack_gbrg_raw(fusion_out)
			test_denoise = depack_gbrg_raw(denoise_out)

			test_gt = (gt_raw - 240) / (2 ** 12 - 1 - 240)
			if cfg.use_realism:
				test_real_psnr = compare_psnr(test_gt, (
						np.uint16(test_real * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
													 2 ** 12 - 1 - 240), data_range=1.0)
				test_real_ssim = compute_ssim_for_packed_raw(test_gt, (
						np.uint16(test_real * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
																	2 ** 12 - 1 - 240))

			test_raw_psnr = compare_psnr(test_gt, (
						np.uint16(test_result * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
													 2 ** 12 - 1 - 240), data_range=1.0)
			test_raw_ssim = compute_ssim_for_packed_raw(test_gt, (
						np.uint16(test_result * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
																	2 ** 12 - 1 - 240))
			test_raw_psnr_input = compare_psnr(test_gt, (raw - 240) / (2 ** 12 - 1 - 240), data_range=1.0)
			print('scene {} frame{} test raw psnr : {}, '
				  'test raw input psnr : {}, '
				  'test raw ssim : {} '.format(scene_id, time_ind,
											   test_raw_psnr, test_raw_psnr_input,
											   test_raw_ssim))
			context = 'scene {} frame{} raw psnr/ssim: {}/{}, input_psnr:{}'.format(scene_id, time_ind, test_raw_psnr, test_raw_ssim, test_raw_psnr_input) + '\n'
			f.write(context)
			if cfg.use_realism:
				context = 'stage2 photo real psnr/ssim: {}/{}'.format(test_real_psnr,test_real_ssim) + '\n'
				print(context)
				f.write(context)
				frame_avg_real_psnr += test_real_psnr
				frame_avg_real_ssim += test_real_ssim

			frame_avg_raw_psnr += test_raw_psnr
			frame_avg_raw_ssim += test_raw_ssim

			output = test_result * (2 ** 12 - 1 - 240) + 240
			fusion = test_fusion * (2 ** 12 - 1 - 240) + 240
			denoise = test_denoise * (2 ** 12 - 1 - 240) + 240


			if cfg.vis_data:
				noisy_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(raw), axis=0))
				noisy_srgb_frame = tensor2numpy(isp(noisy_raw_frame))[0]
				gt_raw_frame = np.expand_dims(pack_gbrg_raw(test_gt * (2 ** 12 - 1 - 240) + 240), axis=0)
				gt_srgb_frame = tensor2numpy(isp(preprocess(gt_raw_frame)))[0]
				cv2.imwrite(output_dir + 'ISO{}/noisy/scene{}_frame{}_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(noisy_srgb_frame * 255))
				cv2.imwrite(output_dir + 'ISO{}/gt/scene{}_frame{}_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(gt_srgb_frame * 255))
			denoised_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(output), axis=0))
			denoised_srgb_frame = tensor2numpy(isp(denoised_raw_frame))[0]
			cv2.imwrite(output_dir + 'ISO{}/refine/scene{}_frame{}_sRGB.png'.format(iso, scene_id, time_ind),
						np.uint8(denoised_srgb_frame * 255))
			if cfg.use_realism:
				denoised_real_frame = preprocess(np.expand_dims(pack_gbrg_raw(real), axis=0))
				denoised_srgb_real_frame = tensor2numpy(isp(denoised_real_frame))[0]
				cv2.imwrite(output_dir + 'ISO{}/real/scene{}_frame{}_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(denoised_srgb_real_frame * 255))

			if cfg.vis_data:
				fusion_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(fusion), axis=0))
				fusion_srgb_frame = tensor2numpy(isp(fusion_raw_frame))[0]
				cv2.imwrite(output_dir + 'ISO{}/fusion/scene{}_frame{}_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(fusion_srgb_frame * 255))

				denoise_midres_raw_frame = preprocess(np.expand_dims(pack_gbrg_raw(denoise), axis=0))
				denoised_mid_res_srgb_frame = tensor2numpy(isp(denoise_midres_raw_frame))[0]
				cv2.imwrite(output_dir + 'ISO{}/denoise/scene{}_frame{}_sRGB.png'.format(iso, scene_id, time_ind),
							np.uint8(denoised_mid_res_srgb_frame * 255))

				cv2.imwrite(output_dir + 'ISO{}/gamma/scene{}_frame{}.png'.format(iso, scene_id, time_ind), np.uint8(gamma_out[0] * 255))
				cv2.imwrite(output_dir + 'ISO{}/omega/scene{}_frame{}.png'.format(iso, scene_id, time_ind), np.uint8(omega_out[0] * 255))
				print('gamma.max:', gamma_out.max())
				print('gamma.min:', gamma_out.min())
				# print('frame_psnr', frame_psnr)
		frame_psnr = frame_psnr / 7
		print('frame_psnr average', frame_psnr.item())
		frame_avg_raw_psnr = frame_avg_raw_psnr / 7
		frame_avg_raw_ssim = frame_avg_raw_ssim / 7
		context = 'frame average raw psnr:{},frame average raw ssim:{}'.format(frame_avg_raw_psnr,
																			   frame_avg_raw_ssim) + '\n'
		f.write(context)
		scene_avg_raw_psnr += frame_avg_raw_psnr
		scene_avg_raw_ssim += frame_avg_raw_ssim

		if cfg.use_realism:
			frame_avg_real_psnr /= 7
			frame_avg_real_ssim /= 7
			context = 'stage2 photo real frame average psnr/ssim: {}/{}'.format(frame_avg_real_psnr, frame_avg_real_ssim) + '\n'
			print(context)
			f.write(context)
			scene_avg_real_psnr += frame_avg_real_psnr
			scene_avg_real_ssim += frame_avg_real_ssim

	scene_avg_raw_psnr = scene_avg_raw_psnr / 5
	scene_avg_raw_ssim = scene_avg_raw_ssim / 5
	context = 'scene average raw psnr:{},scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
																				 scene_avg_raw_ssim) + '\n'
	print(context)
	f.write(context)
	iso_average_raw_psnr += scene_avg_raw_psnr
	iso_average_raw_ssim += scene_avg_raw_ssim

	if cfg.use_realism:
		scene_avg_real_psnr /= 5
		scene_avg_real_ssim /= 5
		context = 'stage2 photo real scene average psnr/ssim: {}/{}'.format(scene_avg_real_psnr,
																			scene_avg_real_ssim) + '\n'
		print(context)
		f.write(context)
		iso_average_real_psnr += scene_avg_real_psnr
		iso_average_real_ssim += scene_avg_real_ssim


iso_average_raw_psnr = iso_average_raw_psnr / len(iso_list)
iso_average_raw_ssim = iso_average_raw_ssim / len(iso_list)
context = 'iso average raw psnr:{},iso frame average raw ssim:{}'.format(iso_average_raw_psnr, iso_average_raw_ssim) + '\n'
f.write(context)
print(context)

if cfg.use_realism:
	iso_average_real_psnr /= len(iso_list)
	iso_average_real_ssim /= len(iso_list)
	context = 'stage2 photo real iso average psnr/ssim: {}/{}'.format(iso_average_real_psnr,
																		iso_average_real_ssim) + '\n'
	print(context)
	f.write(context)



