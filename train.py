import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2
import warnings
warnings.filterwarnings('ignore')
from torchstat import stat

import utils
from dataset import *
import config      as cfg

import structure         as structure
import netloss           as netloss
from load_data import *
import time

from arch.modules import PixelShuffle


iso_list = cfg.iso_list # [1600, 3200, 6400, 12800, 25600]
a_list = cfg.a_list # [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = cfg.b_list # [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

loss_save=cfg.log_dir
pixel_shuffle = PixelShuffle(2)
pixel_unshuffle = PixelShuffle(0.5)


def initialize():
	"""
	# clear some dir if necessary
	make some dir if necessary
	make sure training from scratch
	:return:
	"""
	##
	# if not os.path.exists(cfg.model_name):
	# 	os.mkdir(cfg.model_name)

	# if not os.path.exists(cfg.debug_dir):
	# 	os.makedirs(cfg.debug_dir)

	if not os.path.exists(cfg.log_dir):
		os.makedirs(cfg.log_dir)


	#if cfg.checkpoint == None:
	#	s = input('Are you sure training the model from scratch? y/n \n')
	#	if not (s=='y'):
	#		return


def duplicate_output_to_log(name):
	tee = utils.Tee(name)
	return tee


def train(in_data, gt_raw_data, noisy_level, model, loss, device, optimizer):
	l1loss_list = []
	l1loss_total = 0

	l1_fu = 0
	l1_de = 0

	coeff_a = (noisy_level[0] / (2 ** 12 - 1 - 240)).float().to(device)
	coeff_a = coeff_a[:,None,None,None]
	coeff_b = (noisy_level[1] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
	coeff_b = coeff_b[:, None, None, None]
	for time_ind in range(cfg.frame_num):
		ft1 = in_data[:, time_ind * 4: (time_ind + 1) * 4, :, :] 	 #  the t-th input frame
		fgt = gt_raw_data[:, time_ind * 4: (time_ind + 1) * 4, :, :] #  the t-th gt frame

		ft1  = pixel_unshuffle(ft1)

		if time_ind == 0:
			ft0_fusion = ft1
		else:
			ft0_fusion = ft0_fusion_data							 # the t-1 fusion frame

		input = torch.cat([ft0_fusion, ft1], dim=1)

		model.train()
		gamma, fusion_out, denoise_out, omega, refine_out = model(input, coeff_a, coeff_b)

		ft0_fusion_data = fusion_out
		# print(gamma.shape, fusion_out.shape, denoise_out.shape, omega.shape, refine_out.shape, fgt.shape, ft1.shape)

		fusion_out = pixel_shuffle(fusion_out)
		denoise_out = pixel_shuffle(denoise_out)
		refine_out = pixel_shuffle(refine_out)
		omega = pixel_shuffle(omega)
		gamma = pixel_shuffle(gamma)
		ft1 = pixel_shuffle(ft1)
				
		# print(gamma.shape, fusion_out.shape, denoise_out.shape, omega.shape, refine_out.shape, fgt.shape, ft1.shape)

		loss_refine = loss(refine_out, fgt)
		loss_fusion = loss(fusion_out, fgt)
		loss_denoise = loss(denoise_out, fgt)

		l1loss = loss_refine

		l1loss_list.append(l1loss)
		l1loss_total += l1loss

		l1_fu += loss_fusion
		l1_de += loss_denoise

		
	loss_ct = netloss.loss_color(model, ['ct.net1.weight', 'cti.net1.weight'], device)
	loss_ft = netloss.loss_wavelet(model, device)
	total_loss = l1loss_total / (cfg.frame_num) + loss_ct + loss_ft
	l1_fu /= cfg.frame_num
	l1_de /= cfg.frame_num
	l1_re = l1loss_total / (cfg.frame_num)

	optimizer.zero_grad()
	total_loss.backward()
	torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
	optimizer.step()

	print('Total Loss  ', ('%.8f' % total_loss.item()),
	'| color Loss  ', ('%.8f' % loss_ct.item()),
	'| freq Loss  ', ('%.8f' % loss_ft.item()),
	'| fusion Loss  ', ('%.8f' % l1_fu.item()),
	'| denoise Loss  ', ('%.8f' % l1_de.item()),
	'| refine Loss  ', ('%.8f' % l1_re.item()))


	del in_data, gt_raw_data
	return	ft1, fgt, refine_out, fusion_out, denoise_out, gamma, omega, total_loss, loss_ct, loss_ft, l1_fu, l1_de, l1_re # loss_fusion, loss_denoise

def evaluate(model, psnr, writer, iter):
	print('Evaluate...')
	cnt = 0
	total_psnr = 0
	total_psnr_raw = 0
	model.eval()
	with torch.no_grad():
		for scene_ind in range(7,9):
			for noisy_level in range(0,5):
				in_data, gt_raw_data = load_eval_data(noisy_level, scene_ind)
				frame_psnr = 0
				frame_psnr_raw = 0
				for time_ind in range(cfg.frame_num):
					ft1 = in_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]
					fgt = gt_raw_data[:, time_ind * 4: (time_ind + 1) * 4, :, :]

					ft1 = pixel_unshuffle(ft1)

					if time_ind == 0:
						ft0_fusion = ft1
					else:
						ft0_fusion = ft0_fusion_data

					coeff_a = a_list[noisy_level] / (2 ** 12 - 1 - 240)
					coeff_b = b_list[noisy_level] / (2 ** 12 - 1 - 240) ** 2
					input = torch.cat([ft0_fusion, ft1], dim=1)

					gamma, fusion_out, denoise_out, omega, refine_out = model(input, coeff_a, coeff_b)

					ft0_fusion_data = fusion_out

					refine_out = pixel_shuffle(refine_out)
					ft1 = pixel_shuffle(ft1)

					frame_psnr += psnr(refine_out, fgt)
					frame_psnr_raw += psnr(ft1, fgt)

				frame_psnr = frame_psnr / (cfg.frame_num)
				frame_psnr_raw = frame_psnr_raw / (cfg.frame_num)
				print('---------')
				print('Scene: ', ('%02d' % scene_ind), 'Noisy_level: ', ('%02d' % noisy_level), 'PSNR: ', '%.8f' % frame_psnr.item())
				total_psnr += frame_psnr
				total_psnr_raw += frame_psnr_raw
				cnt += 1
				del in_data, gt_raw_data
		total_psnr = total_psnr / cnt
		total_psnr_raw = total_psnr_raw / cnt
	print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
	writer.add_scalar('PSNR', total_psnr.item(), iter)
	writer.add_scalar('PSNR_RAW', total_psnr_raw.item(), iter)
	writer.add_scalar('PSNR_IMP', total_psnr.item() - total_psnr_raw.item(), iter)
	torch.cuda.empty_cache()
	return	total_psnr, total_psnr_raw

def main():
	"""
	Train, Valid, Write Log, Write Predict ,etc
	:return:
	"""
	checkpoint = cfg.checkpoint
	start_epoch = cfg.start_epoch
	start_iter = cfg.start_iter
	best_psnr = 0

	## use gpu
	device = cfg.device
	ngpu = cfg.ngpu
	cudnn.benchmark = True

	## tensorboard --logdir runs
	writer = SummaryWriter(cfg.log_dir)

	## initialize model
	model = structure.MainDenoise()

	## compute GFLOPs
	# stat(model, (8,512,512))

	model = model.to(device)
	loss = netloss.L1Loss().to(device)
	psnr = netloss.PSNR().to(device)

	learning_rate = cfg.learning_rate
	optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)

	## load pretrained model
	if checkpoint is not None:
		print('--- Loading Pretrained Model ---')
		print(checkpoint)
		checkpoint = torch.load(checkpoint)
		start_epoch = checkpoint['epoch']
		start_iter = checkpoint['iter']
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
	iter = start_iter

	if torch.cuda.is_available() and ngpu > 1:
		model = nn.DataParallel(model, device_ids=list(range(ngpu)))

	shutil.copy('structure.py', cfg.log_dir) # os.path.join(cfg.model_name))
	shutil.copy('train.py', cfg.log_dir) # os.path.join(cfg.model_name))
	shutil.copy('netloss.py', cfg.log_dir) # os.path.join(cfg.model_name))

	train_data_name_queue = generate_file_list(['1', '2', '3', '4', '5', '6'])
	train_dataset = loadImgs(train_data_name_queue)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = cfg.batch_size, num_workers = cfg.num_workers, shuffle = True, pin_memory = True)

	eval_data_name_queue = generate_file_list(['7', '8'])
	eval_dataset = loadImgs(eval_data_name_queue)
	eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = cfg.batch_size, num_workers = cfg.num_workers, shuffle = True, pin_memory = True)

	for epoch in range(start_epoch, cfg.epoch):
		print('------------------------------------------------')
		print('Epoch                |   ', ('%08d' % epoch))
		for i, (input, label, noisy_level) in enumerate(train_loader):
			print('------------------------------------------------')
			print('Iter                 |   ', ('%08d' % iter))
			in_data = input.permute(0, 3, 1, 2).to(device)
			gt_raw_data = label.permute(0, 3, 1, 2).to(device)

			ft1, fgt, refine_out, fusion_out, denoise_out, gamma, omega, \
			total_loss, loss_ct, loss_ft, loss_fusion, loss_denoise, loss_refine = train(in_data, gt_raw_data, noisy_level, model, loss, device, optimizer)
			loss_save = cfg.log_dir
			file_save = open(loss_save, mode='a')
			file_save.write(
				'\n' + 'Iter:' + str(iter) + '  total_loss:' + str(total_loss) + '  loss_ct:' + str(loss_ct)+ '  loss_ft:' + str(loss_ft)+ '  loss_fusion:' + str(loss_fusion)+ '  loss_denoise:' + str(loss_denoise)+ '  loss_refine:' + str(loss_refine))
			file_save.close()
			iter = iter + 1
			if iter % cfg.log_step == 0:
				input_gray = torch.mean(ft1, 1, True)
				label_gray = torch.mean(fgt, 1, True)
				predict_gray = torch.mean(refine_out, 1, True)
				fusion_gray = torch.mean(fusion_out, 1, True)
				denoise_gray = torch.mean(denoise_out, 1, True)
				gamma_gray = torch.mean(gamma[:, 0:1, :, :], 1, True)
				omega_gray = torch.mean(omega[:, 0:1, :, :], 1, True)

				writer.add_image('input', make_grid(input_gray.cpu(), nrow=4, normalize=True), iter)
				writer.add_image('fusion_out', make_grid(fusion_gray.cpu(), nrow=4, normalize=True), iter)
				writer.add_image('denoise_out', make_grid(denoise_gray.cpu(), nrow=4, normalize=True), iter)
				writer.add_image('refine_out', make_grid(predict_gray.cpu(), nrow=4, normalize=True), iter)
				writer.add_image('label', make_grid(label_gray.cpu(), nrow=4, normalize=True), iter)

				writer.add_image('gamma', make_grid(gamma_gray.cpu(), nrow=4, normalize=True), iter)
				writer.add_image('omega', make_grid(omega_gray.cpu(), nrow=4, normalize=True), iter)

				writer.add_scalar('L1_Loss', total_loss.item(), iter)
				writer.add_scalar('L1_Color', loss_ct.item(), iter)
				writer.add_scalar('L1_Wavelet', loss_ft.item(), iter)
				writer.add_scalar('L1_Denoise', loss_denoise.item(), iter)
				writer.add_scalar('L1_Fusion', loss_fusion.item(), iter)
				writer.add_scalar('L1_Refine', loss_refine.item(), iter)

				torch.save({
					'epoch': epoch,
					'iter': iter,
					'model': model.state_dict(),
					'optimizer': optimizer.state_dict()},
					cfg.model_save_root)

			if iter % cfg.valid_step == 0 and iter > cfg.valid_start_iter:
				eval_psnr, eval_psnr_raw = evaluate(model, psnr, writer, iter)

				if eval_psnr>best_psnr:
					best_psnr = eval_psnr
					torch.save({
						'epoch': epoch,
						'iter': iter,
						'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'best_psnr': best_psnr},
						cfg.best_model_save_root)
	writer.close()


if __name__ == '__main__':
	initialize()
	main()
