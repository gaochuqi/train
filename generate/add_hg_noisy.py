import random
import torch
import torch.distributions as tdist


# 噪声初始值，以CRVD为例
iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]


# 加噪函数
def add_hg_noise(cfg, image, shot_noise=13.486051, read_noise=130.818508):
    random_int = random.randint(100-cfg.noise_random, 100+cfg.noise_random)    # cfg.noise_random = 15 效果比较好
    cur_rate = random_int / 100
    image = image.permute(0, 2, 3, 1)           # Permute the image tensor to n xHxWxC format from CxHxW format
    variance = (image * shot_noise + read_noise) * cur_rate
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise = n.sample()
    img_N = image + noise
    img_N = img_N.permute(0, 3, 1, 2)           # Re-Permute the tensor back to n xCxHxW format
    img_N = torch.clamp(img_N, 0, 1)
    return img_N


# 此函数为emvd的train.py 里面的train
def train(cfg, gt, noisy_level, device, other_para):
    # 2 ** 12 - 1 - 240 = 3855  对应CRVD的数据集
    # 2 ** 10 - 1 - 64  = 959   对应mi的数据集
    coeff_a = (noisy_level[0] / (2 ** 12 - 1 - 240)).float().to(device)
    coeff_a = coeff_a[:, None, None, None]
    coeff_b = (noisy_level[1] / (2 ** 12 - 1 - 240) ** 2).float().to(device)
    coeff_b = coeff_b[:, None, None, None]
    for time_ind in range(cfg.frame_num):
        gt_n = add_hg_noise(cfg, gt, coeff_a, coeff_b)           # 此处输入干净的gt(n,c,h,w)-value:(0-1), 输出加噪后的gt_n


