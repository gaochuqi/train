import os
import cv2
import numpy as np
import torch

import config as cfg
from torch.utils.data import Dataset
from dataset.dataset import norm_raw
# from dataset import norm_raw


def load_cvrd_data(shift, noisy_level,
                   scene_ind, frame_ind, xx, yy):
    frame_list = cfg.frame_list
    if scene_ind in cfg.obj_motion:
        video_type = 'obj_motion'
    else:
        video_type = 'camera_motion'
    iso_list = cfg.iso_list
    scene_ind = '{:0>4d}'.format(scene_ind)
    gt_name = os.path.join(cfg.data_root, '{}/{}/raw_bin/frame_{}.npy'.format(
        scene_ind, video_type, frame_list[frame_ind + shift]))

    if not os.path.exists(gt_name):
        print('Not Exist: ',gt_name)

    gt_raw = np.load(gt_name, mmap_mode='r')
    gt_raw_full = gt_raw
    gt_raw_patch = gt_raw_full[(yy//2):(yy//2) + cfg.image_height,
                                (xx//2):(xx//2) + cfg.image_width]  # 256 * 256
    gt_raw_pack = norm_raw(gt_raw_patch, black_level=cfg.black_level, white_level=cfg.white_level) # h,w

    noisy_frame_index_for_current = np.random.randint(1, cfg.noisy_num+1)
    noisy_name = os.path.join(cfg.data_root, '{}/{}/noisy/iso{}/frame_{}_iso{}_noisy{}.npy'.format(
        scene_ind, video_type, iso_list[noisy_level], frame_list[frame_ind + shift], iso_list[noisy_level],
        noisy_frame_index_for_current))
    noisy_raw = np.load(noisy_name, mmap_mode='r')
    noisy_raw_full = noisy_raw
    noisy_patch = noisy_raw_full[yy:yy + cfg.image_height * 2, xx:xx + cfg.image_width * 2]
    input_pack = norm_raw(noisy_patch, black_level=cfg.black_level, white_level=cfg.white_level) # h,w
    return input_pack, gt_raw_pack


def load_eval_data(noisy_level, scene_ind):
    input_batch_list = []
    gt_raw_batch_list = []

    input_pack_list = []
    gt_raw_pack_list = []

    xx = 400 # 200
    yy = 400 # 200

    for shift in range(0, cfg.frame_num):
        # load gt raw
        frame_ind = 0
        input_pack, gt_raw_pack = load_cvrd_data(shift, noisy_level,
                                                 scene_ind, frame_ind,
                                                 xx, yy)
        input_pack_list.append(input_pack)
        gt_raw_pack_list.append(gt_raw_pack)

    input_pack_frames = np.stack(input_pack_list, axis=0)
    gt_raw_pack_frames = np.stack(gt_raw_pack_list, axis=0)

    input_batch_list.append(input_pack_frames)
    gt_raw_batch_list.append(gt_raw_pack_frames)

    input_batch = np.stack(input_batch_list, axis=0)
    gt_raw_batch = np.stack(gt_raw_batch_list, axis=0)

    in_data = torch.from_numpy(input_batch.copy()).cuda()  # 1 * 25 * 128 * 128
    gt_raw_data = torch.from_numpy(gt_raw_batch.copy()).cuda()  # 1 * 25 * 128 * 128
    return in_data, gt_raw_data

def generate_file_list(scene_list):
    iso_list = cfg.iso_list
    file_num = 0
    data_name = []
    for scene_ind in scene_list:
        if scene_ind in cfg.obj_motion:
            video_type = 'obj_motion'
        else:
            video_type = 'camera_motion'
        # scene_ind = '{:0>4d}'.format(scene_ind)
        for iso in iso_list:
            for frame_ind in range(8):
                gt_name = 'videoType,{},scene,{},frame,{},iso,{}'.format(video_type, scene_ind, frame_ind, iso)
                data_name.append(gt_name)
                file_num += 1
    random_index = np.random.permutation(file_num)
    data_random_list = []
    for i, idx in enumerate(random_index):
        data_random_list.append(data_name[idx])
    return data_random_list

def read_img(img_name, xx, yy):
    raw = cv2.imread(img_name, -1)
    raw_full = raw
    raw_patch = raw_full[yy:yy + cfg.image_height * 2,
                         xx:xx + cfg.image_width * 2]  # 256 * 256
    raw_pack_data = norm_raw(raw_patch)
    return raw_pack_data


def read_img_gt(img_name, xx, yy):
    raw = np.load(img_name, mmap_mode='r')
    raw_full = raw
    raw_patch = raw_full[yy:yy + cfg.image_height,
                         xx:xx + cfg.image_width]
    raw_pack_data = norm_raw(raw_patch, black_level=cfg.black_level, white_level=cfg.white_level)
    return raw_pack_data

def read_img_noisy(img_name, xx, yy):
    # print(img_name)
    raw = np.load(img_name, mmap_mode='r')
    raw_full = raw
    raw_patch = raw_full[yy:yy + cfg.image_height * 2,
                         xx:xx + cfg.image_width * 2]
    raw_pack_data = norm_raw(raw_patch, black_level=cfg.black_level, white_level=cfg.white_level)
    return raw_pack_data

def decode_data(data_name):
    # r = np.random.rand()
    # print(r)
    frame_list = cfg.frame_list
    # if r < cfg.percent:
    #     frame_list = cfg.frame_list
    # else:
    #     frame_list = [1 * np.random.randint(0, 8)] * len(cfg.frame_list)
    H = cfg.height
    W = cfg.width
    iso_list = cfg.iso_list
    a_list = cfg.a_list
    b_list = cfg.b_list

    # data_name : 'videoType,{},scene,{},frame,{},iso,{}'
    _, video_type, _, scene_ind, _, frame_ind, _, iso_ind = data_name.split(',')
    scene_ind = '{:0>4d}'.format(int(scene_ind))
    noisy_level_ind = iso_list.index(int(iso_ind))
    noisy_level = [a_list[noisy_level_ind], b_list[noisy_level_ind]]

    xx = np.random.randint(0, (W - cfg.image_width * 2 + 1) / 4) * 4
    yy = np.random.randint(0, (H - cfg.image_height * 2 + 1) / 4) * 4
    xx_list = []
    yy_list = []

    xx_gt = xx // 2     # start from even pixel
    yy_gt = yy // 2
    xx_gt_list = []
    yy_gt_list = []

    gt_name_list = []
    noisy_name_list = []

    for shift in range(0, cfg.frame_num):
        gt_name = os.path.join(cfg.data_root, '{}/{}/raw_bin/frame_{}.npy'.format(
            scene_ind, video_type,
            frame_list[int(frame_ind) + shift]))
        gt_name_list.append(gt_name)
        xx_gt_list.append(xx_gt)
        yy_gt_list.append(yy_gt)

        noisy_frame_index_for_current = np.random.randint(1, cfg.noisy_num+1)
        noisy_name = os.path.join(cfg.data_root, '{}/{}/noisy/iso{}/frame_{}_iso{}_noisy{}.npy'.format(
                                                    scene_ind, video_type, iso_ind,
                                                    frame_list[int(frame_ind) + shift], iso_ind,
                                                    noisy_frame_index_for_current))
        noisy_name_list.append(noisy_name)
        xx_list.append(xx)
        yy_list.append(yy)
    gt_raw_data_list  = list(map(read_img_gt,
                                 gt_name_list,
                                 xx_gt_list, yy_gt_list))
    noisy_data_list = list(map(read_img_noisy,
                               noisy_name_list,
                               xx_list, yy_list))
    gt_raw_batch = np.stack(gt_raw_data_list, axis=0)
    noisy_raw_batch = np.stack(noisy_data_list, axis=0)
    return noisy_raw_batch, gt_raw_batch, noisy_level


class loadImgs(Dataset):
    def __init__(self, filelist):
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        self.data_name = self.filelist[item]
        image, label, noisy_level = decode_data(self.data_name)
        self.image = image
        self.label = label
        self.noisy_level = noisy_level
        return self.image, self.label, self.noisy_level


def test():
    train_data_name_queue = generate_file_list(['{:0>4d}'.format(i) for i in range(1, cfg.scene + 1)])
    train_dataset = loadImgs(train_data_name_queue)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               shuffle=True,
                                               pin_memory=True)


    for i, (input, label, noisy_level) in enumerate(train_loader):
        print('############', i)


def main():
    test()

if __name__ == '__main__':
    main()
