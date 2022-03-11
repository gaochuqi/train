import glob
import os.path
import numpy as np
import cv2
import torch
from torch import nn
import random
import rawpy
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#######################################################
# PixelShuffle & PixelUnShuffle         channel first
#######################################################
def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)

class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


pxs = PixelShuffle(2)
pxuns = PixelShuffle(0.5)
avgpool = torch.nn.AvgPool2d(2, 2)

def convert_to_raw():
    data_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/'
    dataset_name = 'DRV' # 'MIT_Adobe_FiveK' #
    if dataset_name == 'DRV':
        raw_file_pattern = data_dir + dataset_name + '/long/*/*.ARW'
        # img_pattern = data_dir + dataset_name + '/long/*/*.png'
        # img_list = glob.glob(img_pattern)
        # tmp = []
        # for img_path in img_list:
        #     if 'half' in img_path:
        #         continue
        #     else:
        #         tmp.append(img_path)
        # tmp.sort()
        # img_list = tmp
    elif dataset_name == 'MIT_Adobe_FiveK':
        raw_file_pattern = data_dir + dataset_name + '/fivek_dataset/raw_photos/*/photos/*.dng'
    raw_list = glob.glob(raw_file_pattern)
    raw_list.sort()
    out_dir = data_dir + dataset_name + '/raw'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir + '/gt')
        os.makedirs(out_dir + '/clean')
    f = open(data_dir + dataset_name + '/{}.txt'.format(dataset_name), 'w')
    content = 'raw_name h w black_level white_level color_desc raw_pattern\n'
    f.writelines(content)
    cnt = 0
    for i, raw_path in enumerate(raw_list):
        if i > 0 and i % 100 == 0:
            print('{} / {}'.format(len(raw_list), i))
        # out_dir = os.path.dirname(raw_path)
        # img_path = img_list[i]
        raw_name = os.path.basename(raw_path).split('.')[0]
        # img_name = os.path.basename(img_path).split('.')[0]
        # assert raw_name == img_name
        raw_obj = rawpy.imread(raw_path)
        raw_image = raw_obj.raw_image
        raw_image_visible = raw_obj.raw_image_visible
        black_level_per_channel = raw_obj.black_level_per_channel
        black_level = black_level_per_channel[0]
        camera_white_level_per_channel = raw_obj.camera_white_level_per_channel
        camera_whitebalance = raw_obj.camera_whitebalance
        color_desc = raw_obj.color_desc
        color_matrix = raw_obj.color_matrix
        daylight_whitebalance = raw_obj.daylight_whitebalance
        raw_pattern = raw_obj.raw_pattern
        rgb_xyz_matrix = raw_obj.rgb_xyz_matrix
        white_level = raw_obj.white_level
        raw_shape = raw_image_visible.shape
        if len(raw_shape) > 2:
            cnt += 1
            print(raw_name, raw_shape)
            continue
            # if raw_shape[-1] == 4:
            #     tmp = np.transpose(raw_image_visible,(2,0,1))
            #     tmp = np.expand_dims(tmp.astype(np.float32), axis=0)
            #     tmp = pxs(torch.from_numpy(tmp))
            #     raw_image_visible = tmp.data.cpu().numpy().squeeze()
            #     raw_shape = raw_image_visible.shape
        elif raw_shape[0] % 2 !=0 or raw_shape[1] % 2 !=0:
            cnt += 1
            print(raw_name, raw_shape)
            continue
        if dataset_name == 'DRV':
            suffix = raw_path.split('/')[-2]+'_'
            raw_name = suffix + raw_name
        # print(np.min(raw_image_visible), np.max(raw_image_visible))
        raw_image_visible.tofile(out_dir + '/clean' + '/{}.raw'.format(raw_name))
        raw_norm = np.maximum(raw_image_visible - black_level, 0) / (white_level - black_level)
        raw_norm_gray = Image.fromarray(np.uint8(raw_norm * 255))
        raw_norm_gray.save(out_dir + '/clean' + '/{}.png'.format(raw_name))
        raw_torch = torch.from_numpy(np.expand_dims(np.expand_dims(raw_image_visible.astype(np.float32), axis=0), axis=0))
        raw = pxuns(raw_torch)
        raw_bin = avgpool(raw)
        raw_bin = pxs(raw_bin)
        raw_bin_np = raw_bin.data.cpu().numpy().squeeze()
        raw_bin_np.astype(np.uint16).tofile(out_dir + '/gt'+'/{}_bin.raw'.format(raw_name))
        raw_bin_np_norm = np.maximum(raw_bin_np - black_level, 0) / (white_level - black_level)
        raw_bin_np_norm_gray = Image.fromarray(np.uint8(raw_bin_np_norm * 255))
        raw_bin_np_norm_gray.save(out_dir + '/gt' + '/{}_bin.png'.format(raw_name))
        content = '{} {} {} {} {} {} {}\n'.format(out_dir+'/{}.raw'.format(raw_name),
                                                  raw_shape[0], raw_shape[1],
                                                  black_level_per_channel[0], white_level,
                                                  color_desc, raw_pattern)
        f.writelines(content)
        # img = cv2.imread(img_path, -1)
        # img_shape = img.shape
        # print(raw_name)
    f.close()
    print('end')
    print('ERROR file', cnt)

def objects_motion():
    H = 3672
    W = 5496
    data_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/' # 3672 5496 800 16380 b'RGBG'
    obj_sub_dir = 'objects/'
    raw_sub_dir = 'long/'
    obj_list = glob.glob(data_dir+obj_sub_dir+'*.jpg')
    obj_list.sort()
    for obj_mask_path in obj_list:
        obj_mask_name = os.path.basename(obj_mask_path)
        obj_mask_name_split = obj_mask_name.split('_')
        if len(obj_mask_name_split) < 4:
            continue
        raw_sub_folder = obj_mask_name_split[0] + '/'
        raw_name = '_'.join(obj_mask_name_split[1:3])[4:] # 'halfxxxxxx'
        y_min, y_max, x_min, x_max = obj_mask_name_split[4:8]
        raw_path = data_dir + raw_sub_dir + raw_sub_folder + raw_name + '.raw'
        raw = np.fromfile(raw_path, dtype=np.uint16)
        raw = np.reshape(raw, (H, W))
        y_min = int(y_min) * 4
        y_max = (int(y_max)+1) * 4
        x_min = int(x_min) * 4
        x_max = (int(x_max)+1) * 4
        tmp = Image.open(obj_mask_path)
        w, h = tmp.size
        tmp = tmp.resize((w * 4, h * 4), resample=Image.BICUBIC)
        tmp = np.array(tmp)
        tmp = np.sum(tmp, axis=-1)
        mask = np.zeros_like(tmp, dtype=np.uint16)
        mask[tmp > 0] = 1
        crop = raw[y_min:y_max, x_min:x_max]
        obj = crop * mask
        x1 = 0 + w * 2
        y1 = 0 + h * 2
        x2 = W - w * 2
        y2 = H - h * 2
        y_c = np.random.randint(y1, y2)
        x_c = np.random.randint(x1, x2)
        raw1 = np.copy(raw)
        back = raw1[(y_c - 2 * h):(y_c + 2 * h), (x_c - 2 * w):(x_c + 2 * w)]
        raw1[(y_c - 2 * h):(y_c + 2 * h), (x_c - 2 * w):(x_c + 2 * w)] = np.where(obj > 0, obj, back)
        raw1.tofile(raw_name + '_01.raw')

# Affine transformation
'''
仿射变换通过一系列原子变换复合实现，具体包括：平移（Translation）、缩放（Scale）、旋转（Rotation）、翻转（Flip）和错切（Shear）
'''
def affine_transformation_DRV():
    H = 3672
    W = 5496
    data_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/'
    dataset_name = 'DRV'  # 'MIT_Adobe_FiveK' #
    raw_file_pattern = data_dir + dataset_name + '/raw/clean/png/*/*.png'
    raw_list = glob.glob(raw_file_pattern)
    raw_list.sort()
    out_dir = data_dir + dataset_name + '/frames'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, raw_path in enumerate(raw_list):
        sub_folder = os.path.dirname(raw_path).split('/')[-2]
        out_sub_dir = os.path.join(out_dir, sub_folder)
        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)
        # raw = np.fromfile(raw_path, dtype=np.uint16)
        # raw = np.reshape(raw, (H, W))
        img = cv2.imread(raw_path)
        res = cv2.warpAffine(img, H, (rows, cols))
        m_rotate = cv2.getRotationMatrix2D()
        res = cv2.warpAffine(img, m_rotate, (W//2, H//2))
        print(i)



def main():
    setup_seed(666)
    # convert_to_raw()
    objects_motion()

if __name__ == '__main__':
    main()

