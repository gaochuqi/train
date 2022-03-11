import glob
import math
import csv
import os.path
import random
import torch
import torch.nn as nn
import cv2
import numpy as np
import yaml
import rawpy
from PIL import Image
import shutil

from process import process
from unprocess import random_ccm
from tools import binning_raw, pixel_unShuffle, \
    tensor2numpy, pixel_unShuffle_RGBG, \
    setup_seed, convert_to_GBRG, norm_raw
from Gyro import *


isp = torch.load('ISP_CNN.pth')


def test_Gyro():
    gyro_info = genGyroInfo()
    print('GenGyroInfro. IS OK!')
    # gen Grid
    grid = genMeshGrid()
    deltatime = 2.378e-4
    frame_path = 'synthesisVideo/0001_5.png'
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    print(gyro_info.shape[0])
    # M = cv2.getRotationMatrix2D(center, angle, scale)
    for i in range(7, 30 * 15 * 3, 14):
        rotMatrix = GytoMatrix(gyro_info[i], deltatime)
        for idy in range(cropGridNum_y):
            for idx in range(cropGridNum_x):
                pointInhomoCod = np.array([grid[idy][idx][0], grid[idy][idx][1], 1])
                pointInhomoCodNew = rotMatrix @ pointInhomoCod
                assert pointInhomoCodNew[2] != 0
                grid[idy][idx][0] = pointInhomoCodNew[0] / pointInhomoCodNew[2]
                grid[idy][idx][1] = pointInhomoCodNew[1] / pointInhomoCodNew[2]
        # rerotMatrix = ([[1, 0, 959.51], [0, 1, 539.51],[0, 0, 1]])
        # grid = np.dot(rotMatrix, grid)
        # grid[:,:,0]=grid[:,:,0]+959.5
        # grid[:,:,1]=grid[:,:,1]+539.51
        gridCV1080P_map = cv2.remap(grid.astype(np.float32), gridCV1080P_map_x, gridCV1080P_map_y, cv2.INTER_LINEAR)
        warppedFrame = cv2.remap(frame, gridCV1080P_map[:, :, 0], gridCV1080P_map[:, :, 1],
                                 cv2.INTER_CUBIC)  # INTER_CUBIC INTER_LINEAR
        # warppedFrame = cv.remap(frame, grid[:, :, 0], grid[:, :, 1], cv.INTER_CUBIC)
        fname = os.path.basename(frame_path).split('.')[0]
        cv2.imwrite('synthesisVideo/out/{}_{}.png'.format(fname, i), warppedFrame)
        # cv.imshow('Gyro Show', warppedFrame)
        # cv.waitKey(30)


# https://blog.csdn.net/liuweiyuxiang/article/details/82799999
# https://blog.csdn.net/leviopku/article/details/87936578           仿射
# https://blog.csdn.net/AI_girl/article/details/114846797           透视
'''                 
           ^ y   
           |   / z
           |  /
           | /
-----------|-----------> x
           |
           |
'''


def affineTransform(param_rotate_x_axis_1=100,
                    param_rotate_x_axis_2=100):
    frame_path = 'synthesisVideo/0001_5.png'
    frame_num = 7

    img = cv2.imread(frame_path)
    h, w, ch = img.shape
    # Rotate around the X-axis
    r1_x, r1_y = np.random.randint(0, param_rotate_x_axis_1, 2)
    r2_x, r2_y = np.random.randint(0, param_rotate_x_axis_2, 2)
    if np.random.random() > 0.5:
        r1_x *= -1
        r2_x *= -1
    pts1 = np.float32([[0, 0],
                       [w - 1, 0],
                       [0, h - 1],
                       [w - 1, h - 1]])
    pts2 = np.float32([[0 + r1_x, 0 + r1_y],  # left Top
                       [w - 1 - r1_x, 0 + r1_y],  # right Top
                       [0 - r2_x, h - 1 - r2_y],  # left Bottom
                       [w - 1 + r2_x, h - 1 - r2_y]])  # right Bottom
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (w, h))
    cv2.imwrite('synthesisVideo/out/affineTransform.png', dst)


def example_AffineTrans():
    frame_path = 'synthesisVideo/0001_5.png'
    img = cv2.imread(frame_path)
    h, w, ch = img.shape
    # point(x,y)  leftTop, rightTop, leftDown
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])  # , [h - 1, w - 1]
    pts2 = np.float32([[0 + 200, 0 + 200], [w - 1 - 100, 0 + 100], [0 + 100, h - 1 - 100]])  # , [h - 1-100, w - 1-100]
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite('synthesisVideo/out/affineTransform.png', dst)


def example_PerspectiveTrans():
    frame_path = 'synthesisVideo/0001_5.png'
    img = cv2.imread(frame_path)
    h, w, ch = img.shape
    # point(x,y)  leftTop, rightTop, leftDown
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    a = 100
    b = 200
    pts2 = np.float32([[0 - b, 0 - a],
                       [w - 1 + b, 0 - a],
                       [0 + b, h - 1 - a],
                       [w - 1 - b, h - 1 - a]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # print(M)
    dst = cv2.warpPerspective(img, M, (w, h), borderValue=(114, 114, 114))
    cv2.imwrite('synthesisVideo/out/perspectiveTransform.png', dst)


def test():
    # test_Gyro()
    # example_AffineTrans()
    example_PerspectiveTrans()


def random_perspective(im,
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    # print('M', M)
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    return im


def random_augment(hyp='hyps_gen_syn.yaml',
                   out_dir='synthesisVideo/out/',
                   num_frames=1):
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    frame_path = 'synthesisVideo/0001_5.png'
    sub_folder = os.path.basename(frame_path).split('.')[0]  # dirname
    sub_dir = os.path.join(out_dir, sub_folder)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    for i in range(num_frames):
        image = random_perspective(frame,
                                   degrees=hyp['degrees'],
                                   translate=hyp['translate'],
                                   scale=hyp['scale'],
                                   shear=hyp['shear'],
                                   perspective=hyp['perspective'])
        cv2.imwrite(sub_dir + '/frame_{}.png'.format(i), image)


def get_meta_data(raw_path):
    content = ''
    meta = {}
    ###########################################################
    # ISP Meta data
    ###########################################################
    raw_obj = rawpy.imread(raw_path)
    black_level_per_channel = raw_obj.black_level_per_channel
    meta['black_level_per_channel'] = black_level_per_channel
    camera_white_level_per_channel = raw_obj.camera_white_level_per_channel
    meta['camera_white_level_per_channel'] = camera_white_level_per_channel
    camera_whitebalance = raw_obj.camera_whitebalance
    meta['camera_whitebalance'] = camera_whitebalance
    color_desc = raw_obj.color_desc
    meta['color_desc'] = color_desc
    color_matrix = raw_obj.color_matrix
    meta['color_matrix'] = color_matrix
    daylight_whitebalance = raw_obj.daylight_whitebalance
    meta['daylight_whitebalance'] = daylight_whitebalance
    num_colors = raw_obj.num_colors
    meta['num_colors'] = num_colors
    raw_pattern = raw_obj.raw_pattern
    meta['raw_pattern'] = raw_pattern
    raw_type = raw_obj.raw_type
    meta['raw_type'] = raw_type
    rgb_xyz_matrix = raw_obj.rgb_xyz_matrix
    meta['rgb_xyz_matrix'] = rgb_xyz_matrix
    sizes = raw_obj.sizes
    meta['sizes'] = sizes
    tone_curve = raw_obj.tone_curve
    meta['tone_curve'] = tone_curve
    white_level = raw_obj.white_level
    meta['white_level'] = white_level
    for k, v in meta.items():
        temp = '# ' + k + '\n' + str(v) + '\n'
        content += temp
    ###########################################################
    # CCM
    ###########################################################
    # rgb2cam = random_ccm(rgb2xyz=raw_obj.rgb_xyz_matrix[:-1, :])
    # rgb2cam = random_ccm()
    # Load the camera's (or image's) ColorMatrix2
    xyz2cam = raw_obj.rgb_xyz_matrix[:-1,
              :]  # torch.FloatTensor(np.reshape(np.asarray(info[info['camera'][0][i]]['ColorMatrix2']), (3, 3)))
    xyz2cam = torch.from_numpy(xyz2cam)
    # print(bayer_pattern, xyz2cam)
    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.inverse(torch.from_numpy(rgb2cam.numpy()))
    cam2rgb = cam2rgb.data.cpu().numpy()
    ###########################################################
    # norm
    ###########################################################
    black_level = black_level_per_channel[0]
    ###########################################################
    # Specify red and blue gains here (for White Balancing)
    ###########################################################
    red_gains, green_gains, blue_gains, _ = raw_obj.daylight_whitebalance
    # red_gains, green_gains, blue_gains, _ = raw_obj.camera_whitebalance
    # red_gains /= green_gains
    # blue_gains /= green_gains
    return cam2rgb, black_level, white_level, red_gains, green_gains, blue_gains, content



def get_random_perspective_Matrix(hs=[], ws=[],
                                  degrees=10,
                                  translate=.1,
                                  scale=.1,
                                  shear=10,
                                  perspective=0.0,
                                  border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    Ms = []
    assert len(hs) == len(ws)
    for i in range(len(hs)):
        h = hs[i]
        w = ws[i]

        height = h + border[0] * 2  # shape(h,w,c)
        width = w + border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -w / 2  # x translation (pixels)
        C[1, 2] = -h / 2  # y translation (pixels)
        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        Ms.append(M)
    return Ms


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    noise = tf.random_normal(tf.shape(image), stddev=tf.sqrt(variance))
    return image + noise

# dataset_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/'
def gen_DRV_video_perspective(hyp='hyps_gen_syn.yaml',
                  dataset_dir='/home/wen/Documents/dataset/DRV/trainval_v2',
                  height=3672,  # 1836  # 1080
                  width=5496,  # 2748   # 1920
                  num_frames=7):
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    print('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    bayer_pattern = 'RGGB'

    out_dir = dataset_dir + 'trainval_v2/'
    dataset_pattern_png = dataset_dir + 'long/*/*.png'
    dataset_pattern_raw = dataset_dir + 'long/*/*.raw'
    dataset_pattern_arw = dataset_dir + 'long/*/*.ARW'
    png_list = glob.glob(dataset_pattern_png)
    raw_list = glob.glob(dataset_pattern_raw)
    arw_list = glob.glob(dataset_pattern_arw)
    arw_list.sort()
    tmp = []
    for png in png_list:
        if 'half' in png or 'bin' in png:
            continue
        tmp.append(png)
    png_list = tmp
    png_list.sort()
    tmp = []
    for raw in raw_list:
        if 'bin' in raw:
            continue
        tmp.append(raw)
    raw_list = tmp
    raw_list.sort()
    # assert len(raw_list) == len(png_list) == len(arw_list)
    assert len(png_list) == len(arw_list)  # > len(raw_list)
    num = len(raw_list)
    px = nn.PixelShuffle(2)
    for i in range(num):
        if i > 20:
            break
        png_path = png_list[i]
        raw_path = raw_list[i]
        arw_path = arw_list[i]
        assert os.path.basename(png_path).split('.')[0] == os.path.basename(raw_path).split('.')[0] == \
               os.path.basename(arw_path).split('.')[0]
        print(i, num, arw_path)

        sub_folder = png_path.split('/')[-2]  # dirname
        sub_dir = os.path.join(out_dir, sub_folder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(sub_dir + '/raw'):
            os.makedirs(sub_dir + '/raw')
        if not os.path.exists(sub_dir + '/raw_bin'):
            os.makedirs(sub_dir + '/raw_bin')
        if not os.path.exists(sub_dir + '/png'):
            os.makedirs(sub_dir + '/png')

        # get meta data
        cam2rgb, black_level, white_level, red_gains, green_gains, blue_gains, content = get_meta_data(arw_path)
        # cam2rgb.squeeze().tofile(sub_dir + '/cam2rgb.raw')
        # cam2rgb = np.expand_dims(cam2rgb, axis=0)
        cam2rgb = None

        f = open(sub_dir + '/meta.data', 'w')
        f.write(content)
        f.close()

        raw = np.fromfile(raw_path, dtype=np.uint16)
        # raw.tofile(sub_dir + '/raw/frame_0.raw')
        np.save(sub_dir + '/raw/frame_0.npy', raw)
        raw = np.reshape(raw, (height, width))

        bSave_png = True
        if bSave_png:
            raw_norm = np.maximum(raw - black_level, 0) / (white_level - black_level)
            with torch.no_grad():
                raw_isp = tensor2numpy(isp(torch.from_numpy(
                pixel_unShuffle_RGBG(raw_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
                cv2.imwrite(sub_dir + '/raw/frame_0.png', np.uint8(raw_isp * 255))

        raw_bin = binning_raw(raw).astype(np.uint16)  # , black_level, white_level)
        # raw_bin.tofile(sub_dir + '/raw_bin/frame_0.raw')
        np.save(sub_dir + '/raw_bin/frame_0.npy', raw_bin)

        raw_bin_norm = np.maximum(raw_bin - black_level, 0) / (white_level - black_level)
        with torch.no_grad():
            tmp = pixel_unShuffle_RGBG(raw_bin_norm.astype(np.float32), bayer_pattern)
            tmp = np.pad(tmp,[(0,0), (0,0), (1,1), (1,1)])
            raw_bin_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(sub_dir + '/raw_bin/frame_0.png', np.uint8(raw_bin_isp * 255))

        frame = cv2.imread(png_path, cv2.IMREAD_COLOR)
        cv2.imwrite(sub_dir + '/png/frame_0.png', frame)
        hf, wf, _ = frame.shape

        for i in range(num_frames):
            Mraw, Mf = get_random_perspective_Matrix(hs=[height, hf],
                                                     ws=[width, wf],
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'],
                                                     perspective=hyp['perspective'])

            tmp = np.transpose(pixel_unShuffle(raw), (0, 2, 3, 1))  # 1,h,w,4
            raw_pers = cv2.warpPerspective(tmp[0],  # np.expand_dims(raw, axis=-1),
                                           Mf,  # Mraw,
                                           dsize=(wf, hf),  # (width, height),
                                           borderValue=(0, 0, 0, 0))
            t = np.transpose(np.expand_dims(raw_pers, axis=0), (0, 3, 1, 2)).astype(np.float32)
            raw_pers = px(torch.from_numpy(t)).squeeze().data.cpu().numpy().astype(np.uint16)
            # raw_pers.tofile(sub_dir + '/raw/frame_{}.raw'.format(i + 1))
            np.save(sub_dir + '/raw/frame_{}.npy'.format(i + 1), raw_pers)

            raw_pers_norm = np.maximum(raw_pers - black_level, 0) / (white_level - black_level)
            with torch.no_grad():
                raw_pers_isp = tensor2numpy(isp(torch.from_numpy(
                    pixel_unShuffle_RGBG(raw_pers_norm.astype(np.float32), bayer_pattern)).cuda()))[0]
                cv2.imwrite(sub_dir + '/raw/frame_{}.png'.format(i + 1), np.uint8(raw_pers_isp * 255))

            raw_pers_bin = binning_raw(raw_pers).astype(np.uint16)  # , black_level, white_level)
            # raw_pers_bin.tofile(sub_dir + '/raw_bin/frame_{}.raw'.format(i + 1))
            np.save(sub_dir + '/raw_bin/frame_{}.npy'.format(i + 1), raw_pers_bin)

            raw_pers_bin_norm = np.maximum(raw_pers_bin - black_level, 0) / (white_level - black_level)
            with torch.no_grad():
                tmp = pixel_unShuffle_RGBG(raw_pers_bin_norm.astype(np.float32), bayer_pattern)
                tmp = np.pad(tmp, [(0, 0), (0, 0), (1, 1), (1, 1)])
                raw_pers_bin_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                cv2.imwrite(sub_dir + '/raw_bin/frame_{}.png'.format(i + 1), np.uint8(raw_pers_bin_isp * 255))

            frame_pers = cv2.warpPerspective(frame,
                                             Mf,
                                             dsize=(wf, hf),
                                             borderValue=(114, 114, 114))
            cv2.imwrite(sub_dir + '/png/frame_{}.png'.format(i + 1), frame_pers)

def get_objs(obj_path,
             # image,
             raw_npy,
             hyp):
    px = nn.PixelShuffle(2)
    obj = cv2.imread(obj_path, cv2.IMREAD_COLOR)
    raw = raw_npy
    canvas = obj
    bPerspective = True
    if bPerspective:
        height, width, _ = obj.shape
        M = get_random_perspective_Matrix(hs=[height//2],
                                          ws=[width//2],
                                          degrees=hyp['degrees'],
                                          translate=hyp['translate'],
                                          scale=hyp['scale'],
                                          shear=hyp['shear'],
                                          perspective=hyp['perspective'])
        # image_pers = cv2.warpPerspective(image,
        #                                  M[0],
        #                                  dsize=(width, height),
        #                                  borderValue=(0, 0, 0))

        tmp = np.transpose(pixel_unShuffle(raw_npy), (0, 2, 3, 1))  # 1,h,w,4
        raw_pers = cv2.warpPerspective(tmp[0],
                                         M[0],
                                         dsize=(width//2, height//2),
                                         borderValue=(0, 0, 0))
        t = np.transpose(np.expand_dims(raw_pers, axis=0), (0, 3, 1, 2)).astype(np.float32)
        raw = px(torch.from_numpy(t)).squeeze().data.cpu().numpy().astype(np.uint16)

        tmp = cv2.resize(obj, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
        tmp = cv2.warpPerspective(tmp,
                                     M[0],
                                     dsize=(width//2, height//2),
                                     borderValue=(0, 0, 0))
        canvas = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_LINEAR)

    mask = np.mean(canvas, axis=-1)
    mask[mask>0] = 1
    mask = mask.astype(np.uint16)
    idx = np.where(mask==1)
    idx_x = idx[1]
    idx_y = idx[0]
    x_min = np.min(idx_x) // 2 * 2  # // 4 * 4
    x_max = np.max(idx_x)
    y_min = np.min(idx_y) // 2 * 2  # // 4 * 4
    y_max = np.max(idx_y)
    tmp = mask[y_min:y_max, x_min:x_max]
    # cv2.imwrite(obj_path.split('.')[0]+'_mask.png', np.stack([tmp,tmp,tmp],axis=-1).astype(np.int)*255)
    # return tmp, x_min, x_max, y_min, y_max
    # crop = image_pers[y_min:y_max + 1, x_min:x_max + 1, :] * np.stack([tmp, tmp, tmp], axis=-1)
    crop_npy = raw[y_min:y_max, x_min:x_max] * tmp
    # return crop, crop_npy
    return crop_npy


def paste_obj_to_raw(image, raw_npy, obj_path,
                     sub_dir, name, num_frames,
                     hyp='hyps_gen_syn.yaml',
                     black_level=64,
                     white_level=2**10-1):
    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    H, W, _ = image.shape
    # r = np.random.randint(11, 17)
    # for i in range(r):
    xc = np.random.randint(0, W) # np.random.randint(0, (W - w * 2 + 1) / 2) * 2
    yc = np.random.randint(0, H) # np.random.randint(0, (H - h * 2 + 1) / 2) * 2
    delta1 = 100
    delta2 = 1000
    xflag = 1
    yflag = 1
    with torch.no_grad():
        for i in range(1, num_frames+1):
            # crop, crop_npy = get_objs(obj_path, image, raw_npy, hyp)
            crop_npy = get_objs(obj_path, raw_npy, hyp)
            # h, w, _ = crop.shape
            h, w = crop_npy.shape

            # canvas = image.copy()
            canvas_npy = raw_npy.copy()

            x1 = np.maximum(xc - w // 2, 0)
            x1 = x1 // 2 * 2  # // 4 * 4
            x2 = np.minimum(xc + w // 2 + w % 2, W)
            x2 = np.minimum(x2, x1 + w)

            y1 = np.maximum(yc - h // 2, 0)
            y1 = y1 // 2 * 2  # // 4 * 4
            y2 = np.minimum(yc + h // 2 + h % 2, H)
            y2 = np.minimum(y2, y1 + h)

            xx1 = np.maximum(w // 2 - (xc - x1), 0)
            xx1 = xx1 // 2 * 2  # // 4 * 4
            xx2 = xx1 + (x2 - x1)
            # xx2 = w // 2 + (x2 - xc) # + (w + 1) % 2

            yy1 = np.maximum(h // 2 - (yc - y1), 0)
            yy1 = yy1 // 2 * 2  # // 4 * 4
            yy2 = yy1 + (y2 - y1)
            # yy2 = h // 2 + (y2 - yc) # + (h+1) % 2

            # bk = canvas[y1:y2, x1:x2, :]
            # obj = crop[yy1: yy2, xx1: xx2, :]
            # canvas[y1:y2, x1:x2, :] = np.where(obj > 0, obj, bk)
            bk = canvas_npy[y1:y2, x1:x2]
            obj = crop_npy[yy1: yy2, xx1: xx2]
            canvas_npy[y1:y2, x1:x2] = np.where(obj > 0, obj, bk)
            np.save(os.path.join(sub_dir, 'raw/{}{}.npy'.format(name, i)), canvas_npy)
            tmp = norm_raw(canvas_npy, black_level, white_level)
            tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
            raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(os.path.join(sub_dir, 'raw/{}{}.png'.format(name, i)),
                        np.uint8(raw_isp * 255))
            # cv2.imwrite(os.path.join(sub_dir,  'raw/{}{}.png'.format(name, i)), canvas)

            raw_bin = binning_raw(canvas_npy).astype(np.uint16)
            np.save(os.path.join(sub_dir, 'raw_bin/{}{}.npy'.format(name, i)), raw_bin)
            tmp = norm_raw(raw_bin, black_level, white_level)
            tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
            tmp = np.pad(tmp,[(0,0), (0,0), (1,1), (1,1)])
            raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(os.path.join(sub_dir, 'raw_bin/{}{}.png'.format(name, i)),
                        np.uint8(raw_isp * 255))
            # cv2.imwrite(os.path.join(sub_dir, 'raw_bin/{}{}.png'.format(name, i)), raw_bin)

            xdelta = np.random.randint(delta1, delta2)
            ydelta = np.random.randint(delta1, delta2)

            if xc + xdelta * xflag >= W:
                xflag = -1
            if xc + xdelta * xflag <= 0:
                xflag = 1
            xc += xdelta * xflag

            if yc + ydelta * yflag >= H:
                yflag = -1
            if yc + ydelta * yflag <= 0:
                yflag = 1
            yc += ydelta * yflag
        # return canvas, canvas_npy

def gen_perspective(raw_npy,
                    hyp,
                    name,
                    sub_dir,
                    num_frames=7,
                    black_level=64,
                    white_level=2 ** 10 - 1
                    ):
    px = nn.PixelShuffle(2)
    raw_npy_pxu = pixel_unShuffle(raw_npy)
    raw_npy_pxu_chL = np.transpose(raw_npy_pxu, (0, 2, 3, 1))  # 1,h,w,4
    _, h, w, _ = raw_npy_pxu_chL.shape
    with torch.no_grad():
        for i in range(1, num_frames+1):
            M = get_random_perspective_Matrix(hs=[h],
                                                 ws=[w],
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
            raw_pers = cv2.warpPerspective(raw_npy_pxu_chL[0],
                                           M[0],
                                           dsize=(w, h),
                                           borderValue=(0, 0, 0, 0))
            t = np.transpose(np.expand_dims(raw_pers, axis=0), (0, 3, 1, 2)).astype(np.float32)
            raw_pers = px(torch.from_numpy(t)).squeeze().data.cpu().numpy().astype(np.uint16)
            np.save(os.path.join(sub_dir, 'raw/{}{}.npy'.format(name, i)), raw_pers)
            tmp = norm_raw(raw_pers, black_level, white_level)
            tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
            raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(os.path.join(sub_dir, 'raw/{}{}.png'.format(name, i)),
                        np.uint8(raw_isp * 255))

            raw_bin = binning_raw(raw_pers).astype(np.uint16)
            np.save(os.path.join(sub_dir, 'raw_bin/{}{}.npy'.format(name, i)), raw_bin)
            tmp = norm_raw(raw_bin, black_level, white_level)
            tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
            tmp = np.pad(tmp,[(0,0), (0,0), (1,1), (1,1)])
            raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(os.path.join(sub_dir, 'raw_bin/{}{}.png'.format(name, i)),
                        np.uint8(raw_isp * 255))
            # cv2.imwrite(os.path.join(sub_dir, 'raw_bin/{}{}.png'.format(name, i)), raw_bin)

            # canvas = cv2.warpPerspective(image,
            #                              Mf,
            #                              dsize=(width, height),
            #                              borderValue=(114, 114, 114))
            # cv2.imwrite(os.path.join(sub_dir,  'raw/{}{}.png'.format(name, i)), canvas)

def get_DRV_data_thumb():
    dataset_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/long'
    pattern = dataset_dir + '/*/half*.png'
    out_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/thumb/'
    png_list = glob.glob(pattern)
    for image in png_list:
        name = os.path.basename(image)
        sub_folder = os.path.dirname(image).split('/')[-1]
        img_name = '_'.join([sub_folder, name])
        shutil.copy(image, out_dir + img_name)

def gen_DRV_video(dataset_dir='/home/wen/Documents/dataset/DRV/',
                  sub_folder = 'trainval_v3/', # 'trainval_v2/'
                  height=3672,
                  width=5496,
                  black_level = 800,
                  white_level = 16380,
                  bayer_pattern = 'RGGB',
                  hyp='hyps_gen_syn.yaml',
                  num_frames=7):
    out_dir = dataset_dir + sub_folder

    dataset_pattern_png = out_dir + '*/raw/frame_0.png'
    dataset_pattern_npy = out_dir + '*/raw/frame_0.npy'
    # dataset_pattern_obj = out_dir + '*/frame_0_obj.png'

    png_list = glob.glob(dataset_pattern_png)
    npy_list = glob.glob(dataset_pattern_npy)
    # obj_list = glob.glob(dataset_pattern_obj)

    png_list.sort()
    npy_list.sort()
    # obj_list.sort()

    list_obj_motion = []

    list_rotate = []
    list_translate = []
    list_scale = []
    list_perspective = []
    list_all = []

    num = 50
    for i in range(num):
        png_path = png_list[i]
        npy_path = npy_list[i]

        # obj_path = obj_list[i]
        obj_path = os.path.dirname(os.path.dirname(png_path)) + '/frame_0_obj.png'

        if os.path.exists(obj_path):
            video_type = 'obj_motion'
        else:
            video_type = 'camera_motion'

        scene_folder = png_path.split('/')[-3]  # dirname
        print(scene_folder)
        sub_dir = os.path.join(out_dir, scene_folder, video_type)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(sub_dir + '/raw'):
            os.makedirs(sub_dir + '/raw')
        if not os.path.exists(sub_dir + '/raw_bin'):
            os.makedirs(sub_dir + '/raw_bin')

        image = cv2.imread(png_path, cv2.IMREAD_COLOR)
        raw_npy = np.load(npy_path, mmap_mode='r')
        if len(raw_npy.shape)<2:
            raw_npy = np.reshape(raw_npy, (height, width))
        name = 'frame_'
        #######################################################
        # scale to 10 bit & GBRG
        #######################################################
        black_level_10 = 64
        white_level_10 = 2 ** 10 - 1
        raw_npy_norm = norm_raw(raw_npy, black_level, white_level)
        raw_npy_norm_10bit = raw_npy_norm * (white_level_10 - black_level_10) + black_level_10
        # raw_npy_norm_10bit = raw_npy_norm_10bit.astype(np.uint16)
        raw = convert_to_GBRG(raw_npy_norm_10bit, bayer_pattern)
        #######################################################
        tmp = norm_raw(raw, black_level_10, white_level_10)
        tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
        with torch.no_grad():
            raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(os.path.join(sub_dir, 'raw/{}0.png'.format(name)),
                        np.uint8(raw_isp * 255))
        np.save(os.path.join(sub_dir, 'raw/{}0.npy'.format(name)), raw)

        # origin bit, before convert
        cv2.imwrite(os.path.join(sub_dir, '{}0.png'.format(name)), image)

        raw_bin = binning_raw(raw).astype(np.uint16)
        np.save(os.path.join(sub_dir, 'raw_bin/{}0.npy'.format(name)), raw_bin)
        tmp = norm_raw(raw_bin, black_level_10, white_level_10)
        tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
        tmp = np.pad(tmp, [(0, 0), (0, 0), (1, 1), (1, 1)])
        with torch.no_grad():
            raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
            cv2.imwrite(os.path.join(sub_dir, 'raw_bin/{}0.png'.format(name)),
                        np.uint8(raw_isp * 255))
        # cv2.imwrite(os.path.join(sub_dir, 'raw_bin/{}0.png'.format(name)), raw_bin)

        if video_type == 'obj_motion':
            list_obj_motion.append(scene_folder)
            paste_obj_to_raw(image, raw, obj_path,
                             sub_dir, name, num_frames)
        elif video_type == 'camera_motion':
            # Hyperparameters
            if isinstance(hyp, str):
                with open(hyp, errors='ignore') as f:
                    hyp = yaml.safe_load(f)  # load hyps dict

            degrees = 0.0
            translate = 0.0
            scale = 0.0
            perspective = 0.0

            seed = np.random.random()
            #######################################################
            # rotation
            #######################################################
            if seed < 0.2:
                degrees = 5.0
                list_rotate.append(scene_folder)
            elif seed >= 0.2 and seed < 0.4:
                translate = 0.05
                list_translate.append(scene_folder)
            elif seed >= 0.4 and seed < 0.6:
                scale = 0.05
                list_scale.append(scene_folder)
            elif seed >= 0.6 and seed < 0.8:
                perspective = 0.00005
                list_perspective.append(scene_folder)
            else:
                degrees = 5.0
                translate = 0.05
                scale = 0.05
                perspective = 0.00005
                list_all.append(scene_folder)

            hyp['degrees'] = degrees
            hyp['translate'] = translate
            hyp['scale'] = scale
            hyp['perspective'] = perspective

            # content = 'hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items())
            # print(content)
            #
            # f = open(sub_dir + '/perspective_hyper_param.data', 'w')
            # f.write(content)
            # f.close()

            gen_perspective(raw, hyp, name, sub_dir)
    print('obj_motion                   ', list_obj_motion)
    print('camera_motion,   rotate      ', list_rotate)
    print('camera_motion,   translate   ', list_translate)
    print('camera_motion,   scale       ', list_scale)
    print('camera_motion,   perspective ', list_perspective)
    print('camera_motion,   all         ', list_all)

    print('end')


def get_obj_mask():
    mask_path = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/trainval_v2/0008/frame_0_obj.png'
    obj = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    mask = np.mean(obj, axis=-1)
    mask[mask > 0] = 1
    tmp = np.stack([mask,mask,mask],axis=-1).astype(np.uint8) * 255
    cv2.imwrite(mask_path, tmp)

def raw2npy():
    data_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/DRV/trainval_v2/'
    height = 3672
    width = 5496
    for i in range(40, 51):
        sub_folder = '{:0>4d}'.format(i)
        raw_path = data_dir + sub_folder + '/raw/frame_0.raw'
        raw = np.fromfile(raw_path, dtype=np.uint16)
        raw = raw.reshape(height, width)
        np.save(raw_path.split('.')[0]+'.npy', raw)

###################################################################
# CRVD
###################################################################
def gen_CRVD_video():
    hyp = 'hyps_gen_syn.yaml'
    data_root = '/media/wen/C14D581BDA18EBFA/work/dataset/CRVD_dataset'
    iso_list = [1600, 3200, 6400, 12800, 25600]
    black_level = 240
    white_level = 2 ** 12 -1
    name = 'frame_'
    for scene_ind in range(1,12):
        for iso_ind in iso_list:
            for frame_ind in range(1,8):
                gt_name = os.path.join(data_root,
                                       'indoor_raw_gt/indoor_raw_gt_scene{}/scene{}/ISO{}/'
                                       'frame{}_clean_and_slightly_denoised.tiff'.format(
                                           scene_ind, scene_ind, iso_ind, frame_ind))
                raw = cv2.imread(gt_name, -1)
                #######################################################
                # object motion
                #######################################################
                out_dir_obj = data_root + '/synthesis/obj_motion/scene{}/ISO{}'.format(scene_ind, iso_ind)
                if not os.path.exists(out_dir_obj):
                    os.makedirs(out_dir_obj)
                if not os.path.exists(out_dir_obj + '/raw'):
                    os.makedirs(out_dir_obj + '/raw')
                if not os.path.exists(out_dir_obj + '/raw_bin'):
                    os.makedirs(out_dir_obj + '/raw_bin')
                np.save(out_dir_obj+'/raw/frame_{}.npy'.format(frame_ind), raw)
                raw_bin = binning_raw(raw).astype(np.uint16)
                np.save(os.path.join(out_dir_obj, 'raw_bin/{}{}.npy'.format(name, frame_ind)), raw_bin)
                if frame_ind == 1:
                    tmp = norm_raw(raw_bin, black_level, white_level)
                    tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
                    tmp = np.pad(tmp, [(0, 0), (0, 0), (1, 1), (0, 0)])
                    raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(os.path.join(out_dir_obj, 'raw_bin/{}{}.png'.format(name, frame_ind)),
                                np.uint8(raw_isp * 255))
                #######################################################
                # camera motion
                #######################################################
                out_dir_cam = data_root + '/synthesis/cam_motion/scene{}/ISO{}'.format(scene_ind, iso_ind)
                # np.save(out_dir_cam+'/raw/frame_{}.npy'.format(frame_ind), raw)
                if not os.path.exists(out_dir_cam):
                    os.makedirs(out_dir_cam)
                if not os.path.exists(out_dir_cam + '/raw'):
                    os.makedirs(out_dir_cam + '/raw')
                if not os.path.exists(out_dir_cam + '/raw_bin'):
                    os.makedirs(out_dir_cam + '/raw_bin')
                px = nn.PixelShuffle(2)
                raw_npy_pxu = pixel_unShuffle(raw)
                raw_npy_pxu_chL = np.transpose(raw_npy_pxu, (0, 2, 3, 1))  # 1,h,w,4
                _, h, w, _ = raw_npy_pxu_chL.shape
                # Hyperparameters
                if isinstance(hyp, str):
                    with open(hyp, errors='ignore') as f:
                        hyp = yaml.safe_load(f)  # load hyps dict
                degrees = 5.0
                translate = 0.05
                scale = 0.05
                perspective = 0.00005
                hyp['degrees'] = degrees
                hyp['translate'] = translate
                hyp['scale'] = scale
                hyp['perspective'] = perspective
                M = get_random_perspective_Matrix(hs=[h],
                                                  ws=[w],
                                                  degrees=hyp['degrees'],
                                                  translate=hyp['translate'],
                                                  scale=hyp['scale'],
                                                  shear=hyp['shear'],
                                                  perspective=hyp['perspective'])
                raw_pers = cv2.warpPerspective(raw_npy_pxu_chL[0],
                                               M[0],
                                               dsize=(w, h),
                                               borderValue=(0, 0, 0, 0))
                t = np.transpose(np.expand_dims(raw_pers, axis=0), (0, 3, 1, 2)).astype(np.float32)
                raw_pers = px(torch.from_numpy(t)).squeeze().data.cpu().numpy().astype(np.uint16)
                np.save(os.path.join(out_dir_cam, 'raw/{}{}.npy'.format(name, frame_ind)), raw_pers)
                if frame_ind == 1:
                    tmp = norm_raw(raw_pers, black_level, white_level)
                    tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
                    raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(os.path.join(out_dir_cam, 'raw/{}{}.png'.format(name, frame_ind)),
                                np.uint8(raw_isp * 255))

                raw_bin = binning_raw(raw_pers).astype(np.uint16)
                np.save(os.path.join(out_dir_cam, 'raw_bin/{}{}.npy'.format(name, frame_ind)), raw_bin)
                if frame_ind == 1:
                    tmp = norm_raw(raw_bin, black_level, white_level)
                    tmp = pixel_unShuffle_RGBG(tmp, bayer_pattern='GBRG')
                    tmp = np.pad(tmp, [(0, 0), (0, 0), (1, 1), (0, 0)])
                    raw_isp = tensor2numpy(isp(torch.from_numpy(tmp).cuda()))[0]
                    cv2.imwrite(os.path.join(out_dir_cam, 'raw_bin/{}{}.png'.format(name, frame_ind)),
                                np.uint8(raw_isp * 255))


###################################################################
def main():
    setup_seed(666)

    gen_CRVD_video()

    # gen_DRV_video()

    # raw2npy()
    # get_obj_mask()
    # test()
    # random_augment()

    print('end')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# raw_norm = np.maximum(raw - black_level, 0) / (white_level - black_level)
# raw_norm = np.clip(raw_norm, 0, 1)
# cv2.imwrite(sub_dir + '/raw/frame_0.png', np.uint8(raw_norm * 255))
# raw_norm_pxu_chLast = np.transpose(pixel_unShuffle(raw_norm), (0, 2, 3, 1))
# raw_process = process(raw_norm_pxu_chLast, [red_gains], [blue_gains], cam2rgb)  # tf code
# cv2.imwrite(sub_dir + '/raw/frame_0_process.png', np.uint8(raw_process[0].numpy() * 255))
# raw_process_sim = process_simple(raw, black_level, white_level, red_gains, green_gains, blue_gains)
# cv2.imwrite(sub_dir + '/raw/frame_0_process_sim.png', np.uint8(raw_process_sim * 255))
# raw_process_sim_save = Image.fromarray(np.uint8(raw_process_sim * 255))
# raw_process_sim_save.save(sub_dir + '/raw/frame_0_process_sim.png')
