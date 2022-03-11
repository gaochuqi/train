import cv2
import math
import numpy as np
from PIL import Image
# from demo_actual import *


def PSNR(a, b):
    return 10 * np.log(1 / (np.mean(np.square(a - b))))


def unpack_4raw_to_1raw(im):
    img_shape = im.shape
    H = img_shape[1] * 2
    W = img_shape[2] * 2
    out = np.zeros((H, W))
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            out[i,   j  ] = im[0][i//2][j//2]
            out[i,   j+1] = im[1][i//2][j//2]
            out[i+1, j  ] = im[2][i//2][j//2]
            out[i+1, j+1] = im[3][i//2][j//2]
    return out


def pack_1raw_to_4raw(im):
    '''
    block unpixel shuffle
    '''
    im = np.squeeze(im)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    out1, out2, out3, out4 = np.zeros((1, H//2, W//2)), np.zeros((1, H//2, W//2)),\
                             np.zeros((1, H//2, W//2)), np.zeros((1, H//2, W//2))

    for i in range(0, H, 4):
        for j in range(0, W, 4):
            out1[0][i // 2, j // 2] = im[i, j]
            out1[0][i // 2, j // 2 + 1] = im[i, j + 1]
            out1[0][i // 2 + 1, j // 2] = im[i + 1, j]
            out1[0][i // 2 + 1, j // 2 + 1] = im[i + 1, j + 1]

            out2[0][i // 2, j // 2] = im[i, j + 2]
            out2[0][i // 2, j // 2 + 1] = im[i, j + 3]
            out2[0][i // 2 + 1, j // 2] = im[i + 1, j + 2]
            out2[0][i // 2 + 1, j // 2 + 1] = im[i + 1, j + 3]

            out3[0][i // 2, j // 2] = im[i + 2, j]
            out3[0][i // 2, j // 2 + 1] = im[i + 2, j + 1]
            out3[0][i // 2 + 1, j // 2] = im[i + 3, j]
            out3[0][i // 2 + 1, j // 2 + 1] = im[i + 3, j + 1]

            out4[0][i // 2, j // 2] = im[i + 2, j + 2]
            out4[0][i // 2, j // 2 + 1] = im[i + 2, j + 3]
            out4[0][i // 2 + 1, j // 2] = im[i + 3, j + 2]
            out4[0][i // 2 + 1, j // 2 + 1] = im[i + 3, j + 3]

    out = np.concatenate((out1, out2, out3, out4), axis=0)
    return out


if __name__ == "__main__":
    # img1 = np.load('/home/tyy/Project/Datasets/CRVD_DAVIS/CRVD_all_noise_npy/indoor_raw_noisy_scene1_ISO1600_frame1_noisy0.npy')
    # for i in range(1, 10):
    #     img2 = np.load('/home/tyy/Project/Datasets/CRVD_DAVIS/CRVD_all_gt_npy/indoor_raw_gt_scene1_ISO1600_frame1_clean_and_slightly_denoised.npy'.format(i))
    #     img2_1raw = unpack_4raw_to_1raw(img2)

    #     img2_1raw_4raw = pack_1raw_to_4raw(img2_1raw)


    #     for i in range(4):
    #         psnr = PSNR(img2_1raw_4raw[0], img2_1raw_4raw[i])
    #         cur_img = img2_1raw_4raw[i]
    #         # 可视化
    #         A_N_img = np.concatenate([np.expand_dims(cur_img, 2), np.expand_dims(cur_img, 2), np.expand_dims(cur_img, 2)], axis=2)
    #         A_N_img = A_N_img * 255
    #         im = Image.fromarray(np.uint8(A_N_img))
    #         im.show()
    #         print(psnr)
    #     print("**********************")

    raw = np.fromfile('new/dataset/Mi11Ultra/20211221193515_0/VideoNight_Time_1640086515373.000000_FrameID_00000001_width_4080_height_3072_Input.raw', dtype=np.uint16)
    H = 3072
    W = 4080
    raw = raw.reshape(H, W)
    noisy_raw = np.maximum(raw - black_level, 0) / (white_level - black_level)  # 0 - 1
    raw = raw_gbrg2rgbg(noisy_raw, black_level, white_level, r_gain, g_gain, b_gain)
    A_N = rgbg_to_rgb(raw)
    im = Image.fromarray(np.uint8((A_N * 255)))