import warnings
warnings.filterwarnings('ignore')
from torch.serialization import SourceChangeWarning
warnings.filterwarnings('ignore', category=SourceChangeWarning)

import os
import glob
from pathlib import Path
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
# from demo_actual import raw_gbrg2rgbg, rgbg_to_rgb
from PIL import Image

def read_mipi10(image_root,image_height, image_width):
    """
    Function:
        read mipi10 raw image data
    """

    if not Path(image_root).is_file():
        raise Exception("image_root is not raw image file!!!!")
    with open(image_root, 'rb') as f:
        rawData = f.read()
    print("raw image length is", len(rawData))
    # validWidth mean theoretical size of a row pixels in mipi10 format
    validWidth = int(image_width * 10 / 8)
    if validWidth % 5 != 0:
        raise Exception("image_width may have error, please check image_width")
    # stride mean acutal size of a row pixels in mipi10 format, need alignment with 8
    if (validWidth % 8) != 0:
        stride = validWidth + 8 - (validWidth % 8)
    else:
        stride = validWidth
    print("raw image validWidth:", validWidth)
    print("raw image stride:", stride)
    # judge that the amount of reading data is consistent with the amount of inputing data
    if len(rawData) != stride * image_height:
        raise Exception("File size is not match mipi10, please check image_width and image_height")
    rawNpArray = np.asarray(list(rawData))
    # remove stride/alignment
    rawNpArray = rawNpArray.reshape([image_height, stride])[:, : validWidth]
    # print(rawNpArray.shape)
    # self.image_pixel is used to store the value of each pixel, data type must int16
    image_pixel = np.zeros([image_height, image_width], dtype=np.int16)
    for j in range(image_height):
        for i in range(0, int(validWidth / 5)):
            image_pixel[j][i * 4] = ((rawNpArray[j][i * 5]) << 2) + (((rawNpArray[j][i * 5 + 4])) & 0x3)
            image_pixel[j][i * 4 + 1] = ((rawNpArray[j][i * 5 + 1]) << 2) + (
                        ((rawNpArray[j][i * 5 + 4]) >> 2) & 0x3)
            image_pixel[j][i * 4 + 2] = ((rawNpArray[j][i * 5 + 2]) << 2) + (
                        ((rawNpArray[j][i * 5 + 4]) >> 4) & 0x3)
            image_pixel[j][i * 4 + 3] = ((rawNpArray[j][i * 5 + 3]) << 2) + (
                        ((rawNpArray[j][i * 5 + 4]) >> 6) & 0x3)
    print("read raw image succeess!!!!")
    return image_pixel

def mipi10raw(img_path, H=3072, W=4080):
    image_dir = '/home/wen/Documents/dataset/Mi11Ultra/'
    sub = '20211230/test/'
    name = os.path.basename(img_path).split('.')[0]
    image_pixel = read_mipi10(img_path,H,W)
    image_pixel.tofile(image_dir+sub+'%s.raw' % name)
    return image_pixel, name

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def rawmipi2uint16(rawmipi_dir, iso_list, H=3072,W=4080):
    for iso in iso_list:
        # subfolder = 'iso{}ex*/camera/'.format(str(iso))
        subfolder = 'iso{}/*/'.format(str(iso))
        sub_dir = rawmipi_dir + subfolder + '*.RAWMIPI10'
        RAWMIPI10_list = glob.glob(sub_dir)
        RAWMIPI10_list.sort()
        for i, img_path in enumerate(RAWMIPI10_list):
            out_dir = os.path.join(os.path.dirname(img_path),'raw')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = os.path.basename(img_path).split('.')[0]
            image_pixel = read_mipi10(img_path, H, W)
            image_pixel.tofile(out_dir + '/%s.raw' % name)

def raw_gbrg2rgbg(raw,black_level=64.0,white_level=1023,r_gain=1.0, g_gain=1.0, b_gain=1.0):
    # im = np.maximum(raw - black_level, 0) / (white_level - black_level)  # subtract the black level
    im = np.expand_dims(raw, axis=2)
    img_shape = im.shape
    H, W = img_shape[0], img_shape[1]
    c = 2
    out = np.concatenate((im[1:H:c, 0:W:c, :] * r_gain,  # r
                          im[1:H:c, 1:W:c, :] * g_gain,  # g
                          im[0:H:c, 1:W:c, :] * b_gain,  # b
                          im[0:H:c, 0:W:c, :] * g_gain), # g
                         axis=2)
    return out

def rgbg_to_rgb(image):
    image_r = image[:, :, 0:1]
    image_g1 = image[:, :, 1:2]
    image_b = image[:, :, 2:3]
    image_g2 = image[:, :, 3:4]
    image = np.concatenate((image_r, (image_g1+image_g2)/2, image_b), axis=2)
    return image

def statisitc(rawmipi_dir, iso_list, H, W):
    coef_a = []
    coef_b = []
    for iso in iso_list:
        mean_var_dict = {}
        if iso <= 200:
            lum_list = [15]
        elif iso <= 800:
            lum_list = [25]
        else:
            lum_list = [35]
        for lum in lum_list:
            raw_list = []
            # subfolder = 'iso{}ex*/camera/raw/'.format(str(iso))
            subfolder = 'iso{}/{}/raw/'.format(str(iso), str(lum))
            # subfolder = 'iso{}/raw/'.format(str(iso))
            sub_dir = rawmipi_dir + subfolder + '*.raw'
            RAW_list = glob.glob(sub_dir)
            meta_list = glob.glob(rawmipi_dir + 'iso{}/{}/*_BPS0_meta.txt'.format(str(iso), str(lum)))
            # meta_list = glob.glob(rawmipi_dir + 'iso{}/*_BPS0_meta.txt'.format(str(iso)))
            # meta_list = glob.glob(rawmipi_dir + 'iso{}ex*/camera/*_BPS0_meta.txt'.format(str(iso)))
            # print(len(RAW_list), len(meta_list))
            # print(RAW_list)
            assert len(RAW_list) == len(meta_list)
            
            RAW_list.sort()
            meta_list.sort()

            coef = 1.0
            black_level = 64.0
            white_level = 2**10-1
            
            luminous_dict = {}
            for i, raw_path in enumerate(RAW_list):
                if i >= 20:
                    break

                raw = np.fromfile(raw_path, dtype=np.uint16)
                raw = np.reshape(raw, (H, W)).astype(np.float32)
                # raw = raw[700:-800, 520:-600] # k1_2007
                # raw = raw[1560:-1300, 540:-800] # k1_yushu
                # raw = raw[700:-840, 740:-1040] # 2022 01 25
                raw = raw[1260:-1560, 260:-160] # 0126
                raw_list.append(raw)

                # meta_path = meta_list[i]
                # # name = '_'.join(os.path.basename(img_path).split('_')[:3])
                # # name = os.path.basename(raw_path).split('.')[0]
                # name = raw_path.split('.')[0]

                # f = open(meta_path,'r')
                # for line in f.readlines():
                #     line = line.strip()
                #     if 'gain_r' in line:
                #         # r_gain, g_gain, b_gain = [float(i) for i in line.split(' ')[-1].split(',')]
                #         r_gain = float(line.split('>')[1].split('<')[0])
                #     if 'gain_g' in line:
                #         g_gain = float(line.split('>')[1].split('<')[0])
                #     if 'gain_b' in line:
                #         b_gain = float(line.split('>')[1].split('<')[0])
                # f.close()
                
                # noisy_raw = np.maximum(raw - black_level, 0) / (white_level - black_level)  # 0 - 1
                # raw_frame = raw_gbrg2rgbg(noisy_raw, black_level, white_level, r_gain, g_gain, b_gain)
                # A_N = rgbg_to_rgb(raw_frame)
                # im = Image.fromarray(np.uint8((A_N * 255 * coef)))
                # # im.save(rawmipi_dir + subfolder + '%s.png' % name)
                # im.save('%s.png' % name)
        
            #################################
            raws = np.stack(raw_list, axis=0) # 10, H, W
            # raws = (raws - np.min(raws))
            raws = np.maximum(raws - black_level, 0)
            # raws = (raws - np.min(raws)) / (np.max(raws) - np.min(raws)) * 1024
            # print(np.min(raws), np.max(raws))
            # raws = (raws - np.min(raws)) / (np.max(raws) - np.min(raws)) * 255
            # raws = np.maximum(raws - black_level, 0.) / (white_level - black_level) * 255
            mean_raw = np.mean(raws, axis=0)
            mean_raw = np.round(mean_raw)
            intensity = np.unique(mean_raw)
            
            thresh = {50: 50, 100: 70, 200: 130, 400: 200, 800: 200, 1600: 200, 3200: 200}
            for luminous in intensity:
                if luminous > thresh[iso]:
                    continue
                indices = np.expand_dims(mean_raw == luminous, 0).repeat(20, axis=0)
                tmp = raws[indices].tolist()
                variance = np.var(tmp)
                if mean_var_dict.get(luminous, None) is None:
                    mean_var_dict[luminous] = [(variance, len(tmp))]
                else:
                    mean_var_dict[luminous].extend([(variance, len(tmp))])
        
        x = []
        y = []
        for k, v in mean_var_dict.items():
            x.append(k)
            var_total = 0.
            num_total = 0.
            for var, num in v:
                var_total += var * num
                num_total += num
            y.append(var_total / num_total)

        coef = np.polyfit(x, y, 1)
        print(iso, coef)
        coef_a.append(coef[0])
        coef_b.append(coef[1])

        plt.figure()
        plt.scatter(x, y)
        p = np.poly1d(coef)
        xp = np.linspace(0, thresh[iso], 1000)
        _ = plt.plot(x, y, '.', xp, p(xp), '-')
        plt.title('ISO {}'.format(iso))
        plt.xlabel("mean value")
        plt.ylabel("variance")
        plt.savefig('mean-variance iso{}.png'.format(iso))
    return iso_list, coef_a, coef_b

def fit_coef_ab(iso_list, coef_a, coef_b):
    coef_coef_a = np.polyfit(iso_list, coef_a, 2)
    func_coef_a = np.poly1d(coef_coef_a)
    coef_coef_b = np.polyfit(iso_list, coef_b, 2)
    func_coef_b = np.poly1d(coef_coef_b)
    plt.subplot(1, 2, 1)
    plt.scatter(iso_list, coef_a, color="red")
    plt.title("coeff a")
    plt.xlabel("iso")
    plt.ylabel("a")
    x_s = np.arange(0, 3200)
    plt.plot(x_s, func_coef_a(x_s), color="green")
    plt.subplot(1, 2, 2)
    plt.scatter(iso_list, coef_b, color="red")
    plt.title("coeff b")
    plt.xlabel("iso")
    plt.ylabel("b")
    x_s = np.arange(0, 3200)
    plt.plot(x_s, func_coef_b(x_s), color="green")
    plt.suptitle("coeff")
    plt.savefig('coeff.png')
    plt.show()
    return func_coef_a, func_coef_b

def get_coef_ab(iso_list, coef_a, coef_b, iso):
    func_coef_a, func_coef_b = fit_coef_ab(iso_list, coef_a, coef_b)
    a = func_coef_a(iso)
    b = func_coef_b(iso)
    return a, b

def calibrate(rawmipi_dir, iso_list, H, W):
    # rawmipi2uint16(rawmipi_dir, iso_list, H, W)
    iso_list, coef_a, coef_b = statisitc(rawmipi_dir, iso_list, H, W)
    func_coef_a, func_coef_b = fit_coef_ab(iso_list, coef_a, coef_b)
    return func_coef_a, func_coef_b
    return None, None

def main():
    setup_seed(666)
    rawmipi_dir = '0126/'
    iso_list = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
    iso_list = [50, 100, 200, 400, 800, 1600, 3200]
    # iso_list = [800, 1600, 3200]
    H = 3072
    W = 4080
    func_coef_a, func_coef_b = calibrate(rawmipi_dir, iso_list, H, W)
    print(func_coef_a)
    print(func_coef_b)
    iso = np.random.uniform(0, 25600)
    # a = func_coef_a(iso)
    # b = func_coef_b(iso)



if __name__ == '__main__':
    main()



'''
800 62 760
1600 52 789
3200 26 1023
6400 17 1023
12800 36 1023
25600 19 795
'''

'''
minmax 255
800 [0.24009768 2.37833015]
1600 [0.36304558 0.09901493]
3200 [ 0.55371207 -2.68953912]
6400 [ 0.55030025 -4.42477507]
12800 [ 0.54730531 -0.85922323]
25600 [ 0.66231632 -6.76186008]
'''

'''
0-200
800 [  0.8443113  -64.06794484]
1600 [  1.12281209 -72.98387766]
3200 [   2.18687491 -134.41119059]
6400 [   2.17415444 -133.64009588]
12800 [   2.17914591 -133.70907042]
25600 [   2.14783813 -130.57532299]
'''

'''
black-white 255
800 [ 0.17097367 -0.3388865 ]
1600 [ 0.27114284 -1.58209553]
3200 [ 0.55972923 -2.56154752]
6400 [ 0.57306416 -4.06530662]
12800 [ 0.55816298 -2.95455981]
25600 [ 0.52790812 -3.36879686]
'''