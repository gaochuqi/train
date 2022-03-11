import os
import glob
from pathlib import Path
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import cv2

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
        subfolder = 'ISO{}ex*/camera/'.format(str(iso))
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

def statisitc(rawmipi_dir, iso_list, H, W):
    coef_a = []
    coef_b = []
    for iso in iso_list:
        sub_dir = rawmipi_dir + '*_scene*/scene*/ISO{}/*_noisy*.tiff'.format(str(iso))
        RAW_list = glob.glob(sub_dir)
        RAW_list.sort()
        raw_list = []
        luminous_dict = {}
        for i, raw_path in enumerate(RAW_list):
            if i > 51:
                break
            if i>0 and i%10==0:
                raws = np.stack(raw_list, axis=0) # 10, H, W
                raws = np.maximum(raws - 240, 0.)
                mean_raw = np.mean(raws, axis=0)
                mean_raw = np.round(mean_raw)
                intensity = np.unique(mean_raw)

                thresh = {1600: 3000, 3200: 70, 6400: 130, 12800: 200, 25600: 200}
                for luminous in intensity:
                    if luminous > 1500:
                        continue
                    indices = np.expand_dims(mean_raw == luminous, 0).repeat(10, axis=0)
                    tmp = raws[indices].tolist() # x whose Ex are same
                    variance_raw = np.var(tmp)
                    if luminous_dict.get(luminous, None) is None:
                        luminous_dict[luminous] = [(variance_raw, len(tmp))]
                    else:
                        luminous_dict[luminous].extend([(variance_raw, len(tmp))])
                
                raw_list = []

            raw = cv2.imread(raw_path, -1).astype(np.float32)
            raw_list.append(raw)

        x = []
        y = []
        for k,v in luminous_dict.items():
            x.append(k)
            var_total = 0
            num_total = 0
            for var, num in v:
                var_total += var * num
                num_total += num
            y.append(np.mean(np.array(var_total / num_total)))

        coef = np.polyfit(x, y, 1)
        print(iso,coef)
        coef_a.append(coef[0])
        coef_b.append(coef[1])

        plt.figure()
        plt.scatter(x, y)
        p = np.poly1d(coef)
        xp = np.linspace(0, 2**12, 1000)
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
    x_s = np.arange(0, 30000)
    plt.plot(x_s, func_coef_a(x_s), color="green")
    plt.subplot(1, 2, 2)
    plt.scatter(iso_list, coef_b, color="red")
    plt.title("coeff b")
    plt.xlabel("iso")
    plt.ylabel("b")
    x_s = np.arange(0, 30000)
    plt.plot(x_s, func_coef_b(x_s), color="green")
    plt.suptitle("coeff")
    plt.savefig('coeff_crvd.png')
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

def main():
    setup_seed(666)
    # rawmipi_dir = '20220112/coef/'
    rawmipi_dir = '/home/pb/yushu/CRVD_dataset/indoor_raw_noisy/'
    iso_list = [1600, 3200, 6400, 12800, 25600]
    H = 1080
    W = 1920
    func_coef_a, func_coef_b = calibrate(rawmipi_dir, iso_list, H, W)
    iso = np.random.uniform(0, 25600)
    a = func_coef_a(iso)
    b = func_coef_b(iso)



if __name__ == '__main__':
    main()

# CRVD
# a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
# b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
