import os
import glob
import time
from pathlib import Path
import numpy as np

from PIL import Image
import cv2

from generate.tools import setup_seed

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


def rawmipi2uint16(rawmipi_dir, iso_list, H=3072,W=4080):
    for iso in iso_list:
        subfolder = 'iso{}*/camera/'.format(str(iso))
        sub_dir = rawmipi_dir + subfolder + '*.RAWMIPI10'
        RAWMIPI10_list = glob.glob(sub_dir)
        RAWMIPI10_list.sort()
        for i, img_path in enumerate(RAWMIPI10_list):
            out_dir = os.path.join(os.path.dirname(img_path),'raw')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            name = os.path.basename(img_path).split('.')[0]
            image_pixel = read_mipi10(img_path, H, W)
            # image_pixel.tofile(out_dir + '/%s.raw' % name)
            np.save(out_dir + '/{}.npy'.format(name), image_pixel)

def cvt_mipi10_to_bayer16(src_mipi, dst_raw):
    with open(src_mipi,'rb') as fin:
        with open(dst_raw,'wb') as fout:
            while True:
                b = fin.read(5)
                if not len(b):
                    break
                n = int.from_bytes(b, "little")

                data_4 = n & 0xff
                data_3 = (n >> 8) & 0xff
                data_2 = (n >> 16) & 0xff
                data_1 = (n >> 24) & 0xff
                data_0 = (n >> 32) & 0xff
                p4 = (data_0 << 2) + (data_4 & 0x03);
                p3 = (data_1 << 2) + ((data_4 & 0x0C) >> 2);
                p2 = (data_2 << 2) + ((data_4 & 0x30) >> 4);
                p1 = (data_3 << 2) + ((data_4 & 0xC0) >> 6);
                fout.write(p1.to_bytes(2, "little"))
                fout.write(p2.to_bytes(2, "little"))
                fout.write(p3.to_bytes(2, "little"))
                fout.write(p4.to_bytes(2, "little"))


def test():
    img_path = '/media/wen/C14D581BDA18EBFA1/work/dataset/Mi11Ultra/' \
               'shuang/20220112/20220112/coef/ISO50exT256000000/camera/' \
               'IMG_20220112_202022-230_req[1]_b[0]_BPS[0][0]_w[4080]_h[3072]_sw[0]_sh[0]_ZSLSnapshotYUVHAL.RAWMIPI10'
    out_dir = 'synthesisVideo/out/'
    H = 3072
    W = 4080
    image_pixel = read_mipi10(img_path, H, W)
    image_pixel = image_pixel.astype(np.uint16)

    name = 'save_raw'
    image_pixel.tofile(out_dir + '{}.raw'.format(name))

    noisy_save = Image.fromarray(np.uint16(image_pixel))
    name = 'save_tiff'
    noisy_save.save(out_dir + '{}.tiff'.format(name))

    name = 'save_npy'
    np.save(out_dir + '{}.npy'.format(name), image_pixel)

    st = time.time()
    name = 'save_tiff'
    a = cv2.imread(out_dir + '{}.tiff'.format(name), -1)
    print(a.shape)
    print('cv2 load tiff time,      ', (time.time() - st) * 1e6)

    st = time.time()
    name = 'save_raw'
    a = np.fromfile(out_dir + '{}.raw'.format(name), dtype=np.uint16)
    a = np.reshape(a, (H,W))
    print(a.shape)
    print('np load raw time,      ', (time.time() - st) * 1e6)

    st = time.time()
    name = 'save_npy'
    a = np.load(out_dir + '{}.npy'.format(name), mmap_mode='r')
    print(a.shape)
    print('np load npy time,      ', (time.time() - st) * 1e6)

def tmp():
    bTIFF = False
    if bTIFF:
        tiff_pattern = '/home/wen/Documents/dataset/DRV/trainval/*/noisy/iso*/*.tiff'
        tiff_list = glob.glob(tiff_pattern)
        tiff_list.sort()
        for tiff_path in tiff_list:
            a = cv2.imread(tiff_path, -1)
            out_path = tiff_path.split('.')[0]
            np.save(out_path + '.npy', a)
    bRAW = True
    if bRAW:
        raw_pattern = '/home/wen/Documents/dataset/DRV/trainval/*/raw_bin/*.raw'
        raw_list = glob.glob(raw_pattern)
        raw_list.sort()
        H = 3672
        W = 5496
        for raw_path in raw_list:
            raw = np.fromfile(raw_path, dtype=np.uint16)
            raw = np.reshape(raw, (H//2, W//2))
            out_path = raw_path.split('.')[0]
            np.save(out_path + '.npy', raw)


def main():
    setup_seed(666)
    # test()
    tmp()
    print('end')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

'''
cv2 load tiff time,       24035.215377807617
np load raw time,       4439.830780029297
np load npy time,       5265.951156616211
'''