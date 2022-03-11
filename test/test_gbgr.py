import numpy as np
from PIL import Image
import cv2
import glob
import os

def rgbg_to_rgb(image):
    image_r = image[:, :, 0:1]
    image_g1 = image[:, :, 1:2]
    image_b = image[:, :, 2:3]
    image_g2 = image[:, :, 3:4]
    image = np.concatenate((image_r, (image_g1+image_g2)/2, image_b), axis=2)
    return image


def pack_raw(raw, r_gain=1.95130301, b_gain=1.77383304):
    im = np.maximum(raw - 64, 0) / (1023 - 64)  # subtract the black level
    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H, W = img_shape[0], img_shape[1]
    # *******************************************************************************
    # GB
    # RG
    c = 2
    out = np.concatenate((im[1:H:c, 0:W:c, :] * r_gain,      # r
                          im[1:H:c, 1:W:c, :],               # g
                          im[0:H:c, 1:W:c, :] * b_gain,      # b
                          im[0:H:c, 0:W:c, :]), axis=2)      # g

    return out

def test():
    folder = '20211221193515_0'
    img_dir = '/home/wen/Documents/dataset/Mi11Ultra/'+folder
    img_paths = glob.glob(img_dir+'/*_Input.raw')
    for img_path in img_paths:
        name = os.path.basename(img_path)
        imgData = np.fromfile(img_path, dtype=np.uint16)
        imgData = imgData.reshape(3072, 4080)
        out = pack_raw(imgData, r_gain=1.95130301, b_gain=1.77383304)
        A_N = rgbg_to_rgb(out)
        im = Image.fromarray(np.uint8((A_N * 255)))
        im.save(img_dir+'/%s.png'%name)
        # im.show()

def main():
    test()

if __name__ == '__main__':
    main()



