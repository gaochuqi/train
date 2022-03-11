import os
import torch
import random
import numpy as np
import cv2
from arch import structure

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sRGB_ft():
    image_path = './generate/synthesisVideo/out/Mi11Ultra/20220112/IMG_20220112_194244.jpg'
    name = os.path.basename(image_path).split('.')[0]
    output_dir = os.path.dirname(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ft = structure.FreTransfer()
    input = torch.from_numpy(np.expand_dims(np.transpose(image,(2, 0, 1)), axis=0).astype(np.float32)) # .cuda()
    feat = ft(input)
    b, c, h, w = feat.shape
    feat = feat.data.cpu().numpy()
    for i in range(c):
        f = feat[0, i, :, :]
        v_max = np.max(f)
        v_min = np.min(f)
        tmp = (f - v_min) / (v_max - v_min)
        cv2.imwrite(output_dir + '/{}_{}.png'.format(name, str(i)), np.uint8(tmp * 255))

def main():
    setup_seed(666)

    sRGB_ft()

    print('end')

if __name__ == '__main__':
    main()