import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from tools import setup_seed

# 一张 A4 纸，在 300 PPI 下的像素尺寸为 3508px * 2480px
def grayscale_vertical_ramps():
    h = 3508
    w = 2480
    vertical = np.asarray(range(h), dtype=np.float32) / h
    vertical = np.reshape(vertical, (-1, 1))
    gray_map = np.tile(vertical,(1,w))
    im = Image.fromarray(np.uint8((gray_map * 255)))
    im.save('./grayscale_vertical_ramps.png')

def gray_level_card():
    h = 3072
    w = 2172
    vertical = np.asarray(range(32), dtype=np.float32) / 31
    vertical = np.reshape(vertical, (-1, 1))
    gray_map = np.tile(vertical, (1, 96))
    gray_map = np.reshape(gray_map, (-1, 1))
    gray_map = np.tile(gray_map, (1, w))
    im = Image.fromarray(np.uint8((gray_map * 255)))
    im.save('./gray_level_card_32.png')

def statisitc(rawmipi_dir, iso_list, H, W):
    coef_a = []
    coef_b = []
    coef = 1.0
    black_level = 64.0
    white_level = 2 ** 10 - 1
    for iso in iso_list:
        subfolder = 'ISO{}*/camera/raw/'.format(str(iso))
        sub_dir = rawmipi_dir + subfolder + '*.raw'
        RAW_list = glob.glob(sub_dir)
        RAW_list.sort()
        raw_list = []
        luminous_dict = {}
        lux_max = 1023
        for i, raw_path in enumerate(RAW_list):
            # if i < 10 or i >=90:
            #     continue
            if i > 0 and i % 10 == 0:
                raws = np.stack(raw_list, axis=0)
                raws = np.maximum(raws - black_level, 0)
                mean_raw = np.mean(raws, axis=0)
                mean_raw = np.round(mean_raw, 0)
                var_raw = np.var(raws, axis=0)
                intensity = np.unique(mean_raw)
                thresh = {50: 50, 100: 70, 200: 130, 400: 200, 800: 200, 1600: 200, 3200: 200}
                for luminous in intensity:
                    # if luminous > lux_max or luminous < lux_min:
                    #     continue
                    if luminous > 500: # 200: # thresh[iso]:
                        continue
                    tmp = var_raw[mean_raw == luminous].tolist()
                    if luminous not in luminous_dict.keys():
                        luminous_dict[luminous] = tmp
                    else:
                        luminous_dict[luminous].extend(tmp)
                raw_list = []
            raw = np.fromfile(raw_path, dtype=np.uint16)
            raw = np.reshape(raw, (H, W))
            print(i, np.min(raw), np.max(raw))
            win = 0  # 1500
            if win:
                crop = raw[H // 2 - win:H // 2 + win, W // 2 - win:W // 2 + win]
                print(i, np.min(crop), np.max(crop))
                raw_list.append(crop)
            else:
                raw_list.append(raw)
        x = []
        y = []
        for k,v in luminous_dict.items():
            # x.append(k)
            # y.append(np.mean(np.array(v)))
            x.extend([k]*len(v))
            y.extend(v)
        coef = np.polyfit(x, y, 1)
        print(iso,coef)
        coef_a.append(coef[0])
        coef_b.append(coef[1])
    return iso_list, coef_a, coef_b

# 2022 01 24 # y1=750,y2=2170,x1=1300,x2=2850
def statisitc_gray_level(rawmipi_dir, iso_list, H=3072, W=4080, y1=800,y2=2200,x1=750,x2=3070):
    coef_a = []
    coef_b = []

    for iso in iso_list:
        subfolder = 'iso{}*/camera/raw/'.format(str(iso))
        sub_dir = rawmipi_dir + subfolder + '*.raw'
        RAW_list = glob.glob(sub_dir)
        RAW_list.sort()
        raw_list = []
        luminous_dict = {}
        lux_max = 400   # 1023
        dict_z = {}
        dict_var = {}
        dict_luminous = {}
        for i, raw_path in enumerate(RAW_list):
            raw = np.fromfile(raw_path, dtype=np.uint16)
            raw = np.reshape(raw, (H, W))
            # plt.imshow(raw)
            # plt.show()
            crop = raw[y1:y2, x1:x2] #  / 1024
            # plt.imshow(crop)
            # plt.show()
            raw_list.append(crop)

        raws = np.stack(raw_list, axis=0)
        # plt.subplot(1, 2, 1)
        # plt.hist(np.reshape(raws[11], [-1]), bins=1025, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # plt.subplot(1, 2, 2)
        # plt.hist(np.reshape(raws[37], [-1]), bins=1025, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # plt.show()
        print(iso, np.min(raws), np.max(raws))
        mean_raw = np.mean(raws, axis=0)
        # mean_raw = np.round(mean_raw, 1)
        # std_raw = np.std(raws, axis=0)
        # var_raw = np.var(raws)
        intensity = np.unique(mean_raw)
        delta = 0.01 * 1024
        v_min = 0.05 * 1024
        v_max = 0.5 * 1024
        for v in intensity:
            if v < v_min or v > v_max:
                continue
            v1 = v - delta / 2
            v2 = v + delta / 2
            tmp = crop[(crop >= v1) & (crop < v2)].tolist()
            if len(tmp) == 0:
                continue
            M = len(tmp)
            if M <= 1:
                continue
            z = np.mean(tmp)
            var = np.var(tmp) * M / (M - 1)
            if v not in dict_z.keys():
                dict_z[v] = [z]
            else:
                dict_z[v].append(z)
            if v not in dict_var.keys():
                dict_var[v] = [var]
            else:
                dict_var[v].append(var)

        x = []
        y = []
        for k, v in dict_var.items():
            # if t < np.min(dict_z[k]) or t > np.max(dict_z[k]):
            #     continue
            x.append(k)
            y.append(np.mean(np.array(v)))
            # y.append(np.sqrt(np.mean(np.array(v))))

        plt.scatter(x, y, color="red")
        plt.title("iso {}".format(iso))
        plt.xlabel("intensity")
        plt.ylabel("var")
        plt.show()
        coef = np.polyfit(x, y, 1)
        print(iso,coef)
        coef_a.append(coef[0])
        coef_b.append(coef[1])
    return iso_list, coef_a, coef_b
# 25600 [3.87632098e-07 8.11202177e-06]
# 25600 [3.96935268e-04 8.50607134e+00]

def statisitc_gray_level_v2(rawmipi_dir, iso_list, H=3072, W=4080, y1=800,y2=2200,x1=750,x2=3070):
    coef_a = []
    coef_b = []
    black_level = 64 # 240
    white_level = 2 ** 10 - 1 # 2 ** 12 - 1
    for iso in iso_list:
        subfolder = 'iso{}*/camera/raw/'.format(str(iso))
        sub_dir = rawmipi_dir + subfolder + '*.raw'
        RAW_list = glob.glob(sub_dir)
        RAW_list.sort()
        raw_list = []
        luminous_dict = {}
        dict_z = {}
        dict_var = {}
        dict_luminous = {}
        coef = 1.0
        for i, raw_path in enumerate(RAW_list):
            raw = np.fromfile(raw_path, dtype=np.uint16)
            raw = np.reshape(raw, (H, W)).astype(np.float32)
            # print(iso, np.min(raw), np.max(raw))
            crop = raw[y1:y2, x1:x2] # (raw[y1:y2, x1:x2] - black_level) / white_level
            # print(iso, np.min(crop), np.max(crop))
            # h, w = crop.shape
            # out = np.stack((crop[1:h:2, 0:w:2]*2,          # r
            #                 (crop[1:h:2, 1:w:2]+crop[0:h:2, 0:w:2])/2, # g
            #                 crop[0:h:2, 1:w:2]*2),          # b
            #                 axis=-1)
            # im = Image.fromarray(np.uint8((out * 255)))
            # im.save('iso_{}_{}_again.png'.format(iso,i))
            # plt.hist(np.reshape(crop*white_level, [-1]), bins=white_level+1, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
            # plt.show()

            # if iso>3200:
            #     coef = iso // 3200
            #     im = Image.fromarray(np.uint8((out * 255 * coef)))
            #     im.save('iso_{}_{}_dgain.png'.format(iso, i))
            #     # plt.hist(np.reshape(crop*white_level * coef, [-1]), bins=white_level+1, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
            #     # plt.show()
            raw_list.append(crop * coef)
        M = i + 1
        raws = np.stack(raw_list, axis=0)
        print(iso, np.min(raws), np.max(raws))
        mean_raw = np.mean(raws, axis=0)
        # mean_raw = np.round(mean_raw, 0)
        # std_raw = np.std(raws, axis=0)
        # var_raw = np.var(raws, axis=0) * M / (M - 1)
        intensity = np.unique(mean_raw)
        delta = 0.01
        lux_max = 0.5 * 1024
        lux_min = 0.05 * 1024
        x = []
        y = []
        thresh = {50: 50, 100: 70, 200: 130, 400: 200, 800: 200, 1600: 200, 3200: 200}
        for luminous in intensity:
            # if luminous > lux_max or luminous < lux_min:
            #     continue
            if luminous > thresh[iso]:
                continue
            tmp = raws[:, mean_raw == luminous]
            var = np.var(tmp)
            x.append(luminous)
            y.append(var)
            # tmp = var_raw[mean_raw == luminous].tolist()
            # if luminous not in luminous_dict.keys():
            #     luminous_dict[luminous] = tmp
            # else:
            #     luminous_dict[luminous].extend(tmp)


        # for k, v in luminous_dict.items():
        #     # x.append(k)
        #     # y.append(np.mean(np.array(v)))
        #     x.extend([k] * len(v))
        #     y.extend(v)
        plt.scatter(x, y, color="red")
        plt.title("iso {}".format(iso))
        plt.xlabel("intensity")
        plt.ylabel("var")
        # plt.show()
        plt.savefig("iso_{}.png".format(iso))
        plt.clf()
        coef = np.polyfit(x, y, 1)
        print(iso,coef)
        coef_a.append(coef[0])
        coef_b.append(coef[1])
    return iso_list, coef_a, coef_b

def statisitc_crvd(raw_dir, iso_list, H, W):
    coef_a = []
    coef_b = []
    for iso in iso_list:
        luminous_dict = {}
        for sceneID in range(1,12):
            print('iso', iso,'scene', sceneID)
            for frameID in range(1, 8):
                sub_dir = raw_dir+'indoor_raw_noisy_scene{}/scene{}/ISO{}/frame{}_noisy*.tiff'.format(sceneID, sceneID, iso,
                                                                                           frameID)
                tiff_list = glob.glob(sub_dir)
                tiff_list.sort()
                raw_list = []
                lux_max = 4096
                for i, tiff_path in enumerate(tiff_list):
                    raw = cv2.imread(tiff_path, -1)
                    # raw = np.reshape(raw, (H, W))
                    if i == 0:
                        print(i, np.min(raw), np.max(raw))
                    win = 0  # 1500
                    if win:
                        crop = raw[H // 2 - win:H // 2 + win, W // 2 - win:W // 2 + win]
                        print(i, np.min(crop), np.max(crop))
                        raw_list.append(crop)
                    else:
                        raw_list.append(raw)
                raws = np.stack(raw_list, axis=0)
                mean_raw = np.mean(raws, axis=0)
                mean_raw = np.round(mean_raw)
                intensity = np.unique(mean_raw)
                test = 2
                if test == 1: # one position pixel's var
                    std_raw = np.std(raws, axis=0)
                    variance_raw = np.square(std_raw)
                    for luminous in intensity:
                        if luminous > lux_max:
                            continue
                        tmp = variance_raw[mean_raw == luminous].tolist()
                        if luminous not in luminous_dict.keys():
                            luminous_dict[luminous] = tmp
                        else:
                            luminous_dict[luminous].extend(tmp)
                elif test == 2: # one picture all same intensity's pixel var
                    for luminous in intensity:
                        if luminous > lux_max:
                            continue
                        tmp = raws[:, mean_raw == luminous].tolist()
                        std_raw = np.std(tmp)
                        variance_raw = np.square(std_raw)
                        if luminous not in luminous_dict.keys():
                            luminous_dict[luminous] = [variance_raw]
                        else:
                            luminous_dict[luminous].extend([variance_raw])
        test2 = 2
        x = []
        y = []
        for k,v in luminous_dict.items():
            if test2 == 1:
                x.extend([k]*len(v))
                y.extend(v)
            elif test2 == 2:
                x.append(k)
                y.append(np.mean(np.array(v)))
        coef = np.polyfit(x, y, 1)
        print(iso,coef)
        coef_a.append(coef[0])
        coef_b.append(coef[1])
    return iso_list, coef_a, coef_b
# CRVD
# a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
# b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
'''
test=1, test2 = 1
1600 [   2.81965671 -475.50681749]
3200 [   5.19996061 -633.96022182]
6400 [   8.95098164 -109.89971701]
12800 [ 17.43442509 462.78344544]
25600 [  32.96763544 1510.70513879]
'''
'''
test=2, test2 = 1
1600 [   2.88953938 -288.60679216]
3200 [  5.29395564 -92.77512638]
6400 [   9.48109963 1018.59782909]
12800 [  16.69211744 4791.44107346]
25600 [   29.30306493 13754.75096274]
'''
'''
test=1, test2 = 2
1600 [   2.84178282 -200.56841662]
3200 [ 5.20963546 75.58766777]
6400 [   9.31550741 1406.6990377 ]
12800 [  16.34253741 5604.9412134 ]
25600 [   28.05641273 16114.91004349]
'''
'''
test=1, test2 = 2
1600 [   2.84231662 -199.88124507]
3200 [ 5.2048853 83.1825518]
6400 [   9.29648906 1434.03133504]
12800 [  16.27205916 5721.31663128]
25600 [   27.95792687 16288.11417636]
'''
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
    iso_list, coef_a, coef_b = statisitc(rawmipi_dir, iso_list, H, W)
    func_coef_a, func_coef_b = fit_coef_ab(iso_list, coef_a, coef_b)
    return func_coef_a, func_coef_b




def main():
    setup_seed(666)

    # grayscale_vertical_ramps()
    # gray_level_card()
    # return
    H = 3072
    W = 4080
    ##########################
    # CRVD
    ##########################
    # ddir = '/media/wen/09C1B27DA5EB573A/work/dataset/CRVD_dataset/indoor_raw_noisy/'
    # iso_list = [1600, 3200, 6400, 12800, 25600]
    # iso_list, coef_a, coef_b = statisitc_crvd(ddir, iso_list, H, W)
    # print(iso_list, coef_a, coef_b)
    # return
    ##########################
    # K1
    ##########################
    data_dir = '/media/wen/09C1B27DA5EB573A/work/dataset/Mi11Ultra/'
    sub_dir = 'shuang/20220112/20220112/coef/'
    # data_dir = '/media/wen/C14D581BDA18EBFA/work/dataset/MiUltra11/'
    # sub_dir = '10600/2022_01_24/2022_01_24/'  #  '10600/2022_01_25/2022_01_25/' #
    # sub_dir = 'shuang/20211230/coeff_ab/a/'
    iso_list = [50, 100, 200, 400, 800, 1600, 3200] # [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600] # [800, 1600, 3200, 6400, 12800, 25600] #
    # iso_list.sort(reverse=True)
    # convert RAWMIPI to uint16 raw
    rawmipi_dir = data_dir + sub_dir
    # rawmipi2uint16(rawmipi_dir, iso_list, H, W)
    # return
    # calibrate coefficient a,b
    iso_list, coef_a, coef_b = statisitc(rawmipi_dir, iso_list, H, W)
    # iso_list, coef_a, coef_b = statisitc_gray_level(rawmipi_dir, iso_list, H, W)
    # iso_list, coef_a, coef_b = statisitc_gray_level_v2(rawmipi_dir, iso_list, H, W)
    print(iso_list)
    print(coef_a)
    print(coef_b)
    # fit curve
    # statisitc: luminous > 200
    # iso_list = [50, 100, 200, 400, 800, 1600, 3200]
    # coef_a = [0.032574964274619124, 0.06128260177053632, 0.11700357590810931, 0.22102404328673808, 0.44366200305081527,
    #  0.8888903439910418, 1.819424625772047]
    # coef_b = [0.10442936993663798, 0.22823438196677287, -0.005996423156320616, 0.3031689746188349, 0.8404323762094531,
    #  2.868500849191923, 8.99710876206047]
    func_coef_a, func_coef_b = fit_coef_ab(iso_list, coef_a, coef_b)
    # get iso's coeff
    iso = np.random.uniform(0, 3200)
    a = func_coef_a(iso)
    b = func_coef_b(iso)
    # '''
    distribution_differences(data_dir, H, W)

if __name__ == '__main__':
    main()

# CRVD
# a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
# b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
''' shuang/20220112/20220112/coef/        statisitc  
[50, 100, 200, 400, 800, 1600, 3200]
[0.028191077741679633, 0.05564613434968623, 0.11239810147762536, 0.22102404328673808, 0.44366200305081527, 0.8888903439910418, 1.819424625772047]
[0.24373232359499103, 0.416079477095509, 0.16865630195612238, 0.3031689746188349, 0.8404323762094531, 2.868500849191923, 8.99710876206047]
'''

# mask = np.where((crop>=v) & (crop<(v+delta)), 1, 0).astype(np.uint8)
# output = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)  # 计算连同域
# num_labels = output[0]
# labels = output[1]
# colors = []
# # 随机生成RGB三颜色
# for i in range(num_labels):
#     b = np.random.randint(0, 256)
#     g = np.random.randint(0, 256)
#     r = np.random.randint(0, 256)
#     colors.append((b, g, r))
# colors[0] = (0, 0, 0)  # 大背景设置成黑色
# h = y2 - y1
# w = x2 - x1
# image = np.zeros((h, w, 3), dtype=np.uint8)  # 创建三通道空图
# for row in range(h):
#     for col in range(w):
#         image[row, col] = colors[labels[row, col]]  # 遍历像素，上色
# plt.imshow(image)
# plt.show()

# unique = np.unique(crop)
# crop_flat = np.reshape(crop,[-1])
# hist, bin_edges = np.histogram(crop_flat)
# plt.subplot(1, 2, 1)
# plt.imshow(crop)
# plt.subplot(1, 2, 2)
# plt.hist(crop_flat, bins=500, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.show()
# v_min = np.min(crop)
# v_max = np.max(crop)
# delta = 0.01
# ones = np.ones_like(crop, dtype=np.float32)
# zeros = np.zeros_like(crop, dtype=np.float32)

'''  2022_01_24 statisitc_gray_level_v2
[50, 100, 200, 400, 800, 1600, 3200]
[0.06231414992519422, 0.08388088315142221, 0.22799784438831155, 0.39775362404991, 0.6841039581417823, 1.133421670656372, 2.0269986627477152]
[-4.038521073291739, -4.708632821135834, -16.018903826071266, -29.560390189772008, -47.032510266068165, -75.20162184925323, -119.40478692507223]
'''
''' shuang/20220112 statisitc
[50, 100, 200, 400, 800, 1600, 3200]
[0.008301621831566883, 0.012232987794999914, 0.01897553040386134, 0.026815828637268613, 0.04951847675832405, 0.09320560848193736, 0.19296701929339524]
[5.020177574588825, 9.297401628983224, 17.065627015159304, 31.831289977080548, 63.908340772328636, 129.48238303547149, 269.0831211355436]
'''
'''
50 [ 0.05235911 -1.71684983]
100 [ 0.08541394 -1.94745782]
200 [ 0.14464327 -2.48958254]
400 [ 0.23760278 -1.16184428]
800 [ 0.46382461 -0.89972503]
1600 [0.91756416 0.3864313 ]
3200 [1.91583949 0.13987658]

'''