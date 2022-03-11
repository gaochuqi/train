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





# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def genGyroInfo():
    gyro_x = []
    gyro_y = []
    gyro_z = []
    gyro_info_ = []
    # only gyro in pitch
    for frame_num in range(1, 31):
        for gyro_num in range(1, 16):
            # only gyro in R_X matrix
            gyro_x.append(math.sin((2.0 * math.pi / 15.0) * gyro_num))
            gyro_y.append(math.sin((2.0 * math.pi / 15.0) * gyro_num) * 0)
            gyro_z.append(math.sin((2.0 * math.pi / 15.0) * gyro_num) * 0)
    for frame_num in range(1, 31):
        for gyro_num in range(1, 16):
            # only gyro in R_y matrix
            gyro_x.append(math.sin((2 * math.pi / 15.0) * gyro_num) * 0)
            gyro_y.append(math.sin((2 * math.pi / 15.0) * gyro_num))
            gyro_z.append(math.sin((2 * math.pi / 15.0) * gyro_num) * 0)
    for frame_num in range(1, 31):
        for gyro_num in range(1, 16):
            # only gyro in R_z matrix
            gyro_x.append(math.sin((2 * math.pi / 15.0) * gyro_num) * 0)
            gyro_y.append(math.sin((2 * math.pi / 15.0) * gyro_num) * 0)
            gyro_z.append(math.sin((2 * math.pi / 15.0) * gyro_num))
    gyro_info_.append(gyro_x)
    gyro_info_.append(gyro_y)
    gyro_info_.append(gyro_z)
    gyro_info = np.array(list(zip(gyro_x, gyro_y, gyro_z)))
    with open('FakeGyro.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in gyro_info_:
            writer.writerow(row)
    return gyro_info

gridCV111_map_x = None
gridCV111_map_y = None
gridCV1080P_map_x = None
gridCV1080P_map_y = None
cropGridNum_x = 35
cropGridNum_y = 27
def genMeshGrid(height = 1836,
                width = 2748,):
    global gridCV111_map_x, gridCV111_map_y, gridCV1080P_map_x, gridCV1080P_map_y
    # pre def, and prepare map one for all
    grid_x = np.arange(0, (width - 1) + 1e-6, (width - 1) / (cropGridNum_x - 1))
    grid_y = np.arange(0, height - 1 + 1e-6, (height - 1) / (cropGridNum_y - 1))
    # grid_x = np.arange(-((width - 1)/2+0.01), (width - 1)/2+0.01, (width - 1)/(cropGridNum_x-1))      # 959.51
    # grid_y = np.arange(-((height - 1)/2+0.01), (height - 1)/2+0.01, (height - 1)/(cropGridNum_y-1))     # 539.51
    X, Y = np.meshgrid(grid_x, grid_y)
    x, y = X.flatten(), Y.flatten()
    grid = list(zip(x, y))
    grid = np.array(grid)
    grid = grid.reshape(cropGridNum_y, cropGridNum_x, 2)
    if gridCV1080P_map_x is None or gridCV1080P_map_y is None:
        gridCV1080P_map_x, gridCV1080P_map_y = np.meshgrid(np.linspace(0, cropGridNum_x - 1, width).astype(np.float32),
                                                           np.linspace(0, cropGridNum_y - 1, height).astype(np.float32))
        # gridCV1080P_map_x, gridCV1080P_map_y = np.meshgrid(np.linspace(-(cropGridNum_x - 1)//2, (cropGridNum_x - 1)//2, width).astype(np.float32),
        #                                                    np.linspace(-(cropGridNum_y - 1)//2, (cropGridNum_y - 1)//2, height).astype(np.float32))
    return grid


def GytoMatrix(gyroRot, deltatime):
    mRotion_x = gyroRot[0] * deltatime
    mRotion_y = gyroRot[1] * deltatime
    mRotion_z = gyroRot[2] * deltatime
    rotMatrix = (
        [[1, 0, 0], [0, math.cos(mRotion_x), math.sin(mRotion_x)],
         [0, -1.0 * math.sin(mRotion_x), math.cos(mRotion_x)]])
    rotMatriy = (
        [[math.cos(mRotion_y), 0, -1.0 * math.sin(mRotion_y)], [0, 1, 0],
         [math.sin(mRotion_y), 0, math.cos(mRotion_y)]])
    rotMatriz = (
        [[math.cos(mRotion_z), math.sin(mRotion_z), 0], [-1.0 * math.sin(mRotion_z), math.cos(mRotion_z), 0],
         [0, 0, 1]])

    rotMat = np.dot(rotMatriy, rotMatriz)
    rotMat = np.dot(rotMatrix, rotMat)
    return rotMat

