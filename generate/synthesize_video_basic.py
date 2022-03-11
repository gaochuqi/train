import math
import csv
import os.path

import cv2
import numpy as np
import cv2 as cv

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

gridCV111_map_x = None
gridCV111_map_y = None
gridCV1080P_map_x=None
gridCV1080P_map_y=None
cropGridNum_x = 35
cropGridNum_y = 27
height = 1080
width = 1920

def genGyroInfo():
    gyro_x=[]
    gyro_y=[]
    gyro_z=[]
    gyro_info_ =[]
    # only gyro in pitch
    for frame_num in range(1,31):
        for gyro_num in range(1,16):
                #only gyro in R_X matrix
                gyro_x.append(math.sin((2.0* math.pi / 15.0) * gyro_num))
                gyro_y.append(math.sin((2.0* math.pi / 15.0) * gyro_num) * 0)
                gyro_z.append(math.sin((2.0* math.pi / 15.0) * gyro_num) * 0)
    for frame_num in range(1, 31):
        for gyro_num in range(1, 16):
                # only gyro in R_y matrix
                gyro_x.append(math.sin((2* math.pi / 15.0) * gyro_num) * 0)
                gyro_y.append(math.sin((2* math.pi / 15.0) * gyro_num))
                gyro_z.append(math.sin((2* math.pi / 15.0) * gyro_num) * 0)
    for frame_num in range(1, 31):
        for gyro_num in range(1, 16):
                # only gyro in R_z matrix
                gyro_x.append(math.sin((2* math.pi / 15.0) * gyro_num) * 0)
                gyro_y.append(math.sin((2* math.pi / 15.0) * gyro_num) * 0)
                gyro_z.append(math.sin((2* math.pi / 15.0) * gyro_num))
    gyro_info_.append(gyro_x)
    gyro_info_.append(gyro_y)
    gyro_info_.append(gyro_z)
    gyro_info = np.array(list(zip(gyro_x, gyro_y, gyro_z)))
    with open('FakeGyro.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in gyro_info_:
            writer.writerow(row)
    return gyro_info


def genMeshGrid():
    global gridCV111_map_x, gridCV111_map_y, gridCV1080P_map_x, gridCV1080P_map_y
    #pre def, and prepare map one for all
    grid_x = np.arange(0, 1919.000001, 1919/(cropGridNum_x-1))
    grid_y = np.arange(0, 1079.000001, 1079/(cropGridNum_y-1))
    # grid_x = np.arange(-959.51, 959.51, 1919/(cropGridNum_x-1))#959.51
    # grid_y = np.arange(-539.51, 539.51, 1079/(cropGridNum_y-1))# 539.51
    X, Y = np.meshgrid(grid_x, grid_y)
    x, y = X.flatten(), Y.flatten()
    grid = list(zip(x, y))
    grid = np.array(grid)
    grid = grid.reshape(cropGridNum_y, cropGridNum_x, 2)
    if gridCV1080P_map_x is None or gridCV1080P_map_y is None:
        gridCV1080P_map_x, gridCV1080P_map_y = np.meshgrid(np.linspace(0, cropGridNum_x - 1, width).astype(np.float32),
                                                       # width
                                                       np.linspace(0, cropGridNum_y - 1, height).astype(np.float32))
    return grid

def GytoMatrix(gyroRot,deltatime):
        mRotion_x = gyroRot[0] * deltatime
        mRotion_y = gyroRot[1] * deltatime
        mRotion_z = gyroRot[2] * deltatime
        rotMatrix=([[1,0,0],[0,math.cos(mRotion_x),math.sin(mRotion_x)],[0,-1.0*math.sin(mRotion_x),math.cos(mRotion_x)]])
        rotMatriy=([[math.cos(mRotion_y),0,-1.0*math.sin(mRotion_y)],[0,1,0],[math.sin(mRotion_y),0,math.cos(mRotion_y)]])
        rotMatriz=([[math.cos(mRotion_z),math.sin(mRotion_z),0],[-1.0*math.sin(mRotion_z),math.cos(mRotion_z),0],[0,0,1]])

        rotMat = np.dot(rotMatriy,rotMatriz)
        rotMat = np.dot(rotMatrix,rotMat)
        return rotMat

def test():
    gyro_info= genGyroInfo()
    print('GenGyroInfro. IS OK!')
    # gen Grid
    grid=genMeshGrid()
    deltatime=2.378e-4
    frame_path = 'synthesisVideo/0001_5.png'
    frame = cv.imread(frame_path, cv2.IMREAD_COLOR)
    print(gyro_info.shape[0])
    # M = cv2.getRotationMatrix2D(center, angle, scale)
    for i in range(7 ,30*15*3, 14):
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
        gridCV1080P_map = cv.remap(grid.astype(np.float32), gridCV1080P_map_x, gridCV1080P_map_y,cv.INTER_LINEAR)
        warppedFrame = cv.remap(frame, gridCV1080P_map[:, :, 0], gridCV1080P_map[:, :, 1], cv.INTER_CUBIC)  # INTER_CUBIC INTER_LINEAR
        # warppedFrame = cv.remap(frame, grid[:, :, 0], grid[:, :, 1], cv.INTER_CUBIC)
        fname = os.path.basename(frame_path).split('.')[0]
        cv.imwrite('synthesisVideo/out/{}_{}.png'.format(fname, i), warppedFrame)
        # cv.imshow('Gyro Show', warppedFrame)
        # cv.waitKey(30)

def main():
    test()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
