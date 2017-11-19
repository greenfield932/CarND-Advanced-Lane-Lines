import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from camera import Camera
from utilities import *
from sliding_windows import *
from line import Line
from path import Path

#test distortion
def output_calibration(cam):
    filename = 'camera_cal/calibration1.jpg'
    img = cv2.imread(filename)
    #cv2.imwrite('examples/calibration_1.jpg', img)
    showScaled('original', img, 0.5)
    
    img = cam.undistort(img)
    showScaled('undistorted', img, 0.5)
    #cv2.imwrite('examples/calibration_2.jpg', img)

#test warp transformation
def warptest(cam):
    img = cv2.imread('camera_cal/calibration8.jpg')     
    img = cam.undistort(img)
    #cv2.imwrite('examples/undistorted_example1.jpg', result)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    nx = 9
    ny = 6
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    src  = np.float32([
        corners[0][0],
        corners[nx-1][0],
        corners[nx*ny-1][0],
        corners[nx*ny-nx][0],    
    ])
    for i in range(0,4):
        if i==2:
            cv2.circle(img, (src[i][0],src[i][1]), 5, (255,0,0), 3)
    imshape = img.shape
    xsize = imshape[1]
    ysize = imshape[0]

    dst  = np.float32([
        [5,5],
        [xsize-5,5],
        [xsize-5, ysize-5],
        [5, ysize-5]
    ])
    for i in range(0,4):
        cv2.circle(img, (dst[i][0],dst[i][1]), 5, (0,255,0), 3)
    print(src)
    print(dst)
    print(xsize)
    print(ysize)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (xsize,ysize))                
    cv2.imshow('frame', warped)