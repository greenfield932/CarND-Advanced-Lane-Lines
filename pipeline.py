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
from test import *

def usage():
    print("Usage: pipeline.py video.mp4") 

def pipeline(img_orig, cam, path, debug = False):

    img_orig_undist = cam.undistort(img_orig)

    img = img_orig_undist
    imshape = img.shape
    xsize = imshape[1]
    ysize = imshape[0]
    
    left_bottom = (130, ysize-50)
    left_top = (550, 470)
    right_top = (760, 470)
    right_bottom = (1180, ysize-50)

    #left_bottom = (140, ysize-50)
    #left_top = (350, 570)
    #right_top = (900, 570)
    #right_bottom = (1080, ysize-50)

    #basler
    #left_bottom = (80, ysize-400)
    #left_top = (500, 330)
    #right_top = (640,330)
    #right_bottom = (1300, ysize-400)

    region_lines = [[(left_bottom[0], left_bottom[1], left_top[0], left_top[1])],
                        [(left_top[0], left_top[1], right_top[0], right_top[1])],
                        [(right_top[0], right_top[1], right_bottom[0], right_bottom[1])],
                        [(right_bottom[0], right_bottom[1], left_bottom[0], left_bottom[1])]]

    if debug == True:
        img2 = img_orig.copy()
        draw_lines_orig(img2, region_lines, [255,0,0], 3)
        showScaled('ROI', img2, 0.5)

    src  = np.float32([
        [left_bottom[0], left_bottom[1]],
        [left_top[0], left_top[1]],
        [right_top[0], right_top[1]],
        [right_bottom[0], right_bottom[1]]
    ])

    dst  = np.float32([
        [0, ysize-1],
        [0,0],
        [xsize-1, 0],
        [xsize-1, ysize-1]
    ])
    w = right_bottom[0] - left_bottom[0]
    h = left_bottom[1] - left_top[1]
    img = colorPipeline(img, debug = debug)
    
    warped, M, Minv = warp(img, src, dst, (xsize,ysize), debug = True)

    if path.size == None:
        path.setSize((warped.shape[1], warped.shape[0]))
        
    if path.hasConfidentData() and canDetectRawPixelsWithProposal(warped, path.left_line.best_fit, path.right_line.best_fit, debug):
        left_fitx, right_fitx, left_fit, right_fit, ploty = detectRawPixelsWithProposal(warped, path.left_line.best_fit, path.right_line.best_fit, debug)
    else:
        left_lane_inds, right_lane_inds = slidingWindowsFindRawPixelsIndexes(warped, debug = debug)
        left_fitx, right_fitx, ploty, left_fit, right_fit = fitCurves(warped, left_lane_inds, right_lane_inds,  debug = debug)
        
    path.addFrameData(left_fitx, right_fitx, left_fit, right_fit, ploty)

    result = drawTargetLane(warped, img_orig_undist, path.left_line.bestx, path.right_line.bestx, ploty, Minv)
    if path.hasConfidentData():
        path.drawInfo(result)

    if path.needReset():
        path.reset()

    return result

cam = Camera()
if cam.calibrationFileExists() == False or cam.load() == False:
    if cam.calibrate() == False:
        print("Fail to calibrate camera")
        sys.exit(1)
        
    cam.save()
path= Path()

videoMode = True
debug = False

#if debug == True:
#    output_calibration(cam)

#frameStart = 1048
frameStart = 0
oneFrame = True
if videoMode == False:
    #img_orig = cv2.imread('test_images/straight_lines1.jpg')
    #img_orig = cv2.imread('test_images/test6.jpg')
    img_orig  = getFrame(sys.argv[1], 547)
    img = pipeline(img_orig, cam, path, debug)
    showAndExit(img)
else:

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    videoFileName = sys.argv[1]
    cap = cv2.VideoCapture(videoFileName)

    if cap.isOpened() == False:
        print("Error opening video file:" + videoFileName)
        sys.exit(2)

    frameCnt = 0

    while(cap.isOpened() and frameCnt < frameStart):
        ret, frame = cap.read()
        frameCnt+=1

    while(cap.isOpened()):
        if cv2.waitKey(25) == 27:
            break

        ret, frame = cap.read()
        #frame = cv2.flip(frame, 0)
        if ret == True: 
            frame = pipeline(frame, cam, path, debug)
            #cv2.putText(frame, 'Frame: '+str(frameCnt), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            #cv2.imshow('Frame',frame)
            showScaled('Output',frame)
        else: 
            break
        frameCnt+=1

        if oneFrame == True:
            if cv2.waitKey(0) == ord('a'):
                continue
            elif cv2.waitKey(0) == 27:
                break
        

    cap.release() 
    cv2.destroyAllWindows()
