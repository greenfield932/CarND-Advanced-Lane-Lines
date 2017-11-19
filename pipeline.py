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
    print("Usage: pipeline.py video.mp4 [output_video.avi]") 

#main pipeline, frames processed here
def pipeline(img_orig, cam, path, debug = False):

    #Undistort image (assume camera calibrated on start)
    img_orig_undist = cam.undistort(img_orig)

    img = img_orig_undist
    imshape = img.shape
    xsize = imshape[1]
    ysize = imshape[0]
    
    #Set region of interest for project_video
    left_bottom = (130, ysize-50)
    left_top = (550, 470)
    right_top = (760, 470)
    right_bottom = (1180, ysize-50)
    #Set region of interest for challenge and harder_challenge video
    #left_bottom = (140, ysize-50) #left_bottom = (140, ysize-50)
    #left_top = (350, 500) #left_top = (350, 570)
    #right_top = (900, 500) #right_top = (900, 570)
    #right_bottom = (1080, ysize-50) #right_bottom = (1080, ysize-50)

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

    #Set transformation coordinates for bird eye view
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
    
    #Process color image with gradient and color tresholds
    img = colorPipeline(img, debug = debug)
    
    #Transform image to bird eye view
    warped, M, Minv = warp(img, src, dst, (xsize,ysize), debug = True)

    
    if path.size == None:
        path.setSize((warped.shape[1], warped.shape[0]))

    # check if we already have previous successfully processed frames with lane lines
    if path.hasConfidentData() and canDetectRawPixelsWithProposal(warped, path.left_line.best_fit, path.right_line.best_fit, debug):
        #if so try to detect lane lines at the approximate same places as for several previous frames (coordinates are averaged)
        left_fitx, right_fitx, left_fit, right_fit, ploty = detectRawPixelsWithProposal(warped, path.left_line.best_fit, path.right_line.best_fit, debug)
    else:
        #if no previously success data or we have a too muich bad frames, than start blind search with sloding windows
        left_lane_inds, right_lane_inds = slidingWindowsFindRawPixelsIndexes(warped, debug = debug)
        left_fitx, right_fitx, ploty, left_fit, right_fit = fitCurves(warped, left_lane_inds, right_lane_inds,  debug = debug)

    # add extracted data to post processing, here we analyze how good is our lines, are they parallel and splitted with enough space, etc.
    path.addFrameData(left_fitx, right_fitx, left_fit, right_fit, ploty)

    #make back transform from bird eye view to normal view and add found path to the original undistorted image
    result = drawTargetLane(warped, img_orig_undist, path.left_line.bestx, path.right_line.bestx, ploty, Minv)

    #draw curvature and shift from the center
    if path.hasConfidentData():
        path.drawInfo(result)

    #check if we have too much bad frames, if so reset to start with sliding windows search
    if path.needReset():
        path.reset()

    return result


#START
cam = Camera()
# check if we already calibrated camera with predefined images, if so just load calibration matrix from disk, otherwise calibrate and save matrix to disk
if cam.calibrationFileExists() == False or cam.load() == False:
    if cam.calibrate() == False:
        print("Fail to calibrate camera")
        sys.exit(1)

    cam.save()


path= Path()

#Video mode or single frame mode
videoMode = True

# if set to True multiple windows with middle processing results will appear
debug = False

#if debug == True:
#    output_calibration(cam)
#frameStart = 1048

#for debug purposes, we can start from some debugging frame
frameStart = 0

#for debugging purposes, in video mode 'a' simbol is waited to move to next frame
oneFrame = False

if videoMode == False:#just load single frame and process it
    
    #img_orig = cv2.imread('test_images/straight_lines1.jpg')
    #img_orig = cv2.imread('test_images/test6.jpg')
    img_orig  = getFrame(sys.argv[1], 547)
    img = pipeline(img_orig, cam, path, debug)
    showAndExit(img)
else: #use video for processing

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    videoWriter = None

    videoFileName = sys.argv[1]
    cap = cv2.VideoCapture(videoFileName)
    
    if cap.isOpened() == False:
    
        print("Error opening video file:" + videoFileName)
        sys.exit(2)

    fps = cap.get(cv2.CAP_PROP_FPS)

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

            if videoWriter == None and len(sys.argv) >= 3:
                fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
                videoWriter = cv2.VideoWriter(sys.argv[2], fourcc, fps,(frame.shape[1], frame.shape[0]),True)

            frame = pipeline(frame, cam, path, debug)
            #cv2.putText(frame, 'Frame: '+str(frameCnt), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            #cv2.imshow('Frame',frame)
            showScaled('Output',frame)
            if videoWriter!=None:
                videoWriter.write(frame)
        else: 
            break
        frameCnt+=1

        if oneFrame == True:
            if cv2.waitKey(0) == ord('a'):
                continue
            elif cv2.waitKey(0) == 27:
                break
        
    if videoWriter!=None:
        videoWriter.release()
    cap.release() 
    cv2.destroyAllWindows()
