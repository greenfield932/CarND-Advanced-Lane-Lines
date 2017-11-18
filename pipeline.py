import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from camera import Camera
from utilities import *
from sliding_windows import *
from collections import deque
#Here links for codes used to create this project

#Camera calibration
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

#Read/write opencv calibration matrix
#https://stackoverflow.com/questions/44056880/how-to-read-write-a-matrix-from-a-persistent-xml-yaml-file-in-opencv-3-with-pyth

def usage():
    print("Usage: pipeline.py video.mp4") 
    
class Path():
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()

    def addFrameData(self, left_fitx, right_fitx, left_fit, right_fit, ploty):
        self.left_line.addFrameData(left_fitx, left_fit, ploty)
        self.right_line.addFrameData(right_fitx, right_fit, ploty)
    def proposeStart(self):
        if self.left_line.proposeStart() == None or self.right_line.proposeStart() == None:
            return None
        
        return (self.left_line.proposeStart(), self.right_line.proposeStart())
        
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  

        # x values of the last n fits of the line
        self.recent_xfitted = deque()
        self.recent_fitted = deque()

        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  

        #polynomial coefficients for the most recent fit
        self.current_fit = None

        #radius of curvature of the line in some units
        self.radius_of_curvature = None 

        #distance in meters of vehicle center from the line
        self.line_base_pos = None 

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

        self.framesAveraging = 5
        self.lastSuccess = 0

    def proposeStart(self):
        if self.allx != None:
            return self.allx[-1]
        return None

    def processData(self):
        #TODO averaging
        return True

    def addFrameData(self, fit_x, fit, ploty):
        self.detected = self.sanityCheck(fit_x, fit)

        if self.detected == True:
            self.lastSuccess = 0

            if len(self.recent_xfitted) == self.framesAveraging:
                self.recent_xfitted.pop()
            self.recent_xfitted.appendleft(fit_x)
            
            if len(self.recent_fitted) == self.framesAveraging:
                self.recent_fitted.pop()
            self.recent_fitted.appendleft(fit)
            
            self.ally = ploty
            self.allx = fit_x

            if self.current_fit!=None:
                self.diffs = self.current_fit - fit

            #self.bestx = np.mean(self.recent_xfitted)
            self.bestx = fit_x
            
            self.best_fit = np.mean(self.recent_fitted)

            self.current_fit = fit
        else:
            self.lastSuccess += 1

    def sanityCheck(self, fit_x, fit):
        if len(fit_x) == 0 or len(fit) == 0:
            return False
        return True

    def drawInfo(self, img):
        start_y = 40
        step_y = 20
        start_x = 20
        cv2.putText(frame, 'Success frames ago: '+str(self.lastSuccess), (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        start_y+=step_y
        #cv2.putText(frame, 'Fit left: '+str(self.current_fit[0][0]) + ' ' + str(self.current_fit[0][1]) +' ' + str(self.current_fit[0][2]), (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        
def pipeline(img_orig, cam, path, debug = False):

    img_orig_undist = cam.undistort(img_orig)
    #img = colorPipeline(img_orig_undist, debug = debug)
    #img = refineImage(img_orig)
    #img = colorTrash(img)
    img = img_orig_undist
    #showScaled('yellow', img)
    #return img_orig
    #return img_orig_undist
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
        showScaled('ROI', img2)

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
    warped, M, Minv = warp(img, src, dst, (xsize,ysize), debug = False)
    #img = cv2.resize(warped, (int(w/2), int(h/2)), cv2.INTER_CUBIC)
    warped = colorPipeline(warped, debug = debug)
    #return img
    #res = windows(warped, img_orig_undist, Minv)
    left_lane_inds, right_lane_inds = slidingWindowsFindRawPixelsIndexes(warped, path.proposeStart(), debug = debug)
    left_fitx, right_fitx, ploty, left_fit, right_fit = fitCurves(warped, left_lane_inds, right_lane_inds,  debug = debug)
    path.addFrameData(left_fitx, right_fitx, left_fit, right_fit, ploty)
    #if line.detected == True:
    #    result = drawTargetLane(warped, img_orig_undist, left_fitx, right_fitx, ploty, Minv)
    #else:
    #    result = drawTargetLane(warped, img_orig_undist, line.best_left_x, line.best_right_x, ploty, Minv)
    result = drawTargetLane(warped, img_orig_undist, path.left_line.bestx, path.right_line.bestx, ploty, Minv)

    return result
    return img_orig_undist

cam = Camera()
if cam.calibrationFileExists() == False or cam.load() == False:
    if cam.calibrate() == False:
        print("Fail to calibrate camera")
        sys.exit(1)
        
    cam.save()
path= Path()

videoMode = True
debug = True
#frameStart = 1048
frameStart = 0
oneFrame = True
if videoMode == False:
    #warptest(cam)
    #cv2.imshow('frame', img)

    #Exit on esc
    #if cv2.waitKey(0) == 27:
    #    cv2.destroyAllWindows()
    #sys.exit(0)
    #img_orig = cv2.imread('test_images/straight_lines1.jpg')     
    img_orig  = getFrame(sys.argv[1], 547)
    
    
    #img_orig = cv2.imread('test_images/test6.jpg')     
    img = pipeline(img_orig, cam, path, debug)
    showAndExit(img)
    #Exit on esc
    #if cv2.waitKey(0) == 27:
    #    cv2.destroyAllWindows()

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

        #if oneFrame == True and cv2.waitKey(25) == ord('a'):
        ret, frame = cap.read()
        #frame = cv2.flip(frame, 0)
        if ret == True: 
            frame = pipeline(frame, cam, path, debug)
            cv2.putText(frame, 'Frame: '+str(frameCnt), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('Frame',frame)
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
