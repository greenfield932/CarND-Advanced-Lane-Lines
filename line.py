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

#Class that represents a lane line and it's parameters
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

        #x values for detected line pixels
        self.allx = None

        #y values for detected line pixels
        self.ally = None

        self.framesAveraging = 5 #set to 5 for project vide,  20 for challenge_video  and 2 for harder_challenge video
        self.lastSuccess = 0
        
        self.size = None

    def hasConfidentData(self):
        if self.needReset() == False and self.bestx!=None and len(self.bestx) > 0:
            return True
        return False
 
    def reset(self):
        self.detected = False  
        self.recent_xfitted = deque()
        self.recent_fitted = deque()
        self.bestx = None
        self.best_fit = None  
        self.current_fit = None
        self.radius_of_curvature = None 
        self.line_base_pos = None 
        self.allx = None
        self.ally = None
        self.lastSuccess = 0

    def hasFit(self):
        if self.current_fit == None or len(self.current_fit) == 0:
            return False
        return True
    
    def getX(self, y):
        return self.current_fit[0]*y**2 + self.current_fit[1]*y + self.current_fit[2]

    def calcConfidence2(self, fit):
        if self.ally == None:
            return True
        if fit == None or len(fit) == 0:
            return False
        new_fitx = fit[0]*self.ally**2 + fit[1]*self.ally + fit[2]
        new_fitx_corr = new_fitx.copy()
        
        # remove all values that outside of analysis window
        # we do not care what happens with the line outside of the analysis window
        new_fitx_corr[(new_fitx < 0) | (new_fitx >= self.size[0])|(self.allx < 0) | (self.allx >= self.size[0])] = 0
        curr_fit_x_corr = self.allx.copy()
        curr_fit_x_corr[(new_fitx < 0) | (new_fitx >= self.size[0])|(self.allx < 0) | (self.allx >= self.size[0])] = 0

        diff = np.abs(curr_fit_x_corr - new_fitx_corr).astype(np.uint32)

        mean_diff = np.mean(diff)
        
        max_diff = np.amax(diff)
        #print('Diff:'+str(diff))
        #print('Mean:'+str(mean_diff))
        #print('Max:'+str(max_diff))
        if  max_diff > 120:
            print('Confidence fail: maxdiff '+str(max_diff)) 
            return False
        return True

    def calcConfidence(self, fit):
        if len(fit) == 0:
            return False
        
        if self.current_fit == None:
            return True

        a0 = np.fabs(self.current_fit[0])
        a1 = np.fabs(fit[0])
        a_confidence = np.maximum(a0, a1)/np.minimum(a0, a1)
        print('a conf: '+str(a_confidence))
        if a_confidence > 10:
            return False
        
        b0 = np.abs(self.current_fit[1])
        b1 = np.abs(fit[1])
        
        b_confidence = np.abs(b1 - b0)/np.maximum(b0, b1)*100
        print('b conf: '+str(b_confidence))
        
        if b_confidence > 100:
            return False

        c0 = np.abs(self.current_fit[2])
        c1 = np.abs(fit[2])

        c_confidence = np.abs(c1 - c0)/np.maximum(c0, c1)*100
        print('c conf: '+str(c_confidence))
        
        if c_confidence > 100:
            return False

        return True

    def setFrameBroken(self):
        self.lastSuccess+=1
        
    def needReset(self):
        if self.lastSuccess >= self.framesAveraging:
            return True
        return False

    def addFrameData(self, fit_x, fit, ploty):
        self.detected = True

        self.lastSuccess = 0

        if len(self.recent_xfitted) == self.framesAveraging:
            self.recent_xfitted.pop()
        self.recent_xfitted.appendleft(fit_x)

        if len(self.recent_fitted) == self.framesAveraging:
            self.recent_fitted.pop()
        self.recent_fitted.appendleft(fit)

        self.ally = ploty
        self.allx = fit_x

        self.bestx = np.average(self.recent_xfitted, axis=0)

        self.best_fit = np.average(self.recent_fitted, axis=0)
        
        self.current_fit = fit

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
