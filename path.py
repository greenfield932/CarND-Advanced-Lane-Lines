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
from line import Line

class Path():
    def __init__(self):
        self.left_line = Line()
        self.right_line = Line()
        self.size = None
       
    def needReset(self):
        if self.left_line.needReset() or self.right_line.needReset():
            print('Path need reset: True')
            return True
        print('Path need reset: False')
        return False
    
    def reset(self):
        print('Path reset')
            
        self.left_line.reset()
        self.right_line.reset()
        
    def calcCurvature(self):
        # Define conversions in x and y from pixels space to meters
        left_curve, right_curve = calcCurvature(self.left_line.bestx, self.left_line.best_fit, self.right_line.bestx, self.right_line.best_fit, self.left_line.ally)
        return np.mean([left_curve, right_curve])
    
    def calcCenterShift(self):
        left_startx = self.left_line.bestx[-1]
        right_startx = self.right_line.bestx[-1]
        midpoint = left_startx + (right_startx - left_startx)/2
        width = self.size[0]
        diff_pix = width/2 - midpoint
        xm_per_pix = 3.7/700
        diff_meters = diff_pix*xm_per_pix
        return diff_meters
    
    def drawInfo(self, frame):
        start_y = 40
        step_y = 40
        start_x = 20
        curvrad = self.calcCurvature()
        cv2.putText(frame, 'Radius of curvature: ' + str(int(curvrad))+' m', (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        start_y += step_y
        shift = self.calcCenterShift()
        cv2.putText(frame, 'Shift from center: {:.2} m'.format(shift), (start_x,start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        
        
    def hasConfidentData(self):
        res = self.left_line.hasConfidentData() and self.right_line.hasConfidentData()
        print('Path confident data:'+str(res))
        return res
    
    def addFrameData(self, left_fitx, right_fitx, left_fit, right_fit, ploty):

        left_conf = self.left_line.calcConfidence2(left_fit)
        right_conf = self.right_line.calcConfidence2(right_fit)
        #print('Left confidence:'+str(left_conf))
        #print('Right confidence:'+str(right_conf))
        if left_conf and right_conf:
            path_confidence = self.checkPathWidth(left_fit, right_fit)
            if path_confidence == True:
                self.left_line.addFrameData(left_fitx, left_fit, ploty)
                self.right_line.addFrameData(right_fitx, right_fit, ploty)
                print('Frame ok')
                return True
        print('Frame bad')
        self.left_line.setFrameBroken()
        self.right_line.setFrameBroken()
       
        return False
        
    def setSize(self, size):
        self.size = size
        self.left_line.size = size
        self.right_line.size = size
        #print('SetSize:'+str(size))
        
    def checkPathWidth(self, left_fit, right_fit):
        #if self.left_line.hasPoints() == False or self.left_line.hasFit() == False:
        #    return False
        #if self.right_line.hasPoints() == False  or self.right_line.hasFit() == False:
        #    return False
        if left_fit == None or len(left_fit) == 0 or right_fit == None or len(right_fit) == 0:
            return False
        
        ploty = np.linspace(0, self.size[1]-1, self.size[1])
    
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        startWidth = np.abs(left_fitx[self.size[1]-1] -right_fitx[self.size[1]-1])
        
        
        # remove all values that outside of analysis window
        left_fitx[(left_fitx < 0) | (left_fitx >= self.size[0])|(right_fitx < 0) | (right_fitx >= self.size[0])] = 0
        right_fitx[(left_fitx < 0) | (left_fitx >= self.size[0])|(right_fitx < 0) | (right_fitx >= self.size[0])] = startWidth
        
        allWidths = np.abs(left_fitx - right_fitx).astype(np.uint32)
        minWidth = np.amin(allWidths)
        maxWidth = np.amax(allWidths)

        #print('Start width:' + str(startWidth))
        #print('Min width:' + str(minWidth))
        #print('Max width:' + str(maxWidth))
        
        if minWidth < startWidth/3:
            #print('Min width confidence fail')
            print('Confidence fail: minWidth<startWidth/3 '+str(minWidth)+' '+str(startWidth/3) )
        
            return False
        
        if maxWidth > startWidth*1.2:
            #print('Min width confidence fail')
            print('Confidence fail: maxWidth>startWidth/3 '+str(maxWidth)+' '+str(startWidth*1.2) )
        
            return False
        
        if startWidth < 700:
            print('Confidence fail: startWidth<700'+str(startWidth)) 
        
            return False
        #if maxWidth > startWidth/3:
        #    print('Max width confidence fail')
        #    return False
        #print('Confidence ok')
        return True