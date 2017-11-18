import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from camera import Camera
from utilities import *
from sliding_windows import *
#Here links for codes used to create this project

#Camera calibration
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

#Read/write opencv calibration matrix
#https://stackoverflow.com/questions/44056880/how-to-read-write-a-matrix-from-a-persistent-xml-yaml-file-in-opencv-3-with-pyth

def usage():
    print("Usage: pipeline.py video.mp4") 

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
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
      

def windows(binary_warped, img_orig_undist, Minv):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #showAndExit(out_img)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()
    #return
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    leftx = left_fitx
    rightx = right_fitx
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    warped = binary_warped
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_orig_undist.shape[1], img_orig_undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img_orig_undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    #plt.show()
    #showAndExit(result)
    return result


def pipeline(img_orig, cam, debug = False):

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
    left_lane_inds, right_lane_inds = slidingWindowsFindRawPixelsIndexes(warped, debug = debug)
    if len(left_lane_inds)>0 and len(right_lane_inds) > 0 :
        left_fitx, right_fitx, ploty = fitCurves(warped, left_lane_inds, right_lane_inds,  debug = debug)
        result = drawTargetLane(warped, img_orig_undist, left_fitx, right_fitx, ploty, Minv)
        return result
    return img_orig_undist

cam = Camera()
if cam.calibrationFileExists() == False or cam.load() == False:
    if cam.calibrate() == False:
        print("Fail to calibrate camera")
        sys.exit(1)
        
    cam.save()

def getFrame(filename, frameStart):
    cap = cv2.VideoCapture(filename)
    frameCnt = 0

    while(cap.isOpened() and frameCnt < frameStart):
        ret, frame = cap.read()
        frameCnt+=1

    return frame
    
videoMode = True
debug = True
#frameStart = 1048
frameStart = 700
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
    img = pipeline(img_orig, cam, debug)
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
            frame = pipeline(frame, cam, debug)
            cv2.putText(frame, 'Frame: '+str(frameCnt), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('Frame',frame)
        else: 
            break
        frameCnt+=1

        if oneFrame == True and cv2.waitKey(0) == ord('a'):
            continue

    cap.release() 
    cv2.destroyAllWindows()
