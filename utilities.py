import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from camera import Camera
from skimage import exposure

#debug function to get one frame from video file
def getFrame(filename, frameStart):
    cap = cv2.VideoCapture(filename)
    frameCnt = 0

    while(cap.isOpened() and frameCnt < frameStart):
        ret, frame = cap.read()
        frameCnt+=1

    return frame

#debug function to draw separated channels for color space
def drawColorSpace(img, names, spaceFromTo = None):
    if spaceFromTo == None:
        channels = img.astype(np.float)
    else:
        channels = cv2.cvtColor(img, spaceFromTo).astype(np.float)

    channel0 = channels[:,:,0]
    channel1 = channels[:,:,1]
    channel2 = channels[:,:,2]
    
    color0 = np.dstack(( np.zeros_like(channel0), np.zeros_like(channel0), channel0))/255
    color1 = np.dstack(( np.zeros_like(channel0), np.zeros_like(channel0), channel1))/255
    color2 = np.dstack(( np.zeros_like(channel0), np.zeros_like(channel0), channel2))/255
    
    showScaled(names[0], color0, 0.5)
    showScaled(names[1], color1, 0.5)
    showScaled(names[2], color2, 0.5)

#Image color normalization with opencv/gamma correction
def refineImage(img):
    #original code https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img

#Keep on color image only yellow/white pixels with predefined tresholds, returns binary mask
def colorTreshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([50,80,150])
    upper_yellow = np.array([255,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([200,200,200])
    upper_white = np.array([255,255,255])
    mask_white = cv2.inRange(img, lower_white, upper_white)

    img = cv2.bitwise_and(img, img, mask = mask_yellow | mask_white)
    #showScaled('color trash', img, 0.5)

    mask_res = np.zeros_like(mask_white)
    mask = mask_yellow | mask_white
    mask[(mask == 255)] = 1

    return mask

#adaptive gamma correction
def equalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.
    img = exposure.equalize_adapthist(img)
    return img

#main color pipeline
def colorPipeline(img, s_thresh=(50, 250), sx_thresh=(5, 200), debug = False):
    #img = np.copy(img)

    imgEqualized = refineImage(img)
    colorMask = colorTreshold(imgEqualized)
    # Convert to HLS color space and separate the V channel
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    combined_binary[(colorMask == 1) & (sxbinary == 1)] = 1
    
    #trashMask = colorTreshold(img)
    #combined_binary[(sxbinary == 1) | (trashMask == 1)] = 255

    if debug == True:
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, colorMask)) * 255
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))* 255

        showScaled('Color pipe line #0', img, 0.5)
        showScaled('Color pipe line #1', color_binary, 0.5)
        showScaled('Color pipe line #2', combined_binary*255, 0.5)

    return combined_binary

#bird eye view transformation and back transfromation
def warp(img, src, dst, size, debug = False):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, size)
    #if debug == True:
    #    showScaled('Bird eye view', warped, 0.5)
    return warped, M, Minv

#debug helper, draw named window with adjusted scale
def showScaled(name, img, scale = None, save = False):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    if scale!=None:
        imshape = img.shape
        xsize = imshape[1]
        ysize = imshape[0]
        cv2.resizeWindow(name, int(xsize*scale), int(ysize*scale))
    if save == True:
        res = img
        if scale!=None:
            res = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite('examples/'+name+'.jpg', res)

#debug helper, draw named window with any size
def show(name, img, w = None, h = None):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    if w!=None and h!=None:
        cv2.resizeWindow(name, w, h)

#debug helper, draw image and exit on escape press
def showAndExit(img):
    cv2.imshow('frame', img)

    #Exit on esc
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    sys.exit(0)

#debug helper, draw lines on image
def draw_lines_orig(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
