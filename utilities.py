import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from camera import Camera

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

def refineImage(img):
    #https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #cv2.imshow('l_channel', l)
    #cv2.imshow('a_channel', a)
    #cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    #cv2.imshow('CLAHE output', cl)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
    #cv2.imshow('final', final)

def colorPipeline(img, s_thresh=(90, 230), sx_thresh=(30, 70), debug = False):
    img = np.copy(img)
    #img = refineImage(img)

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
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    if debug == True:
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        #color_binary = np.dstack(( np.zeros_like(sxbinary), np.zeros_like(sxbinary), s_channel))/255
        #color_binary1 = np.dstack(( np.zeros_like(sxbinary), np.zeros_like(sxbinary), h_channel))/255
        #color_binary2 = np.dstack(( np.zeros_like(sxbinary), np.zeros_like(sxbinary), s_channel))/255
        #color_binary3 = np.dstack(( np.zeros_like(sxbinary), np.zeros_like(sxbinary), v_channel))/255
        #showScaled('H', img, 0.5)
        #showScaled('S', img, 0.5)
        #showScaled('V', img, 0.5)
        #
        #print(color_binary)
        #showScaled('Color pipe line #0', img, 0.5)
        showScaled('Color pipe line #1', color_binary, 0.5)
        showScaled('Color pipe line #2', combined_binary, 0.5)

    return combined_binary

def warp(img, src, dst, size, debug = False):
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, size)
    if debug == True:
        showScaled('Bird eye view', warped, 0.5)
    return warped, M, Minv

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

def show(name, img, w = None, h = None):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    if w!=None and h!=None:
        
        cv2.resizeWindow(name, w, h)

def showAndExit(img):
    cv2.imshow('frame', img)

    #Exit on esc
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    sys.exit(0)
	
def draw_lines_orig(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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