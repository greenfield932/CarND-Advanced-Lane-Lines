import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
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
        
        
class Camera():
    def __init__(self):
        print("Camera init")
        self.calibFilename = 'calibration.xml'
        
    def calibrate(self):
        filesCount = 20
        foundCount = 0
        nx = 9
        ny = 6
        pattern_size = (nx, ny)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        imagePoints = []
        objectPoints = []
        w = 0
        h = 0

        for i in range(0,filesCount):
            filename = 'camera_cal/calibration'+str(i+1)+'.jpg'
            # Read calibration image
            img = cv2.imread(filename)     

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                h, w = img.shape[:2]
                foundCount += 1
                imagePoints.append(corners.reshape(-1, 2))
                objectPoints.append(pattern_points)
                #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        print('Susscessfully detected '+str(foundCount)+' of ' + str(filesCount) + ' calibration patterns')
        if foundCount > 0:
            rms, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, (w, h), None, None)
            print('Camera successfully calibrated')
            #print(self.mtx)
            #print(self.dist)
        
            return True
        
        
        #vis = np.concatenate((img1, img2), axis=0)
        return False
     
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    def calibrationFileExists(self):
        return os.path.isfile(self.calibFilename) 
    
    def save(self):
        print('Save calibration data to file')
        cvFile = cv2.FileStorage(self.calibFilename, cv2.FILE_STORAGE_WRITE)
        cvFile.write('mtx', self.mtx)
        cvFile.write('dist', self.dist)
        cvFile.release()
    
    def load(self):
        print('Load calibration data from file')
        cvFile = cv2.FileStorage(self.calibFilename, cv2.FILE_STORAGE_READ)
        self.mtx = cvFile.getNode('mtx').mat()
        self.dist = cvFile.getNode('dist').mat()
        cvFile.release()
        
        if len(self.mtx) != 0:
            return True
        
        return False

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

def colorPipeline(img, s_thresh=(120, 210), sx_thresh=(30, 70)):
    img = np.copy(img)
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
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    #return color_binary
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255

    return combined_binary

def showAndExit(img):
    cv2.imshow('frame', img)

    #Exit on esc
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    sys.exit(0)

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


def pipeline(img_orig, cam):

    img_orig_undist = cam.undistort(img_orig)
    img = colorPipeline(img_orig_undist)
    #showAndExit(img)

    imshape = img.shape
    xsize = imshape[1]
    ysize = imshape[0]
    left_bottom = (130, ysize-50)
    left_top = (570, 450)
    right_top = (710, 450)
    #left_top = (610, 430)
    #right_top = (670, 430)
    right_bottom = (1180, ysize-50)

    region_lines = [[(left_bottom[0], left_bottom[1], left_top[0], left_top[1])],
                        [(left_top[0], left_top[1], right_top[0], right_top[1])],
                        [(right_top[0], right_top[1], right_bottom[0], right_bottom[1])],
                        [(right_bottom[0], right_bottom[1], left_bottom[0], left_bottom[1])]]

    #draw_lines_orig(img, region_lines, [255,0,0], 3)
    #showAndExit(img)

    src  = np.float32([
        [left_bottom[0], left_bottom[1]],
        [left_top[0], left_top[1]],
        [right_top[0], right_top[1]],
        [right_bottom[0], right_bottom[1]]
    ])

    #dst  = np.float32([
    #    [left_bottom[0], left_bottom[1]],
    #    [left_bottom[0], left_top[1]],
    #    [right_bottom[0], right_top[1]],
    #    [right_bottom[0], right_bottom[1]]
    #])

    dst  = np.float32([
        [0, ysize-1],
        [0,0],
        [xsize-1, 0],
        [xsize-1, ysize-1]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    #w = right_bottom[0] - left_bottom[0]
    #h = left_bottom[1] - left_top[1]
    w = right_top[0] - left_top[0]
    h = left_bottom[1] - left_top[1]
    warped = cv2.warpPerspective(img, M, (xsize,ysize))                
    #cv2.imshow('frame', warped)
    #showAndExit(warped)
    res = windows(warped, img_orig_undist, Minv)
    return res
    #histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    #plt.show()
    #for i in range(0,4):
    #    cv2.circle(img, (dst[i][0],dst[i][1]), 2, (0,0,255), 2)
    #    cv2.circle(img, (src[i][0],src[i][1]), 5, (0,255,0), 2)
    #cv2.imshow('frame', img)
    

cam = Camera()
if cam.calibrationFileExists() == False or cam.load() == False:
    if cam.calibrate() == False:
        print("Fail to calibrate camera")
        sys.exit(1)
        
    cam.save()


#warptest(cam)
#cv2.imshow('frame', img)

#Exit on esc
#if cv2.waitKey(0) == 27:
#    cv2.destroyAllWindows()
#sys.exit(0)
#img_orig = cv2.imread('test_images/straight_lines1.jpg')     
#img_orig = cv2.imread('test_images/test6.jpg')     
#img = pipeline(img_orig, cam)
#showAndExit(img)
#Exit on esc
#if cv2.waitKey(0) == 27:
#    cv2.destroyAllWindows()


if len(sys.argv) < 2:
    usage()
    sys.exit(1)

videoFileName = sys.argv[1]
cap = cv2.VideoCapture(videoFileName)

if cap.isOpened() == False:
    print("Error opening video file:" + videoFileName)
    sys.exit(2)
print(cap.isOpened())
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True: 
        frame = pipeline(frame, cam)
        cv2.imshow('Frame',frame)
    else: 
        break

    if cv2.waitKey(25) == 27:
        break
    
cap.release() 
cv2.destroyAllWindows()
