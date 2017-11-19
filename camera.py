import cv2
import numpy as np
import sys 
import os
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

#Read/write opencv calibration matrix
#https://stackoverflow.com/questions/44056880/how-to-read-write-a-matrix-from-a-persistent-xml-yaml-file-in-opencv-3-with-pyth

#Camera calibration
#http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

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