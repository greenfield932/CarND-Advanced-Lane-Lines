import cv2
import numpy as np
import sys 
from moviepy.editor import VideoFileClip

def usage():
    print("Usage: pipeline.py video.mp4") 
 
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
        cv2.imshow('Frame',frame)
    else: 
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows()
