import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('/data/akwvpAiLFk0_144.680000-150.000000.mp4') #Users/tongyu/Downloads/Documents/2nd_semester/BMEN4000DL/project/deepface_master/

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
while(currentFrame<75):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
