# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:00:25 2019

@author: macfa
"""

import cv2
import os
videos_folder = '../../200_download/separated_data/video'
frames_folder = '../../200_download/separated_data/frame'
#video_name = 'akwvpAiLFk0_144.680000-150.000000'
#video_name = 'AvWWVOgaMlk_090.000000-093.566667'
video_num = 100

def frame_extraction(videos_folder, frames_folder, video_num):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(videos_folder):
        f.extend(filenames)
        break
    
    f = f[:video_num-1]
    
    for i in range(len(f)):
        video_file = f[i]
        video_name = video_file[:-4]
        frame_folder = frames_folder + '/' + video_name
        

        
        video_path = videos_folder + '/' + video_file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("{0}/{1}".format(i, len(f)), f[i], fps)
        if fps < 25:
            cap.release()
            cv2.destroyAllWindows()
            continue
        try:
            os.makedirs(frame_folder)
        except FileExistsError:
            pass
        fps_target = 25
        currentFrame = 0
        targetFrame = 0
        count = 0
        while(count<75):
        # Capture frame-by-frame
            ret, frame = cap.read()
        
            #print ('Creating...' + name)
            #print(currentFrame, targetFrame)
            if (currentFrame == round(targetFrame)):
                #print(currentFrame == int(targetFrame),currentFrame, targetFrame)
                # Saves image of the current frame in jpg file
                frame_path = frame_folder + '/' + str(count) + '.jpg'
                cv2.imwrite(frame_path, frame)
                targetFrame += fps/fps_target
                
                count += 1
                
                
        
            # To stop duplicate images
            currentFrame += 1
        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        

frame_extraction(videos_folder, frames_folder, video_num)
    
    


#video_name = 'ymD5uLlLc0g_036.033000-040.900000'
#video_file = videos_folder + '/' + video_name + '.mp4'
## Playing video from file:
#cap = cv2.VideoCapture(video_file) #Users/tongyu/Downloads/Documents/2nd_semester/BMEN4000DL/project/deepface_master/
#fps = cap.get(cv2.CAP_PROP_FPS)
#print(fps)
#fps_target = 25
#try:
#    if not os.path.exists(frames_folder):
#        os.makedirs(frames_folder)
#except OSError:
#    print ('Error: Creating directory of data')
#
#currentFrame = 0
#targetFrame = 0
#count = 0
#while(currentFrame<75):
#    # Capture frame-by-frame
#    ret, frame = cap.read()
#
#    #print ('Creating...' + name)
#    if (currentFrame == int(targetFrame)):
#        # Saves image of the current frame in jpg file
#        name = frames_folder + '/' + str(count) + '.jpg'
#        cv2.imwrite(name, frame)
#        targetFrame += fps/25
#        count += 1
#        print(currentFrame)
#
#    # To stop duplicate images
#    currentFrame += 1
#
## When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()



