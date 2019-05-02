# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:35:54 2019

@author: macfa
"""
import numpy as np
import os
import csv

video_in_path = '../../download/separated_data/video'
csv_out_path = '../../download/separated_data/video_dict.csv'
def build_video_dict(video_in_path, csv_out_path):
    fnames = ['address', 'count']
    with open(csv_out_path, 'w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames = fnames)
        
        count = 0
        path_dict = dict()
        for (dirpath, dirnames, filenames) in os.walk(video_in_path):
            dirpath = dirpath.replace('\\','/')
            if not dirnames:
                if 'video' in dirpath:
                    for filename in filenames:
                        path = dirpath + '/' + filename
                        video, time = filename.split('_')[0], filename.split('_')[1].strip('.mp4')
                        path_dict[path] = (count, video, time)
                        count += 1
                        writer.writerow({'address': path,'count': count})
                        print(filename)
                        
                        
                        
build_video_dict(video_in_path, csv_out_path)