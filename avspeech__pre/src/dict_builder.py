# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:59:47 2019

@author: macfa
"""
import csv
import os
import sys
csv_out_path = '../../../final_data/data/test_dict.csv'
mp4_fold_path = '../../../final_data/data/test'





def build_dict(infile_path, outfile_path):
    fnames = ['address', 'count', 'video_name', 'time']
    with open(outfile_path, 'w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames = fnames)
        
        count = 0
        path_dict = dict()
        for (dirpath, dirnames, filenames) in os.walk(infile_path):
            dirpath = dirpath.replace('\\','/')
            if not dirnames:
                if 'test' in dirpath:
                    for filename in filenames:
                        path = dirpath + '/' + filename
                        video, time = filename.split('_')[0], filename.split('_')[1].strip('.mp4')
                        path_dict[path] = (count, video, time)
                        count += 1
                        row = [path, int(count), video, time]
                        writer.writerow({'address': path,'count': count, 'video_name': video,'time':time})
                        
build_dict(mp4_fold_path, csv_out_path)
    