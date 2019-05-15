# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:57:00 2019

@author: macfa
"""
from ffmpy import FFmpeg
import subprocess
import os
import csv
csv_out_path = '../../lf/data/test_dict.csv'
def process_and_save(dict_path, n):
    with open(dict_path, 'r') as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count < 100:
                count += 1
                continue
            video_in_path = row[0]
            name = video_in_path.split('/')[-1]
            name = name[:-4]
            video_out_path = '../../lf/separated_data/' + '/video/' + name + '.mp4'
            #video_out_path = '../../download/separated_data/' + '/video/' + row[2] + '-' + row[3] + '.mp4'
            #video_out_path = row[3] + '.mp4'
            wav_out_path = '../../lf/separated_data/' + '/audio/' + name + '.wav'
            #wav_out_path = '../../download/separated_data/' + '/audio/' + row[2]  + '-' + row[3] + '.wav'
            try:
                os.makedirs('../../lf/separated_data/' + 'video')
            except FileExistsError:
                pass
            try:
                os.makedirs('../../lf/separated_data/' + 'audio')
            except FileExistsError:
                pass
            
            #wav_out_path = row[3] + '.wav'
            #separate_mp4(video_in_path, video_out_path, wav_out_path)
            
            command = 'ffmpeg -i {0} -map 0:0 -c:a copy -f mp4 {1} -map 0:1 -c:a copy -f mp4 {2}'.format(video_in_path, video_out_path, wav_out_path)
            subprocess.call(command, shell=True)
            
            print("{0}/{1}".format(count, n))
            count += 1
            if count == n:
                break
            
process_and_save(csv_out_path, 300)