# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:57:00 2019

@author: macfa
"""
from ffmpy import FFmpeg
import subprocess
csv_out_path = '../../download/data/test_dict.csv'
def process_and_save(dict_path):
    with open(dict_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            video_in_path = row[0]
            video_out_path = '../../download/separated_data/' + '/video/' + row[2] + '-' + row[3] + '.mp4'
            #video_out_path = row[3] + '.mp4'
            wav_out_path = '../../download/separated_data/' + '/audio/' + row[2]  + '-' + row[3] + '.wav'
            try:
                os.makedirs('../../download/separated_data/' + 'video')
            except FileExistsError:
                pass
            try:
                os.makedirs('../../download/separated_data/' + 'audio')
            except FileExistsError:
                pass
                    
                         
                         
                         
                         
                         
            #wav_out_path = row[3] + '.wav'
            #separate_mp4(video_in_path, video_out_path, wav_out_path)
            
            command = 'ffmpeg -i {0} -map 0:0 -c:a copy -f mp4 {1} -map 0:1 -c:a copy -f mp4 {2}'.format(video_in_path, video_out_path, wav_out_path)
            subprocess.call(command, shell=True)
            
process_and_save(csv_out_path)