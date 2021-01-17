import cv2
from glob import glob
import os

class VideoFiles:
    '''It gives all the video files names recursively in the root path along with their labels'''
    def __init__(self, video_root_path, extensions):
        self.video_root_path = video_root_path
        self.extensions = extensions

    def get_files_paths(self):
        result = []
        for path, subdirs, files in os.walk('/home/nitish/Downloads/ffmpeg'):
            for name in files:
                print(path+name)
                for e in self.extensions:
                    if name.endswith(e):
                        result.append(path + "/" + name)
        return  result
