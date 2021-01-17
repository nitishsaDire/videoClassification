import cv2
import os

class Frames:
    def __init__(self, video_path, destination_path):
        self.video_path = video_path
        self.destination_path = destination_path + self.video_path.split("/")[-1].split(".")[0]
        os.mkdir(self.destination_path)

    def get_frames_path(self):
        return self.destination_path+"/"

    def generate_frames(self):
        video = cv2.VideoCapture(self.video_path)
        success_flag, frame = video.read()
        numberOfFrames = 1

        while success_flag:
            cv2.imwrite(self.destination_path + '/frame{:06d}.jpg'.format(numberOfFrames), frame)
            success_flag, frame = video.read()
            numberOfFrames += 1

