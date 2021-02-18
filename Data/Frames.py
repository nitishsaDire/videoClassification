import cv2
import os

class Frames:
    def __init__(self, video_path, destination_path):
        self.video_path = video_path
        self.destination_path = destination_path + self.video_path.split("/")[-1].split(".")[0]
        self.minimumNumberOfFrames = 30
        if os.path.isdir(self.destination_path) == False:
            os.mkdir(self.destination_path)
        self.isHavingMinimumNumberOfFrames = True

    def get_frames_path(self):
        return self.destination_path+"/"

    def doesSatisfiesMinimumFramesConstraint(self):
        return self.isHavingMinimumNumberOfFrames

    def generate_frames(self):
        video = cv2.VideoCapture(self.video_path)
        success_flag, frame = video.read()
        numberOfFrames = self.minimumNumberOfFrames + 1
        if len(os.listdir(self.destination_path)) == 0:
            numberOfFrames = 0
            while success_flag and numberOfFrames < self.minimumNumberOfFrames:
                cv2.imwrite(self.destination_path + '/frame{:06d}.jpg'.format(numberOfFrames), frame)
                success_flag, frame = video.read()
                numberOfFrames += 1
        if numberOfFrames < self.minimumNumberOfFrames:
            self.isHavingMinimumNumberOfFrames = False
            os.system("rm -r " + self.destination_path)

