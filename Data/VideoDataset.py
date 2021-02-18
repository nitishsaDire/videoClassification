from .VideoFiles import *
from .Frames import *
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
from sklearn import preprocessing

class VideoDataset(Dataset):
    def __init__(self, video_path, destination_path, extensions):
        self.video_path = video_path
        self.extensions = extensions
        self.destination_path = destination_path
        videoFiles = VideoFiles(self.video_path, self.extensions)
        self.videoFilesPaths = videoFiles.get_files_paths()
        self.n = len(self.videoFilesPaths)
        self.frames=[]
        self.le = preprocessing.LabelEncoder()
        self.lessFramesThanThresholdList = []

        # UCF-11 dataset
        # self.le.fit(["basketball","biking","diving","golf_swing","horse_riding","soccer_juggling","swing","tennis_swing","trampoline_jumping","volleyball_spiking","walking"])
        # UCF-50 dataset
        self.le.fit(["BaseballPitch", "Basketball", "BenchPress", "Biking", "Billiards", "BreastStroke", "CleanAndJerk", "Diving", "Drumming", "Fencing", "GolfSwing", "HighJump", "HorseRace", "HorseRiding", "HulaHoop", "JavelinThrow", "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Lunges", "MilitaryParade", "Mixing", "Nunchucks", "PizzaTossing", "PlayingGuitar", "PlayingPiano", "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps", "Punch", "PushUps", "RockClimbingIndoor", "RopeClimbing", "Rowing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling", "Swing", "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog", "YoYo"])


        for idx, video in enumerate(self.videoFilesPaths):
            print("generating frames for video #", idx, ", ", video)
            self.frames.append(Frames(video, self.destination_path))
            self.frames[idx].generate_frames()
            if self.frames[idx].doesSatisfiesMinimumFramesConstraint() == False:
                self.lessFramesThanThresholdList.append(idx)
        self.removeLessFramesThanThresholdVideos()

    def removeLessFramesThanThresholdVideos(self):
        for idx in sorted(self.lessFramesThanThresholdList, reverse=True):
            del self.frames[idx]
            del self.videoFilesPaths[idx]
        self.n = len(self.videoFilesPaths)

    def get_label(self, videoFilePath):
        '''Assuming video has label name in its name'''
        # print(str(videoFilePath), flush=True)
        # "hi"
        return self.le.transform(list(str(videoFilePath.split("/")[-2]).split(" ")))

    # ! ls - l
    # UCF11_updated_mpg / basketball / v_shooting_01 / v_shooting_01_01.mpg
    def get_images(self, images_path):
        images = []
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

        for path, subdirs, files in os.walk(images_path):
            for i in range(0,30):
                image = Image.open(path + 'frame{:06d}.jpg'.format(i))
                image = transform(image)
                images.append((image))
        return  torch.stack(images, dim=0)

    def __getitem__(self, index):
        frames_path = self.frames[index].get_frames_path()
        # print(self.videoFilesPaths[index], flush=True)
        images, label = self.get_images(frames_path), self.get_label(self.videoFilesPaths[index])
        return images,label

    def __len__(self):
        return len(self.videoFilesPaths)