from .VideoFiles import *
from .Frames import *
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os
from PIL import Image
import re
import torch

class VideoDataset(Dataset):
    def __init__(self, video_path, destination_path, extensions):
        self.video_path = video_path
        self.extensions = extensions
        self.destination_path = destination_path
        videoFiles = VideoFiles(self.video_path, self.extensions)
        self.videoFilesPaths = videoFiles.get_files_paths()
        self.n = len(self.videoFilesPaths)
        self.frames=[]
        for idx, video in enumerate(self.videoFilesPaths):
            self.frames.append(Frames(video, self.destination_path))
            self.frames[-1].generate_frames()

    def get_label(self, frame_path):
        '''Assuming video has label name in its name'''
        m = re.search("v_(.*)_g", frame_path)
        return m.group(1)

    def get_images(self, images_path):
        images = []
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((220, 220))])

        for path, subdirs, files in os.walk(images_path):
            for name in files:
                image = Image.open(path + name).convert('RGB')
                image = transform(image)
                images.append((image))
        return  torch.stack(images[:30], dim=0)

    def __getitem__(self, index):
        frames_path = self.frames[index].get_frames_path()
        images, label = self.get_images(frames_path), self.get_label(frames_path)
        return images,label

    def __len__(self):
        return len(self.videoFilesPaths)


