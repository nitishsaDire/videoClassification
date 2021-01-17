from torch.utils.data import DataLoader
from Data.VideoDataset import *
import torch

def main():
    video_root_path ='/home/nitish/Downloads/ffmpeg/'
    extensions = [".avi"]
    destination_path = '/home/nitish/Downloads/ffmpeg/ALL_FRAMES/'
    videoDataset = VideoDataset(video_root_path, destination_path, extensions)

    dataloader = DataLoader(videoDataset, batch_size=4,
                            shuffle=True, num_workers=0)
    print(dataloader.batch_size)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    print(type(images))
    print(images.shape)
    print(labels.shape)

if __name__ == '__main__':
    main()
