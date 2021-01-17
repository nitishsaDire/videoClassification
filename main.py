from Data.VideoDataset import *
def main():
    video_root_path ='/home/nitish/Downloads/ffmpeg/'
    extensions = [".avi"]
    destination_path = '/home/nitish/Downloads/ffmpeg/ALL_FRAMES/'
    videoDataset = VideoDataset(video_root_path, destination_path, extensions)
    for i in range(len(videoDataset)):
        x,y = videoDataset[i]
        print(len(x), x[0].shape, len(y))



if __name__ == '__main__':
    main()
