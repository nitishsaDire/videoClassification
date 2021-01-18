from torch.utils.data import DataLoader
from Data.VideoDataset import *
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import time,copy
from sklearn.model_selection import train_test_split
from torchsummary import summary

def main():
    video_root_path ='/home/nitish/Downloads/ffmpeg/'
    extensions = [".avi"]
    destination_path = '/home/nitish/Downloads/ffmpeg/ALL_FRAMES/'
    videoDataset = VideoDataset(video_root_path, destination_path, extensions)

    dataloader = DataLoader(videoDataset, batch_size=2,
                            shuffle=True, num_workers=0)
    print(dataloader.batch_size)
    # dataiter = iter(dataloader)
    # images, labels = dataiter.next()
    # print(type(images))
    # print(images.shape)
    # print(labels.shape)



    criterion = nn.CrossEntropyLoss()
    ConvoLSTM = [CNNModel(),decoderLSTM()]
    ConvoLSTM[1](torch.randn((2,10,512)))

    # ConvoLSTMParams = ConvoLSTM.parameters()
    optimizer = torch.optim.SGD(list(decoderLSTM().parameters()), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    train_model(ConvoLSTM, criterion, optimizer, exp_lr_scheduler, dataloader)


class decoderLSTM(nn.Module):
    def __init__(self):
        super(decoderLSTM, self).__init__()
        self.input_dim = 512
        self.hidden_size = 512
        self.num_layers = 3
        self.vocab_sz = 4
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.vocab_sz)
        # self.decoderLSTM = self.get_model()

    # def get_model(self):
    #     return nn.Sequential(
    #         self.lstm,
    #         ,
    #         nn.ReLU(),
    #         nn.Softmax()
    #     )

    def forward(self, x):
        """

        :param x: shape (bs,ts,512)
        :return: shape (bs,ts,4)
        """
        all_embeddings = []
        # for ts in range(x.shape[1]):
        # all_embeddings.append(self.decoderLSTM(x))
        res,h = self.lstm(x)
        self.linear(res)
        return nn.Softmax(nn.ReLU(res))

        # return all_embeddings

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet_head = list(resnet.children())[:-1]
        self.ConvoLSTM = self.get_model()
        print(list(self.ConvoLSTM.parameters()))

    def get_model(self):
        ConvoLSTM = nn.Sequential(
            nn.Sequential(*self.resnet_head),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        return ConvoLSTM


    def forward(self, x):
        """
        :param x: shape (bs, 30, 3, 220, 220)   torch.Size([4, 30, 3, 220, 220])
        :return: shape (bs, 30, 512)
        """
        all_embeddings = []
        for ts in range(x.shape[1]):
            all_embeddings.append(self.ConvoLSTM(x[:,ts,:,:,:]))
        return  all_embeddings



def train_model(model, criterion, optimizer, scheduler, dataloader, num_epochs=25):
    since = time.time()
    cnn,lstm = model
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                cnn.train()
                lstm.train() # Set model to training mode
            else:
                cnn.eval()   # Set model to evaluate mode
                lstm.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels in dataloader[phase]:
            for inputs, labels in dataloader:

                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = lstm(cnn(inputs))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    main()
