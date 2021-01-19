from torch.utils.data import DataLoader
from Data.VideoDataset import *
import torch
import torch.nn as nn
import torchvision.models as models
import time,copy

def main():
    video_root_path ='/home/nitish/Downloads/ffmpeg/'
    extensions = [".avi"]
    destination_path = '/home/nitish/Downloads/ffmpeg/ALL_FRAMES/'
    videoDataset = VideoDataset(video_root_path, destination_path, extensions)

    train_size = int(0.8 * len(videoDataset))
    val_size = len(videoDataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(videoDataset, [train_size, val_size])

    videoDatasetDict = {"train": train_dataset, "val":val_dataset}

    dataset_sizes = {x: len(videoDatasetDict[x]) for x in ['train', 'val']}

    dataloader = {x: DataLoader(videoDatasetDict[x], batch_size=2,
                            shuffle=True, num_workers=0) for x in ['train', 'val']}
    # print(dataloader.batch_size)



    criterion = nn.CrossEntropyLoss()
    ConvoLSTM = [CNNModel(),decoderLSTM()]

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    print(device)

    optimizer = torch.optim.SGD(list(decoderLSTM().parameters()) + list(CNNModel().parameters()), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    train_model(ConvoLSTM, criterion, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device)


class decoderLSTM(nn.Module):
    def __init__(self):
        super(decoderLSTM, self).__init__()
        self.input_dim = 512
        self.hidden_size = 512
        self.num_layers = 3
        self.vocab_sz = 4
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers)
        self.sequential = nn.Sequential(
                        nn.Linear(self.hidden_size, self.vocab_sz),
                        nn.ReLU(),
                        nn.Softmax())

    def forward(self, x):
        """

        :param x: shape (bs,ts,512)
        :return: shape (bs,vocab_sz)
        """

        res,h = self.lstm(x)
        return self.sequential(res[:,-1,:])


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
        return  torch.stack(all_embeddings, 1)



def train_model(model, criterion, optimizer, scheduler, dataloader, dataset_sizes, device, num_epochs=25):
    since = time.time()
    cnn,lstm = model
    PATH = ''
    best_model_wts_cnn, best_model_wts_lstm = copy.deepcopy(cnn.state_dict()), copy.deepcopy(lstm.state_dict())
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
            for inputs, labels in dataloader[phase]:
            # for inputs, labels in dataloader:

                inputs = inputs.to(device)
                labels = labels.to(device)

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
                best_model_wts_cnn, best_model_wts_lstm = copy.deepcopy(cnn.state_dict()), copy.deepcopy(lstm.state_dict())

        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'lstm_state_dict': lstm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, PATH)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model

if __name__ == '__main__':
    main()
