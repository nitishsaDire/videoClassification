from torch.utils.data import DataLoader
from Data.VideoDataset import *
from Models import CNNModel, decoderLSTM
import torch
import torch.nn.functional as F
import time
import gc
import numpy as np
import matplotlib.pyplot as plt


def find_learning_rate(video_root_path = '/content/UCF11_updated_mpg/', extensions = [".mpg", ".mp4", ".avi"], destination_path = '/content/ALL_FRAMES/'):
    train_dataset = VideoDataset(video_root_path, destination_path, extensions)
    dataloader = DataLoader(train_dataset, batch_size=20,
                                shuffle=True, num_workers=4, pin_memory=True)
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    print(device)

    cnnModel = CNNModel().to(device)
    decoderlstm = decoderLSTM().to(device)
    ConvoLSTM = [cnnModel, decoderlstm]

    optimizer = torch.optim.Adam(list(decoderlstm.parameters()) + list(cnnModel.getTrainableParameters()), lr=0.001)

    find_lr_and_plot(ConvoLSTM, optimizer, dataloader, device, 100)


def find_lr_and_plot(model, optimizer, dataloader, device, num_its=100):
    logmin_lr, logmax_lr = -7, 0
    print(num_its)
    log_lrrange = np.linspace(logmin_lr, logmax_lr, num_its)
    lr_list = [10 ** i for i in log_lrrange]
    losses = get_lr(model, optimizer, dataloader, device, lr_list, num_its)
    _, ax = plt.subplots(figsize=(20, 10))

    N = 4
    mv_mean_loss = np.convolve(losses, np.ones(N) / N, mode='valid')
    mv_mean_lr = np.convolve(log_lrrange, np.ones(N) / N, mode='valid')
    ax.plot(mv_mean_lr, mv_mean_loss)
    plt.show()

def get_lr(model, optimizer, dataloader, device, lr_list, num_its=100):
    cnn, lstm = model
    it_losses = []
    current_it = 0

    for epoch in range(100):

        torch.cuda.empty_cache()
        gc.collect()

        print('Epoch {}/{}'.format(epoch, 100 - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        cnn.train()
        lstm.train()  # Set model to training mode

        # Iterate over data.
        count = 0
        it_begin = time.time()
        for inputs, labels in dataloader:
            if (current_it > num_its - 1): return it_losses

            if (count + 9) % 10 == 0:
                it_begin = time.time()
            labels = labels.squeeze(1)
            inputs, labels = inputs.to(device), labels.to(device)
            print(it_losses)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = lstm(cnn(inputs))
                loss = F.cross_entropy(outputs, labels)
                it_losses.append(loss.item())
                # backward + optimize only if in training phase
                loss.backward()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_list[current_it]
                optimizer.step()

            if count % 10 == 0:
                time_elapsed = time.time() - it_begin
                print("Iterated over ", count, "LR=", lr_list[current_it],
                      'Iteration Completed in {:.0f}m {:.0f}s'.format(
                          time_elapsed // 60, time_elapsed % 60))
            count += 1
            current_it += 1
