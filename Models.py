import torch
import torch.nn as nn
import torchvision.models as models


class decoderLSTM(nn.Module):
    def __init__(self):
        super(decoderLSTM, self).__init__()
        self.input_dim = 512
        self.hidden_size = 512
        self.hidden_size2 = 256
        self.num_layers = 3
        self.vocab_sz = 11
        self.dropout_p = 0
        self.lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, batch_first=True)
        self.sequential = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size2, momentum=0.01),
            nn.Linear(self.hidden_size2, 128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(128, self.vocab_sz)
            # nn.ReLU(),
            # nn.Softmax()
        )

    def forward(self, x):
        """

        :param x: shape (bs,ts,512)
        :return: shape (bs,vocab_sz)
        """
        res, (h, c) = self.lstm(x, None)
        return self.sequential(res[:, -1, :])


class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        self.resnet_head = list(resnet.children())[:-1]
        self.resnet_seq = nn.Sequential(*self.resnet_head)
        self.flatten = nn.Flatten(1, -1)
        self.dropout_p = 0
        self.ConvoLSTM = self.get_model()

    def get_model(self):
        ConvoLSTM = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, momentum=0.01),
            nn.ReLU(),
            nn.Linear(1024, 800),
            nn.BatchNorm1d(800, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(800, 512)
            # nn.BatchNorm1d(256),
            # nn.ReLU()
        )
        return ConvoLSTM

    def forward(self, x):
        """
        :param x: shape (bs, 30, 3, 220, 220)   torch.Size([4, 30, 3, 220, 220])
        :return: shape (bs, 30, 512)
        """
        all_embeddings = []
        for ts in range(x.shape[1]):
            with torch.no_grad():
                x_n = self.resnet_seq(x[:, ts, :, :, :])

            # print(x_n.shape) #torch.Size([4, 2048, 1, 1])
            x_n1 = self.flatten(x_n)
            # print(x_n1.shape) ##torch.Size([4, 2048])
            f = self.ConvoLSTM(x_n1)
            # print(f.shape) #torch.Size([4, 512])
            all_embeddings.append(f)
        # print(len(all_embeddings), all_embeddings[0].shape) #30 torch.Size([4, 512])
        return torch.stack(all_embeddings, 1)

    def getTrainableParameters(self):
        parameters = []
        for c in list(self.ConvoLSTM.children()):
            parameters += list(c.parameters())
        return parameters