import torch.nn as nn
import torch


class DanQ(nn.Module):
    def __init__(self):
        super(DanQ, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=16)
        )
        self.Pool = nn.MaxPool1d(kernel_size=8, stride=8)
        self.Drop1 = nn.Dropout(p=0.2)

        self.BiLSTM = nn.LSTM(input_size=16, hidden_size=32, num_layers=2,
                              batch_first=True, dropout=0.5, bidirectional=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.FullyConnection = nn.Sequential(
            nn.Linear(576, 925),
            nn.ReLU(),
            nn.Linear(925, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()

        x = self.Conv(x)
        x = self.Pool(x)
        x = self.Drop1(x)

        x_x = torch.transpose(x, 1, 2)
        x, (hn, hc) = self.BiLSTM(x_x)

        x = self.flatten(x)

        print(x.shape)
        output = self.FullyConnection(x)

        return output