import torch.nn as nn
import torch


class crpts(nn.Module):

    def __init__(self, shape_num):
        super(crpts, self).__init__()
        self.channels_matching = nn.Sequential(
            nn.Conv1d(in_channels=shape_num, out_channels=4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.feature_extraction = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=13, stride=1, padding=6),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self.feature_extraction_lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.feature_extraction_dropout = nn.Dropout(p=0.2)

        self.feature_integration = nn.Sequential(
            nn.BatchNorm1d(num_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def _forward_impl(self, seq, shape):
        seq = seq.float()
        shape = shape.float()

        shape = self.channels_matching(shape)

        shape = self.feature_extraction(shape)
        shape, (_, _) = self.feature_extraction_lstm(shape.permute(0, 2, 1))
        shape = self.feature_extraction_dropout(shape)

        seq = self.feature_extraction(seq)
        seq, (_, _) = self.feature_extraction_lstm(seq.permute(0, 2, 1))
        seq = self.feature_extraction_dropout(seq)

        concat = torch.cat((shape.squeeze(1), seq.squeeze(1)), dim=1)
        Y = self.feature_integration(concat)
        print(Y.shape)

    def forward(self, seq, shape):
        return self._forward_impl(seq, shape)