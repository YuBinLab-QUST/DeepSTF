import torch.nn as nn
import torch


class dlbss(nn.Module):

    def __init__(self, shape_num):
        super(dlbss, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels=shape_num, out_channels=4, kernel_size=1, stride=1, padding=0)

        self.deep_shared_cnn = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=13, stride=1, padding=6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(p=0.2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),

            nn.AdaptiveMaxPool1d(output_size=1)
        )

        self.neural_network = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def _forward_impl(self, seq, shape):
        seq = seq.float()
        shape = shape.float()

        shape = self.conv1x1(shape)

        shape = self.deep_shared_cnn(shape)
        seq = self.deep_shared_cnn(seq)

        fuse = torch.cat((seq, shape), dim=1)

        output = self.neural_network(fuse.flatten(start_dim=1))

        return output

    def forward(self, seq, shape):
        return self._forward_impl(seq, shape)


Seq = torch.ones(size=(32, 4, 101))
Shape = torch.ones(size=(32, 5, 101))
Net = dlbss(shape_num=5)
Net(Seq, Shape)
