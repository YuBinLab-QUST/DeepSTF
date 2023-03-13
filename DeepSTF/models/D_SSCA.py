import torch.nn as nn
import torch


class Attention(nn.Module):

    def __init__(self, channel=64, ratio=8):
        super(Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        b, c, _, _ = F.size()

        F_avg = self.shared_layer(self.avg_pool(F).reshape(b, c))
        F_max = self.shared_layer(self.max_pool(F).reshape(b, c))
        M = self.sigmoid(F_avg + F_max).reshape(b, c, 1, 1)

        return F * M


class d_ssca(nn.Module):

    def __init__(self):
        super(d_ssca, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convolution_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        self.attention_seq = Attention(channel=128, ratio=16)
        self.attention_shape = Attention(channel=128, ratio=16)

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def _forward_impl(self, seq, shape):
        seq = seq.float()
        shape = shape.float()


        conv_seq_1 = self.convolution_seq_1(seq.to(self.device)).to(self.device)
        pool_seq_1 = self.max_pooling_1(conv_seq_1)

        conv_shape_1 = self.convolution_shape_1(shape.to(self.device)).to(self.device)
        pool_shape_1 = self.max_pooling_1(conv_shape_1)

        attention_seq_1 = self.attention_seq(pool_seq_1)

        attention_shape_1 = self.attention_shape(pool_shape_1)##64 128 1 42


        return self.output(torch.cat((attention_seq_1, attention_shape_1), dim=1))

    def forward(self, seq, shape):
        return self._forward_impl(seq, shape)
