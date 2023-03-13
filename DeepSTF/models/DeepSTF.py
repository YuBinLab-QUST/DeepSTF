import torch
import models.Transformer as TF
import torch.nn as nn


class deepstf(nn.Module):

    def __init__(self):
        super(deepstf, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.convolution_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.transformer_shape=TF.Transformer(101,8,8,128,128,0.1)
        self.lstm = nn.LSTM(42,21,6, bidirectional=True, batch_first=True, dropout=0.2)
        self.convolution_seq_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1))
        )
        self.convolution_shape_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1))
        )
        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def execute(self, seq, shape):
        shape = shape.float()
        shape=shape.squeeze(1)
        encoder_shape_output= self.transformer_shape(shape)
        encoder_shape_output = encoder_shape_output.unsqueeze(1)
        conv_shape_1 = self.convolution_shape_1(encoder_shape_output)
        pool_shape_1 = self.max_pooling_1(conv_shape_1)
        pool_shape_1 = pool_shape_1.squeeze(2)
        out_shape, _ = self.lstm(pool_shape_1.to(self.device))
        out_shape1 = out_shape.unsqueeze(2)
        conv_shape_2=self.convolution_shape_2(out_shape1)
        seq = seq.float()
        conv_seq_1 = self.convolution_seq_1(seq.to(self.device)).to(self.device)
        pool_seq_1= self.max_pooling_1(conv_seq_1)
        conv_seq_2=self.convolution_seq_2(pool_seq_1)
        return self.output(torch.cat((conv_shape_2, conv_seq_2), dim=1))

    def forward(self, seq, shape):
        return self.execute(seq, shape)