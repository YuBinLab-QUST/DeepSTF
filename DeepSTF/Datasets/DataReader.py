import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from Initialization.Embedding import one_hot


class SampleReader:

    def __init__(self, file_name):
        self.seq_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + file_name + '/Sequence/'
        self.shape_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/' + file_name + '/Shape/'

    def get_seq(self, Test=False):

        if Test is False:
            row_seq = pd.read_csv(self.seq_path + 'Train_seq.csv', sep=' ', header=None)
        else:
            row_seq = pd.read_csv(self.seq_path + 'Test_seq.csv', sep=' ', header=None)

        seq_num = row_seq.shape[0]
        seq_len = len(row_seq.loc[0, 1])
        completed_seqs = np.empty(shape=(seq_num, seq_len, 4))
        completed_labels = np.empty(shape=(seq_num, 1))
        for i in range(seq_num):
            completed_seqs[i] = one_hot(row_seq.loc[i, 1])
            completed_labels[i] = row_seq.loc[i, 2]
        completed_seqs = np.transpose(completed_seqs, [0, 2, 1])

        return completed_seqs, completed_labels

    def get_shape(self, shapes, Test=False):

        shape_series = []

        if Test is False:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Train' + '_' + shape + '.csv'))
        else:
            for shape in shapes:
                shape_series.append(pd.read_csv(self.shape_path + 'Test' + '_' + shape + '.csv'))

        completed_shape = np.empty(shape=(shape_series[0].shape[0], len(shapes), shape_series[0].shape[1]))

        for i in range(len(shapes)):
            shape_samples = shape_series[i]
            for m in range(shape_samples.shape[1]):
                completed_shape[m][i] = shape_samples.loc[m]
        completed_shape = np.nan_to_num(completed_shape)

        return completed_shape


class Datasets(Dataset):

    def __init__(self, file_name, Test=False):
        shapes = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']

        sample_reader = SampleReader(file_name=file_name)

        self.completed_seqs, self.completed_labels = sample_reader.get_seq(Test=Test)
        self.completed_shape = sample_reader.get_shape(shapes=shapes, Test=Test)

    def __getitem__(self, item):
        return self.completed_seqs[item], self.completed_shape[item], self.completed_labels[item]

    def __len__(self):
        return self.completed_seqs.shape[0]



