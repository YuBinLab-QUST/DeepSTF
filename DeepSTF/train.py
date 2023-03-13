import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
import torch.utils.data as loader
import math


from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Datasets.DataReader import Datasets

import torch
from torch import nn
import numpy as np


class Constructor:
    def __init__(self, model, model_name='deepstf'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        self.loss_function = nn.BCELoss()
        self.batch_size = 64
        self.epochs = 15

    def practise(self, TrainLoader, ValidateLoader):
        path = os.path.abspath(os.curdir)
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()
                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, label = data
                output= self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())
                loss.backward()
                self.optimizer.step()
            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_shape, valid_labels in ValidateLoader:
                    valid_output = self.model(valid_seq.unsqueeze(1).to(self.device), valid_shape.unsqueeze(1).to(self.device))
                    valid_labels = valid_labels.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())
                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)
        torch.save(self.model.state_dict(), path + '\\' + self.model_name + '.pth')


    def demonstration(self, TestLoader):
        path = os.path.abspath(os.curdir)
        self.model.load_state_dict(torch.load(path + '\\' + self.model_name + '.pth', map_location='cpu'))
        predict_value = []
        true_label = []
        self.model.eval()
        with torch.no_grad():
            for seq, shape, label in TestLoader:
                output= self.model(seq.unsqueeze(1), shape.unsqueeze(1))
                predict_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
                true_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
            return predict_value, true_label

    def estimate(self, predict_value, true_label):
        accuracy = accuracy_score(y_pred=np.array(predict_value).round(), y_true=true_label)
        roc_auc = roc_auc_score(y_score=predict_value, y_true=true_label)
        precision, recall, _ = precision_recall_curve(probas_pred=predict_value, y_true=true_label)
        pr_auc = auc(recall, precision)
        return accuracy, roc_auc, pr_auc

    def implement(self, data_file_name, ratio=0.8):
        Train_Validate_Set = Datasets(data_file_name, False)
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)
        TestLoader = loader.DataLoader(dataset=Datasets(data_file_name, True),
                                       batch_size=1, shuffle=False, num_workers=0)
        self.practise(TrainLoader, ValidateLoader)
        predict_value, true_label = self.demonstration(TestLoader)
        accuracy, roc_auc, pr_auc = self.estimate(predict_value, true_label)
        return accuracy, roc_auc, pr_auc


from models.DeepSTF import deepstf
Train = Constructor(model=deepstf())
Train.implement(data_file_name='data')
