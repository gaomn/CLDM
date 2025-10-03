# -*- coding: UTF-8 -*-
# @Date    :2023/4/27 21:45
# @Author  :高猛
# @Project :DTPs 
# @File    :MLP.py
# @IDE     :PyCharm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class MLP_tanh(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim):
        super(MLP_tanh, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.l2(x)
        return x


class MLP_relu(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim):
        super(MLP_relu, self).__init__()
        self.l1 = nn.Linear(in_dim, hid_dim)
        self.l2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class GNN(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim):
        super(GNN, self).__init__()
        self.WQ = nn.Linear(input_dim, hid_dim)
        self.WK = nn.Linear(input_dim, hid_dim)
        self.WV = nn.Linear(input_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        x = torch.matmul(attention_weights, V)
        x = self.fc(x)  # 输出层
        return x

    # def forward(self, x):
    #     Q = self.WQ(x)
    #     K = self.WK(x)
    #     V = self.WV(x)
    #     attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
    #     attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
    #     x = torch.bmm(attention_weights, V)
    #     x = self.fc(x)  # 输出层
    #     return x


def train_net(train_epoch, X_data, y_data, model, optimizer, loss_func, save_path=None, t=None):
    train_X = torch.FloatTensor(X_data)
    train_y = torch.FloatTensor(y_data)
    if X_data.shape[0] > 50:
        p = int(X_data.shape[0]/2)
        X1 = train_X[:p, :]
        X2 = train_X[p:, :]
        y1 = train_y[:p, :]
        y2 = train_y[p:, :]

        loss_lst = []
        for i in range(train_epoch):
            y_pre = model(X1)
            loss1 = loss_func(y1, y_pre)
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            y_pre = model(X2)
            loss2 = loss_func(y2, y_pre)
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            loss_lst.append((float(loss1)+float(loss2))/2.)
    else:
            loss_lst = []
            for i in range(train_epoch):
                y_pre = model(train_X)
                loss = loss_func(train_y, y_pre)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_lst.append(float(loss))

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += '/step_fig'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.figure()
        plt.plot(loss_lst)
        plt.xlabel('Epoch -- loss_min is {:.2g}'.format(loss_lst[-1]), fontsize=18)
        plt.ylabel('loss', fontsize=12)
        plt.title(f'step = {t}', fontsize=18)
        # plt.legend(loc="best")
        plt.savefig(f'{save_path}/step_{t}.png')
        plt.close()

