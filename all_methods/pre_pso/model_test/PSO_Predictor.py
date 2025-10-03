# -*- coding: UTF-8 -*-
# @Date    :2024/1/18 23:17
# @Author  :高猛
# @Project :code_cl_test20240109 
# @File    :PSO+Predictor.py
# @IDE     :PyCharm
# _*_ encoding = utf-8 _*_
# @Time : 2022/11/19 10:20
# @Author : 高猛
# @File : Model.py
# @Software : PyCharm

import copy
import os
import random

import numpy as np
import time
import torch  # 导入torch
import torch.nn as nn  # 导入torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from dtp_base.Predictor import Predictor
from model_test.rbfn import RBFN
from model_test.MLP import *
from model_test.utils import *
from sklearn.metrics.pairwise import pairwise_distances_argmin


class Model:
    def __init__(self, args):
        self.x_bound = None
        self.kc = None
        self.f_std_ = None
        self.f_mean_ = None
        self.args = args
        self.x_dim = args.x_dim
        self.x = []
        self.P = {}

        self.buffer = buffer()
        self.predictor = Predictor(args)
        self.real_data = []  # [s, x, r, s_]
        self.Lt = {}
        self.choose_pso = 0
        self.net_set = {}
        self.temp_data = {}

    def choose_individual(self, t, pso_dict):
        # 存储数据, 并训练预测器
        self._train_predictor(t, pso_dict)

        # 当t <= 5时, 选用PSO
        if t <= 5:
            x_best = pso_dict['best_x']
            data_temp = {'x_best': copy.deepcopy(x_best)}
            self.buffer.store_data(t, data_temp)
            return x_best

        # 是否进行交叉
        x_our_ = self._get_my_solution(t, pso_dict)
        x_pso_ = pso_dict['best_x']

        if self.args.if_crossing:
            key_index = self.predictor.get_ncc_index()
            x_dec_ = [x_our_[ni] if ni in key_index else x_pso_[ni] for ni in range(self.x_dim)]
        else:
            x_dec_ = x_our_

        # 是否需要反馈
        if self.args.if_feedback:
            x_best = x_pso_ if self.choose_pso else x_dec_
        else:
            x_best = x_dec_

        # 存储最优解，并返回
        self.buffer.store_data(t, {'x_best': copy.deepcopy(x_best)})
        return copy.deepcopy(x_best)

    def _get_my_solution(self, t, pso_dict):
        x_list = pso_dict['x_list']
        f_list = pso_dict['fit_list']
        fp_list = []
        for index_, x_ in enumerate(x_list):
            f_pre, _, _ = self.predictor.predict(list(x_) + [t])
            fp_list.append(f_pre + f_list[index_])

        return x_list[np.argmax(fp_list)]

    def get_tra_data(self, t):
        x_tra, f_this, f_next = self.buffer.get_pre_data2(t - 1)
        return x_tra, f_this, f_next

    '''=========================  other function  ============================='''
    def km_plot(self, x_list, f_list, x_choose, f_choose, xd, fd, labels, t, kc):
        # 绘制数据点
        x = np.array(x_list)
        y = np.array(f_list)
        xc = np.array(x_choose)
        yc = np.array(f_choose)

        plt.figure()
        plt.scatter(x[:, 0], y, c=labels, cmap='viridis')
        plt.scatter(xc[:, 0], yc, s=20, c='red')

        plt.scatter(xd[0], fd, c='r', marker='X', s=300)
        plt.axvline(x=0, color='black', linestyle='--')
        plt.xlabel(r'$X_1$')
        plt.ylabel('Fitness of x')
        plt.title(f'k-means centers: {self.args.k_means_center})  time step: ({t})  key variable:({kc})')

        save_path = f'{self.args.save_k_means}/{self.args.bt_change}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += '/step_fig'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(f'{save_path}/step_{t}.png')
        if (t+1) % 300 == 0:
            plt.show()
        else:
            plt.close()

    def plot_data(self, x_tra, f_this, f_next, t, kc):
        t_lst = list(range(len(f_this)))
        x1_lst = [xi[0] for xi in x_tra]
        f_this = self.min_max_normalization(f_this, 1)
        f_next = self.min_max_normalization(f_next, 2)
        f_val = self.min_max_normalization([i + j for i, j in zip(f_this, f_next)], 3)

        plt.figure()
        # plt.plot(t_lst, x1_lst, 'ro-', label='x1_lst')
        # plt.plot(t_lst, f_this, 'b*-', label='f_this')
        # plt.plot(t_lst, f_next, 'ko--', label='f_next')
        # plt.plot(t_lst, f_val, 'go-', label='f_val ')

        # plt.plot(x1_lst, x1_lst, 'ro-', label='x1_lst')
        for xi in x1_lst:
            plt.axvline(x=xi, color='g', linestyle='--', alpha=0.5, linewidth=0.5)
        plt.scatter(x1_lst, f_this, c='b', marker='o', label='f_this')
        plt.scatter(x1_lst, f_next, c='k', marker='^', label='f_next')
        plt.scatter(x1_lst, f_val, c='r', marker='x', label='f_val ')

        plt.legend()
        plt.title(f'Value({t}) -- key variable:({kc})')

        save_path = f'{self.args.save_k_means}/{self.args.bt_change}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += '/value_fig'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(f'{save_path}/step_value_{t}.png')
        if t < 10 or (t+1) % 30 == 0:
            plt.show()
        else:
            plt.close()

    def _train_predictor(self, t, pso_dict):
        # 将数据存入buffer
        x_pso_best = pso_dict['best_x']
        f_pso_best = pso_dict['best_v']

        # 训练出一个模型

        data_temp = {'f_best': copy.deepcopy(f_pso_best),
                     'x_pso': copy.deepcopy(x_pso_best)}

        self.buffer.store_data(t, data_temp)
        if t > 0:
            self.buffer.store_data(t - 1, {'f_next': copy.deepcopy(f_pso_best)})

        x_list, f_list = self.buffer.get_pre_data(t - 1)
        self.predictor.train(np.array(x_list), np.array(f_list), t - 1)

    def save_fit(self, t, f):
        self.buffer.store_data(t, {'f_this': copy.deepcopy(f)})

    @classmethod
    def min_max_normalization(cls, arr, v):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        scaled_arr = v + (arr - min_val) / (max_val - min_val)
        return scaled_arr