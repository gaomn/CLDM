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
            x_dec_ = self._get_my_solution(t, pso_dict)

        # 是否需要反馈
        if self.args.if_feedback:
            x_best = x_pso_ if self.choose_pso else x_dec_
        else:
            x_best = x_dec_
        # print(f'step({t})   x_pso: {round(x_pso_[0], 2)}  x_our: {round(x_our_[0], 2)}   x-cro: {round(x_dec_[0], 2)}')

        # 存储最优解，并返回
        self.buffer.store_data(t, {'x_best': copy.deepcopy(x_best)})
        return copy.deepcopy(x_best)

    def _get_my_solution(self, t, pso_dict):
        # 改变网络训练算法，减小训练时间
        t0 = time.time()

        x_list = pso_dict['x_list']
        f_list = pso_dict['fit_list']

        # 获取分类数据
        x_tra, f_this, f_next = self.buffer.get_pre_data2(t - 1)
        f_max = fm = np.max(f_next)

        # 利用k-means, 对f值进行二分类
        center_ = 2
        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
        km_list = np.array([[x] for x in f_next], dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

        # 根据标签，将x和f存入x_cen, f_cen
        i_cen = [[] for _ in range(center_)]
        f_cen = [[] for _ in range(center_)]
        for i in range(len(list(k_means_labels))):
            for k_ in range(center_):
                if int(k_means_labels[i]) == int(k_):
                    i_cen[k_].append(i)
                    f_cen[k_].append(f_next[i])

        # TODO: 是否先淘汰适应值不合格的解
        # f_this_max = np.max(f_list)
        # x_temp_list, f_temp_list = [], []
        # delta_f = np.abs(np.mean(f_cen[0]) - np.mean(f_cen[1]))
        # for x_temp, f_temp in zip(x_list, f_list):
        #     if f_temp >= f_this_max - delta_f:
        #         x_temp_list.append(x_temp)
        #         f_temp_list.append(f_temp)
        # x_list, f_list = x_temp_list, f_temp_list
        # TODO: END

        mu = [np.mean(f_next), np.mean(f_cen[0]), np.mean(f_cen[1])]
        sigma = [np.std(f_next), np.std(f_cen[0]), np.std(f_cen[1])]
        c_ang = sum([np.linalg.norm([mu[i] - mu[(i + 1) % 3], sigma[i] - sigma[(i + 1) % 3]]) for i in range(3)])
        l_vec = sum([np.linalg.norm([mu[i], sigma[i]]) for i in range(3)])
        rd = c_ang / l_vec

        # TODO: 修改阈值
        # if np.random.rand() >= 5*rd - 1.5:
        #     return pso_dict['best_x']

        if rd < 0.9:
            return pso_dict['best_x']
        # TODO: END



        # TODO： 是否使用聚类均值标签
        ave_list = [np.mean(np.array(f_next)[il]) for il in i_cen]
        s_cen = [[[x_tra[j], f_this[j]/fm, f_this[j]/fm + ave_list[i]/fm] for j in i_cen[i]] for i in range(center_)]

        # s_cen = [[[x_tra[j], f_this[j]/fm, f_this[j]/fm + f_next[j]/fm] for j in i_cen[i]] for i in range(center_)]
        # TODO: END

        # 制作数据集
        kc = self.predictor.get_ncc_index()

        all_data = []
        for i in s_cen:
            all_data += i
        train_in = [[xxd / 5. for xxi, xxd in enumerate(s[0]) if xxi in kc] + [s[1]] for s in all_data]
        train_out = [s[2] for s in all_data]

        X_data, y_data = self.buffer.make_batch2(train_in, train_out)

        X_data = np.array(X_data)
        y_data = np.array(y_data)

        t1 = time.time()

        if self.args.net_model == 'rbf':
            n_max = y_data.shape[0]
            cen = 32 if n_max > 32 else n_max - 1
            model = RBFN(cen)
            model.fit(X_data, y_data)
            self.net_set[str(t)] = [kc, model]
        else:
            if self.args.net_model == 'mlp_relu':
                model = MLP_relu(2 * len(kc) + 2, 2, 32)
            elif self.args.net_model == 'mlp_tanh':
                model = MLP_tanh(2 * len(kc) + 2, 2, 32)
            elif self.args.net_model == 'gnn':
                model = GNN(2 * len(kc) + 2, 2, 32)
            else:
                print("Error in model!  model='rbf', 'mlp_relu', 'mlp_tanh', 'gnn'")
                raise RuntimeError
            keys = list(self.net_set.keys())
            keys.reverse()
            if_load = False
            for k in keys:
                if np.all(self.net_set[k][0] == kc):
                    if_load = True
                    model.load_state_dict(self.net_set[k][1])
                    break
            train_epoch = 1000 if if_load else 2000

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            loss_func = torch.nn.MSELoss(reduction='mean')
            save_path = f'{self.args.save_k_means}/{self.args.mode}'

            train_net(train_epoch, X_data, y_data, model, optimizer, loss_func, save_path, t)
            self.net_set[str(t)] = [kc, model.state_dict()]

        t2 = time.time()

        # 使用k_means方法减少排序数据
        fp_list = []
        ncc_index = self.predictor.get_ncc_index()
        km_list = []
        for index_, x_ in enumerate(x_list):
            f_pre, _, _ = self.predictor.predict(list(x_) + [t])
            fp_list.append(f_pre/f_max)

            # TODO: 是否使用预测器聚类
            # km_ = [2 * f_list[index_]/f_max]

            km_ = [2 * f_pre/f_max]
            # TODO: END

            for ncc_i in ncc_index:
                km_.append(x_[ncc_i]/5.)

            km_list.append(copy.deepcopy(km_))

        # compute clustering with K-Means
        center_ = self.args.k_means_center
        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
        km_list = np.array(km_list, dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

        value_ = [0. for _ in range(center_)]
        pop_ = [[] for _ in range(center_)]
        fit_ = [[] for _ in range(center_)]
        fpre = [[] for _ in range(center_)]

        for ind_ in range(len(k_means_labels)):
            for i in range(center_):
                if k_means_labels[ind_] == i:
                    value_[i] += fp_list[ind_]
                    pop_[i].append(x_list[ind_])
                    fit_[i].append(f_list[ind_])
                    fpre[i].append(fp_list[ind_])

        x_choose = []
        f_choose = []

        # TODO: 是否使用预测器挑选初选解
        # for i in range(center_):
        #     if len(fit_[i]) <= 3:
        #         x_choose += pop_[i]
        #         f_choose += fit_[i]
        #     else:
        #         a_index = int(np.argmax(fit_[i]))
        #         x_temp = [pop_[i][a_index]]
        #         f_temp = [fit_[i][a_index]]
        #         for xi in np.random.permutation(len(fit_[i])):
        #             if fit_[i][xi] not in f_temp:
        #                 x_temp.append(pop_[i][xi])
        #                 f_temp.append(fit_[i][xi])
        #             if len(x_temp) >= 3:
        #                 x_choose += x_temp
        #                 f_choose += f_temp
        #                 break

        for i in range(center_):
            if len(fit_[i]) <= 5:
                x_choose += pop_[i]
                f_choose += fit_[i]
            else:
                a_index = int(np.argmax(fit_[i]))
                b_index = int(np.argmax(fpre[i]))
                c_index = int(np.argmax([i + j for i, j in zip(fit_[i], fpre[i])]))
                x_temp = [pop_[i][a_index]]
                f_temp = [fit_[i][a_index]]
                if a_index != b_index:
                    x_temp.append(pop_[i][b_index])
                    f_temp.append(fit_[i][b_index])
                if c_index != a_index and c_index != b_index:
                    x_temp.append(pop_[i][b_index])
                    f_temp.append(fit_[i][b_index])
                for xi in np.random.permutation(len(fit_[i])):
                    if pop_[i][xi] not in x_temp:
                        x_temp.append(pop_[i][xi])
                        f_temp.append(fit_[i][xi])
                    if len(x_temp) >= 5:
                        x_choose += x_temp
                        f_choose += f_temp
                        break
        # TODO: END

        x_sort, f_sort = x_choose, f_choose

        model_input = [[xxd/5. for xxi, xxd in enumerate(xi) if xxi in kc] + [fi/f_max]
                       for xi, fi in zip(x_sort, f_sort)]
        x_index = self.buffer.choose_ssx2(model_input, model,  model=self.args.net_model)

        t3 = time.time()
        # self.km_plot(x_list, f_list, x_choose, f_choose, x_sort[x_index], f_sort[x_index], k_means_labels, t, kc)
        self.plot_data(x_tra, f_this, f_next, t, kc)
        return x_sort[x_index]

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
        if t<10 or (t+1) % 30 == 0:
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