# -*- encoding = utf-8 -*-
# @Time : 2022/7/8 23:01
# @Author : 高猛
# @File : Predictor.py
# @Software : PyCharm
import copy

import numpy as np
import random
import statsmodels.api as sm
import scipy.optimize as optimize
from scipy.stats import mannwhitneyu as rank_sum
from minepy import MINE
from scipy.stats import pearsonr, spearmanr, kendalltau
import time

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin


def mic(x, y, ):
    x = np.array(copy.deepcopy(x))
    y = np.array(copy.deepcopy(y))

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)

    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()
#
#
# def mcn(x, y, alpha_=0.1):
#
#     x = np.array(copy.deepcopy(x))
#     y = np.array(copy.deepcopy(y))
#
#     x = (x - np.mean(x)) / (np.std(x) + 1e-5)
#     y = (y - np.mean(y)) / (np.std(y) + 1e-5)
#
#     mine = MINE(alpha=alpha_)
#     mine.compute_score(x, y)
#     return mine.mcn()
#
#
# def mev(x, y, alpha_=0.1):
#
#     x = np.array(copy.deepcopy(x))
#     y = np.array(copy.deepcopy(y))
#
#     x = (x - np.mean(x)) / (np.std(x) + 1e-5)
#     y = (y - np.mean(y)) / (np.std(y) + 1e-5)
#
#     mine = MINE(alpha=alpha_)
#     mine.compute_score(x, y)
#     return mine.mev()


def ncc(x, y):
    n = len(x)
    b = int(n ** 0.5)
    if (np.max(x) != np.min(x)) and (np.max(y) != np.min(y)):
        detax = (np.max(x) - np.min(x) + 0.00001 * (np.max(x) - np.min(x))) / float(b)
        detay = (np.max(y) - np.min(y) + 0.00001 * (np.max(y) - np.min(y))) / float(b)
        if detax != 0 and detay != 0:
            p = np.zeros((b, b))
            x1 = np.ceil((x - np.min(x) + 0.000005 * (np.max(x) - np.min(x))) / detax)
            y1 = np.ceil((y - np.min(y) + 0.000005 * (np.max(y) - np.min(y))) / detay)
            x1 = [1 if x <= 0 else x for x in x1]
            y1 = [1 if y <= 0 else y for y in y1]
            x1 = [b if x >= b else x for x in x1]
            y1 = [b if y >= b else y for y in y1]
            for i in range(n):
                p[int(x1[i]) - 1][int(y1[i]) - 1] += 1 / n
            ncc = 0
            for i in range(b):
                for j in range(b):
                    if p[i][j] != 0:
                        ncc += p[i][j] * np.log(p[i][j]) / np.log(b)
            for i in range(b):
                if np.sum(p[i]) != 0:
                    ncc -= (np.sum(p[i])) * np.log(np.sum(p[i])) / np.log(b)
            for i in range(b):
                p_i = np.sum([p_row[i] for p_row in p])
                if p_i != 0:
                    ncc -= p_i * np.log(p_i) / np.log(b)
        else:
            ncc = 0
    else:
        ncc = 0
    cor = np.cov(x, y)
    if cor[0, 1] < 0:
        ncc = -ncc
    return ncc


def pearsonr_correlation(x, y):
    corr, p_value = pearsonr(x, y)
    return corr


def kendall_correlation(x, y):
    corr, p_value = kendalltau(x, y)
    return corr


def spearmanr_correlation(x, y):
    corr, p_value = spearmanr(x, y)
    return corr



class Predictor:
    def __init__(self, args):
        self.args = args
        self.x_dim = args.x_dim
        self.Index = None
        self.Quadratic_param = None
        self.Stype_param = None
        self.Trigonometric_param = None
        self.ki = dict()
        self.alpha = args.alpha
        self.pre_bound = [0., 0.]
        self.Qv = 1
        self.Sv = 1
        self.Tv = 1
        self.Zpre = None
        self.t = 0

    def train(self, data_x_, data_y_, t):
        if t > 3:
            data_x_ = np.array(copy.deepcopy(data_x_))
            data_y = np.array(copy.deepcopy(data_y_))
            self.dim_reduction_ms(data_x_, data_y, t)

        self.t = t
        if t < 5:
            return False

        self.pre_bound[0] = self.pre_bound[0] if self.pre_bound[0] < np.min(data_y) else np.min(data_y)
        self.pre_bound[1] = self.pre_bound[1] if self.pre_bound[1] > np.max(data_y) else np.max(data_y)

        data_x = data_x_[:, self.Index]

        # 二次函数拟合
        Q_data = []
        for line in data_x:
            Q_line_data = []
            for i in range(len(self.Index)):
                for j in range(len(self.Index)):
                    Q_line_data.append(line[i] * line[j])
            for i in range(len(self.Index)):
                Q_line_data.append(line[i])
            Q_data.append(Q_line_data)
        X_model = sm.add_constant(np.array(Q_data))
        model = sm.GLS(data_y, X_model)
        self.Quadratic_param = model.fit().params

        # S型函数拟合
        S_data = []
        for line in data_x:
            S_line_data = []
            for i in range(len(self.Index)):
                S_line_data.append((1 / (1 + np.exp(self.alpha * line[i]))) - 0.5)
            S_data.append(S_line_data)
        X_model = sm.add_constant(np.array(S_data))
        model = sm.GLS(data_y, X_model)
        # print('二次函数的输入(%d) ： ' % len(X_model), X_model[:10], '\n二次函数的输出 :', datay[:10])
        self.Stype_param = model.fit().params

        # 三角函数拟合
        T_data = data_x[:, -1]
        p0 = [75., 40, np.pi / 6., 1]
        bound = [[20., 10., 0.01, 0.01], [120., 60, 3.1416, 5]]
        param = optimize.curve_fit(self.target_func, T_data, data_y, p0=p0, maxfev=500000, bounds=bound)
        self.Trigonometric_param = param[0]

        # 权重系数拟合 ki  Q, S, T
        y_list = [[self.get_y(data_x_[i, :], 'q') for i in range(data_x_.shape[0])],
                  [self.get_y(data_x_[i, :], 's') for i in range(data_x_.shape[0])],
                  [self.get_y(data_x_[i, :], 't') for i in range(data_x_.shape[0])]]
        mse_list = [[(y_list[j][i] - data_y[i]) ** 2 for i in range(len(data_y))] for j in range(3)]
        min_i = np.argmin([np.sum(mse_list[mi]) for mi in range(3)])
        rs_list = [round(rank_sum(mse_list[min_i], mse_list[mi])[1], 4) for mi in range(3)]
        i_list = [mi for mi in range(3) if rank_sum(mse_list[min_i], mse_list[mi])[1] > 0.05]
        fi_array = np.array([[y_list[i][j] for i in i_list] for j in range(len(y_list[0]))])
        model = sm.GLS(data_y, fi_array)
        _ki = model.fit().params
        ki = []
        ki_index = 0
        for i in range(3):
            if i in i_list:
                ki.append(_ki[ki_index])
                ki_index += 1
            else:
                ki.append(0)
        self.ki = ki

    def target_func(self, x, a0, a1, a2, a3):
        return a0 + a1 * np.sin(a2 * x + a3)

    def dim_reduction_ms(self, data_x, data_y, t):
        data_x = np.array(data_x)
        data_y = np.array(data_y)

        if self.args.use_label:
            center1_ = 2
            k_means1 = KMeans(init='k-means++', n_clusters=center1_, n_init=10)
            km_list1 = np.array([[f] for f in data_y], dtype=float)
            k_means1.fit(np.array(km_list1))

            k_means_cluster_centers1 = np.sort(k_means1.cluster_centers_, axis=0)
            k_means_labels1 = pairwise_distances_argmin(np.array(km_list1), k_means_cluster_centers1)
            data_y = np.array(k_means_labels1)
        index_ = []
        values_ = []
        for i in range(self.x_dim):
            x_list = data_x[:, i]
            values_.append(self.cal_coe(x_list, data_y))

        for i, d in enumerate(values_):
            if d > np.mean(values_) + self.args.mic_a * np.std(values_):
                index_.append(i)
        if len(index_) < 1:
            index_.append(np.argmax(values_))

        self.Index = index_
        # self.Index = [0, 1]

    def dim_reduction_kmic(self, data_x, data_y, t):
        if self.args.use_label:
            center1_ = 2
            k_means1 = KMeans(init='k-means++', n_clusters=center1_, n_init=10)
            km_list1 = np.array([[f] for f in data_y], dtype=float)
            k_means1.fit(np.array(km_list1))

            k_means_cluster_centers1 = np.sort(k_means1.cluster_centers_, axis=0)
            k_means_labels1 = pairwise_distances_argmin(np.array(km_list1), k_means_cluster_centers1)
            data_y = np.array(k_means_labels1)

        Index_ = []
        NCC_ = []
        for i in range(self.x_dim):
            x_list = np.array(copy.deepcopy(data_x))[:, i]
            NCC_.append(self.cal_coe(x_list, data_y))

        if t < self.x_dim:
            Index_.append(np.argmax(NCC_))
        else:
            sort_index_ = np.argsort(NCC_)
            ncc_interval = [NCC_[sort_index_[i + 1]] - NCC_[sort_index_[i]] for i in range(len(NCC_) - 1)]
            Index_ += list(sort_index_[np.argmax(ncc_interval) + 1:])
        # print(f't={t}: key_component: {Index_}  ->  {[round(ni, 2) for ni in NCC_]}')
        self.Index = Index_
        # self.Index = [0, 1]

    def cal_coe(self, x, y):
        if self.args.key_variable == 'mic' or self.args.key_variable == 'm':
            x = np.array(copy.deepcopy(x))
            y = np.array(copy.deepcopy(y))

            x = (x - np.mean(x)) / (np.std(x) + 1e-5)
            y = (y - np.mean(y)) / (np.std(y) + 1e-5)

            mine = MINE()
            mine.compute_score(x, y)
            return mine.mic()
        elif self.args.key_variable == 'pearson' or self.args.key_variable == 'p':
            corr, p_value = pearsonr(x, y)
            return corr
        elif self.args.key_variable == 'ncc' or self.args.key_variable == 'n':
            n = len(x)
            b = int(n ** 0.5)
            if (np.max(x) != np.min(x)) and (np.max(y) != np.min(y)):
                detax = (np.max(x) - np.min(x) + 0.00001 * (np.max(x) - np.min(x))) / float(b)
                detay = (np.max(y) - np.min(y) + 0.00001 * (np.max(y) - np.min(y))) / float(b)
                if detax != 0 and detay != 0:
                    p = np.zeros((b, b))
                    x1 = np.ceil((x - np.min(x) + 0.000005 * (np.max(x) - np.min(x))) / detax)
                    y1 = np.ceil((y - np.min(y) + 0.000005 * (np.max(y) - np.min(y))) / detay)
                    x1 = [1 if x <= 0 else x for x in x1]
                    y1 = [1 if y <= 0 else y for y in y1]
                    x1 = [b if x >= b else x for x in x1]
                    y1 = [b if y >= b else y for y in y1]
                    for i in range(n):
                        p[int(x1[i]) - 1][int(y1[i]) - 1] += 1 / n
                    ncc = 0
                    for i in range(b):
                        for j in range(b):
                            if p[i][j] != 0:
                                ncc += p[i][j] * np.log(p[i][j]) / np.log(b)
                    for i in range(b):
                        if np.sum(p[i]) != 0:
                            ncc -= (np.sum(p[i])) * np.log(np.sum(p[i])) / np.log(b)
                    for i in range(b):
                        p_i = np.sum([p_row[i] for p_row in p])
                        if p_i != 0:
                            ncc -= p_i * np.log(p_i) / np.log(b)
                else:
                    ncc = 0
            else:
                ncc = 0
            cor = np.cov(x, y)
            if cor[0, 1] < 0:
                ncc = -ncc
            return ncc
        else:
            print("Error in find key variable.    args.key_variable:'mic', 'pearson', 'ncc'")
            raise RuntimeError



    def dim_reduction2(self, data_x, data_y, t):
        data_x = np.array(data_x)
        data_y = np.array(data_y)

        ncc_list = [ncc(data_x[:, i], data_y) for i in range(self.x_dim)]
        kendall_list = [kendall_correlation(data_x[:, i], data_y) for i in range(self.x_dim)]
        pearsonr_list = [pearsonr_correlation(data_x[:, i], data_y) for i in range(self.x_dim)]
        spearmanr_list = [spearmanr_correlation(data_x[:, i], data_y) for i in range(self.x_dim)]

        all_list = [ncc_list, kendall_list, pearsonr_list, spearmanr_list]
        r_list = []
        i_list = []
        for ai, al in enumerate(all_list):
            sort_index = np.argsort(-np.array(al))
            sort_data = np.array(al)[sort_index]
            edge = [sort_data[i] - sort_data[i + 1] for i in range(len(sort_data) - 1)]
            result_lst = sort_index[:np.argmax(edge) + 1]
            rate = np.max(edge) / (np.max(sort_data) - np.min(sort_data) + 1e-5)
            r_list.append(rate + float(np.e)**(-len(result_lst)))
            i_list.append(result_lst)

        if t <= self.x_dim:
            self.Index = [np.argmax(i_list[np.argmax(r_list)])]
        else:
            self.Index = i_list[np.argmax(r_list)]

    def predict(self, data):  # data = [x,f,t]
        if self.t <= 4:
            f_next = data[-2] + random.random()
            q_v = s_v = t_v = f_next
        else:
            q_v = self.get_y(data, 'q')
            s_v = self.get_y(data, 's')
            t_v = self.get_y(data, 't')
            f_next = self.ki[0] * q_v + self.ki[1] * s_v + self.ki[2] * t_v

        f_next = self.pre_bound[0] if self.pre_bound[0] > f_next else f_next
        f_next = self.pre_bound[1] if self.pre_bound[1] < f_next else f_next
        return f_next, q_v, s_v

    def get_y(self, z, type):
        # 二次函数拟合
        if type == 'Q' or type == 'q':
            Q_value = 0.
            datax = np.array(copy.deepcopy(z))[self.Index]
            for i in range(len(self.Index)):
                for j in range(len(self.Index)):
                    Q_value += self.Quadratic_param[int(1 + i * len(self.Index) + j)] \
                                           * datax[int(i)] * datax[int(j)]
            for i in range(len(self.Index)):
                Q_value += self.Quadratic_param[1 + len(self.Index) ** 2 + i] * datax[i]
            Q_value += self.Quadratic_param[0]
            return Q_value

        # S型函数拟合
        elif type == 'S' or type == 's':
            S_value = 0.
            datax = np.array(copy.deepcopy(z))[self.Index]
            for i in range(len(self.Index)):
                S_value += (1 / (1 + np.exp(self.alpha * datax[i])) - 0.5) * self.Stype_param[i + 1]
            S_value += self.Stype_param[0]
            return S_value

        # 三角函数拟合
        elif type == 'T' or type == 't':
            t_param = copy.deepcopy(self.Trigonometric_param)
            T_value = self.target_func(z[-1], t_param[0], t_param[1], t_param[2], t_param[3])
            return T_value

    def get_ncc_index(self):
        return self.Index
