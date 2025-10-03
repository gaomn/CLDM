# -*- coding: UTF-8 -*-
# @Date    :2023/5/24 14:45
# @Author  :高猛
# @Project :DTPs
# @File    :SQL_Predictor.py
# @IDE     :PyCharm

import copy
import random
import numpy as np
import scipy.optimize as optimize
import statsmodels.api as sm
from scipy.stats import mannwhitneyu as rank_sum
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau


class Predictor:
    def __init__(self, args):
        self.args = args
        self.s_dim = args.s_dim
        self.x_dim = args.x_dim
        self.I = {}
        self.Quadratic_param = {}
        self.Stype_param = {}
        self.Trigonometric_param = {}
        self.ki = dict()
        self.alpha = args.alpha
        self.Qv = 1
        self.Sv = 1
        self.Tv = 1
        self.Zpre = None
        self.t = 0
        self.bound = [[float('+Inf'), float('-Inf')], [float('+Inf'), float('-Inf')]]
        self.Zprv = []
        self.Snxt = []

    def add_data(self, z, s):
        self.Zprv.append(copy.deepcopy(z))
        self.Snxt.append(copy.deepcopy(s))

    def train(self, t):

        self.update_bound(self.Snxt)

        self.t = t
        if t <= 4:
            return False


        self.dim_reduction(self.Zprv, self.Snxt, t)
        for s_num in range(2):
            datax = [[line[int(i)] for i in self.I[str(s_num)]] for line in self.Zprv]
            datay = np.array([s_all[s_num] for s_all in self.Snxt], dtype='float64')

            # 二次函数拟合
            Q_data = []
            for line in datax:
                Qline_data = []
                for i in range(len(self.I[str(s_num)])):
                    for j in range(len(self.I[str(s_num)])):
                        Qline_data.append(line[i] * line[j])
                for i in range(len(self.I[str(s_num)])):
                    Qline_data.append(line[i])
                Q_data.append(Qline_data)
            X_model = sm.add_constant(np.array(Q_data, dtype='float64'))
            model = sm.GLS(datay, X_model)
            self.Quadratic_param[str(s_num)] = model.fit().params

            # S型函数拟合
            S_data = []
            for line in datax:
                Sline_data = []
                for i in range(len(self.I[str(s_num)])):
                    Sline_data.append((1 / (1 + np.exp(self.alpha * line[i]))) - 0.5)
                S_data.append(Sline_data)
            X_model = sm.add_constant(np.array(S_data, dtype='float64'), has_constant='add')
            model = sm.GLS(datay, X_model)
            # print('二次函数的输入(%d) ： ' % len(X_model), X_model[:10], '\n二次函数的输出 :', datay[:10])
            self.Stype_param[str(s_num)] = model.fit().params
            # print(f" s_num : {s_num}   data1 : {S_data}    data2 : {X_model}   param : {self.Stype_param[str(s_num)]}")

            # 三角函数拟合
            T_data = np.array([float(s_all[-1]) for s_all in self.Zprv], dtype='float64')
            p0 = [75., 40, np.pi / 6., 1]
            bound = [[20., 10., 0.01, 0.01], [120., 60, 3.1416, 5]]
            param = optimize.curve_fit(self.target_func, T_data, datay, p0=p0, maxfev=50000, bounds=bound)[0]

            # param, _ = optimize.curve_fit(self.target_func, T_data, datay, maxfev=500000)
            self.Trigonometric_param[str(s_num)] = param
        # print(f'==========train===={self.Trigonometric_param}=========== : ')

        # 权重系数拟合 ki  Q, S, T
        for s_num in range(2):

            datay = np.array([s_all[s_num] for s_all in self.Snxt])
            y_list = [[self.get_y(self.Zprv[i], 'q')[s_num] for i in range(len(self.Zprv))],
                      [self.get_y(self.Zprv[i], 's')[s_num] for i in range(len(self.Zprv))],
                      [self.get_y(self.Zprv[i], 't')[s_num] for i in range(len(self.Zprv))]]
            # print('===ylist===: ', y_list)
            mse_list = [[(y_list[j][i] - datay[i]) ** 2 for i in range(len(self.Zprv))] for j in range(3)]
            min_i = np.argmin([np.sum(mse_list[mi]) for mi in range(3)])
            i_list = [mi for mi in range(3) if rank_sum(mse_list[min_i], mse_list[mi])[1] > 0.05]
            fi_array = np.array([[y_list[i][j] for i in i_list] for j in range(len(y_list[0]))])
            # print('===fi===: ', fi_array)
            model = sm.GLS(datay, fi_array)
            _ki = model.fit().params
            ki = []
            ki_index = 0
            for i in range(3):
                if i in i_list:
                    ki.append(_ki[ki_index])
                    ki_index += 1
                else:
                    ki.append(0)
            self.ki[str(s_num)] = ki
            # print('===========Pred(%d)===========' % s_num, _ki)
        # self.check_pre()

    def target_func(self, x, a0, a1, a2, a3):
        return a0 + a1 * np.sin(a2 * x + a3)

    def dim_reduction(self, Zprv, Snxt, t):
        for j in range(2):
            y_list = [s[j] for s in Snxt]
            Ij = []
            NCC_ = []
            for i in range(self.x_dim):
                x_list = np.array(Zprv)[:, i]
                # nk = self.C_NCC(x_list, y_list)
                # nk = np.cov(x_list, y_list)[0][1]
                nk = pearsonr(x_list, y_list)[0]

                NCC_.append(nk)
            if t <= self.x_dim + 3:
                Ij.append(np.argmax(NCC_))
            else:
                sort_index_ = np.argsort(NCC_)
                ncc_interval = [NCC_[sort_index_[i + 1]] - NCC_[sort_index_[i]] for i in range(len(NCC_) - 1)]
                Ij = list(sort_index_[np.argmax(ncc_interval) + 1:])
            self.I[str(j)] = Ij

    def dim_reduction2(self, Zprv, Snxt, t):
        for j in range(2):
            y_list = [s[j] for s in Snxt]
            Ij = []
            NCC = {}
            for i in range(2 + self.x_dim):
                x_list = [z[i] for z in np.array(Zprv)]
                ncc = self.C_NCC(x_list, y_list)
                NCC[str(i)] = ncc
            if t <= self.x_dim + 3:
                Ij.append(np.argmax([ncc for ncc in NCC.values()]))
            else:
                # np.sort是一个升序排列函数，通过加符号将NCC强行降序排列（其实没有必要）
                NCCstar = [-ki for ki in np.sort([-float(kj) for kj in NCC.values()])]
                n1 = int(np.argmax([NCCstar[i] - NCCstar[i + 1] for i in range(len(NCCstar) - 1)]))
                Ij = [k for k, v in NCC.items() if v in NCCstar[:n1 + 1]]
            self.I[str(j)] = Ij

    def predict(self, s, x):
        if self.t <= 4:
            s_next = [si + random.random() for si in s]
        else:
            # z = list(s) + list(x) + [self.t]
            z = list(x) + [self.t]
            q_v = self.get_y(z, 'q')
            s_v = self.get_y(z, 's')
            t_v = self.get_y(z, 't')
            s_next = [self.ki['0'][0] * q_v[0] + self.ki['0'][1] * s_v[0] + self.ki['0'][2] * t_v[0],
                      self.ki['1'][0] * q_v[1] + self.ki['1'][1] * s_v[1] + self.ki['1'][2] * t_v[1]]

            s_next[0] = self.bound[0][0] if self.bound[0][0] > s_next[0] else s_next[0]
            s_next[0] = self.bound[0][1] if self.bound[0][1] < s_next[0] else s_next[0]
            s_next[1] = self.bound[1][0] if self.bound[1][0] > s_next[1] else s_next[1]
            s_next[1] = self.bound[1][1] if self.bound[1][1] < s_next[1] else s_next[1]
        return s_next

    def get_y(self, z, type):
        # 二次函数拟合
        if type == 'Q' or type == 'q':
            Q_value = [0., 0.]
            for s_num in range(2):
                datax = [z[int(i)] for i in self.I[str(s_num)]]
                for i in range(len(self.I[str(s_num)])):
                    for j in range(len(self.I[str(s_num)])):
                        Q_value[int(s_num)] += self.Quadratic_param[str(s_num)][
                                                   int(1 + i * len(self.I[str(s_num)]) + j)] \
                                               * datax[int(i)] * datax[int(j)]
                for i in range(len(self.I[str(s_num)])):
                    Q_value[s_num] += self.Quadratic_param[str(s_num)][1 + len(self.I[str(s_num)]) ** 2 + i] * datax[i]
                Q_value[s_num] += self.Quadratic_param[str(s_num)][0]
            return Q_value

        # S型函数拟合
        elif type == 'S' or type == 's':
            S_value = [0., 0.]
            for s_num in range(2):
                datax = [z[int(i)] for i in self.I[str(s_num)]]

                for i in range(len(self.I[str(s_num)])):
                    # print(f'  num:  {s_num}     param({i}) : {self.Stype_param}      index: {self.I}')
                    S_value[s_num] += (1 / (1 + np.exp(self.alpha * datax[i])) - 0.5) * self.Stype_param[str(s_num)][
                        i + 1]
                S_value[s_num] += self.Stype_param[str(s_num)][0]
            return S_value

        # 三角函数拟合
        elif type == 'T' or type == 't':
            T_value = [0., 0.]
            t_0 = [int(i) for i in self.Trigonometric_param[str(0)]]
            t_1 = [int(i) for i in self.Trigonometric_param[str(1)]]
            T_value[0] = self.target_func(z[-1], t_0[0], t_0[1], t_0[2], t_0[3])
            T_value[1] = self.target_func(z[-1], t_1[0], t_1[1], t_1[2], t_1[3])
            return T_value
        else:
            print('Error in Predictor , ensure your key is right!')
            return None

    def get_y2(self, z, type):
        # 二次函数拟合
        if type == 'Q' or type == 'q':
            Q_value = [0., 0.]
            for s_num in range(2):
                datax = [z[int(i)] for i in self.I[str(s_num)]]
                for i in range(len(self.I[str(s_num)])):
                    for j in range(len(self.I[str(s_num)])):
                        Q_value[int(s_num)] += self.Quadratic_param[str(s_num)][
                                                   int(1 + i * len(self.I[str(s_num)]) + j)] \
                                               * datax[int(i)] * datax[int(j)]
                for i in range(len(self.I[str(s_num)])):
                    Q_value[s_num] += self.Quadratic_param[str(s_num)][1 + len(self.I[str(s_num)]) ** 2 + i] * datax[i]
                Q_value[s_num] += self.Quadratic_param[str(s_num)][0]
            return Q_value

        # S型函数拟合
        elif type == 'S' or type == 's':
            S_value = [0., 0.]
            for s_num in range(2):
                datax = [z[int(i)] for i in self.I[str(s_num)]]

                for i in range(len(self.I[str(s_num)])):
                    # print(f'  num:  {s_num}     param({i}) : {self.Stype_param}      index: {self.I}')
                    S_value[s_num] += (1 / (1 + np.exp(self.alpha * datax[i])) - 0.5) * self.Stype_param[str(s_num)][
                        i + 1]
                S_value[s_num] += self.Stype_param[str(s_num)][0]
            return S_value

        # 三角函数拟合
        elif type == 'T' or type == 't':
            T_value = [0., 0.]
            t_0 = [int(i) for i in self.Trigonometric_param[str(0)]]
            t_1 = [int(i) for i in self.Trigonometric_param[str(1)]]
            T_value[0] = self.target_func(z[-1], t_0[0], t_0[1], t_0[2], t_0[3])
            T_value[1] = self.target_func(z[-1], t_1[0], t_1[1], t_1[2], t_1[3])
            return T_value
        else:
            print('Error in Predictor , ensure your key is right!')
            return None

    def update_bound(self, Snxt):
        self.bound = [[float('+Inf'), float('-Inf')], [float('+Inf'), float('-Inf')]]
        s1_list_ = [si[0] for si in Snxt]
        s2_list_ = [si[1] for si in Snxt]
        self.bound[0][0] = self.bound[0][0] if self.bound[0][0] < np.min(s1_list_) else np.min(s1_list_)
        self.bound[0][1] = self.bound[0][1] if self.bound[0][1] > np.max(s1_list_) else np.max(s1_list_)
        self.bound[1][0] = self.bound[1][0] if self.bound[1][0] < np.min(s2_list_) else np.min(s2_list_)
        self.bound[1][1] = self.bound[1][1] if self.bound[1][1] > np.max(s2_list_) else np.max(s2_list_)

    @classmethod
    def C_NCC(cls, x, y):
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

    def check_pre(self):
        pass
        # t = self.t
        # random_x_1 = np.arange(-5, 5, 0.1)
        # rx = [[random_x_1[i]] + [5 * random.random() for _ in range(10)] for i in range(len(random_x_1))]
        # s_nxt = [self.predict(self.Snxt[-1], xi) for xi in rx]
        # s1_lst_ = [s[0] for s in s_nxt]
        # s2_lst_ = [s[1] for s in s_nxt]

        # plt.figure()
        # s1 = plt.scatter(random_x_1, s1_lst_, color='red', s=15, )
        # s2 = plt.scatter(random_x_1, s2_lst_, color='blue', s=10, marker='*')
        # plt.xlabel(r'$x_1 Value$', fontsize=18)
        # plt.ylabel('value', fontsize=18)
        # plt.title(f'predict-fl step={t} -- (b={self.args.time_fac})', fontsize=18)
        # plt.legend((s1, s2), (f"$s_1 lst$", f"$s_2 lst$"), loc='best')
        # plt.savefig(self.args.fig_filename + f'/train_fl(b={self.args.time_fac})_{t}.jpg')
        # plt.close()

    def get_ncc_index(self):
        if '0' in self.I.keys():
            return self.I[str(0)]
        else:
            return [i for i in range(self.x_dim)]