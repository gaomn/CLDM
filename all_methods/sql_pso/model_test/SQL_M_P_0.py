# -*- coding: UTF-8 -*-
# @Date    :2023/6/1 22:41
# @Author  :高猛
# @Project :DTPs 
# @File    :SQL_M_P_0.py
# @IDE     :PyCharm

import csv
import random
import time

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest

from dtp_base.PSO import PSO
from model_test.rbfn import RBFN
import copy
import numpy as np
import scipy.optimize as optimize
import statsmodels.api as sm
from scipy.stats import mannwhitneyu as rank_sum
from scipy.stats import pearsonr, spearmanr, kendalltau



class DM:
    def __init__(self, dim, bound=None, zero_one=False):
        self.dim = dim
        self.update_bound = False
        self.zero_one = zero_one
        if bound is not None:
            self.buffer = copy.deepcopy(bound)
            # self.buffer = [[float(bound[0]), float(bound[1])] for _ in range(dim)]  # 存储数组各个位置的上下界
        else:
            self.update_bound = True
            self.buffer = [[float('+Inf'), float('-Inf')] for _ in range(dim)]  # 存储数组各个位置的上下界

    def test(self, u, v):
        # 更新上界、下界
        if self.update_bound:
            for i in range(self.dim):
                self.buffer[i][0] = u[i] if u[i] < self.buffer[i][0] else self.buffer[i][0]
                self.buffer[i][1] = u[i] if u[i] > self.buffer[i][1] else self.buffer[i][1]
                self.buffer[i][0] = v[i] if v[i] < self.buffer[i][0] else self.buffer[i][0]
                self.buffer[i][1] = v[i] if v[i] > self.buffer[i][1] else self.buffer[i][1]

        # 计算dm
        dm = 0.
        for i in range(self.dim):
            lb = self.buffer[i][0]
            ub = self.buffer[i][1]
            ub = lb + 1 if ub == lb else ub
            dm += float(np.absolute(u[i] - v[i])) / float(ub - lb)
        dm /= self.dim
        return dm

    def print_buffer(self, u):
        if self.update_bound:
            for i in range(self.dim):
                self.buffer[i][0] = u[i] if u[i] < self.buffer[i][0] else self.buffer[i][0]
                self.buffer[i][1] = u[i] if u[i] > self.buffer[i][1] else self.buffer[i][1]
        print(f'Buffer  :  {self.buffer}')

    def normalization(self, org_x):
        x = org_x
        if self.zero_one:
            if type(x) == type(1.) or type(x) == type(1):
                temp_x = (x - self.buffer[0][0]) / (self.buffer[0][1] - self.buffer[0][0])
            else:
                temp_x = [(xi - bo[0]) / (bo[1] - bo[0]) for xi, bo in zip(x, self.buffer)]
        else:
            if type(x) == type(1.) or type(x) == type(1):
                temp_x = 2 * (x - self.buffer[0][0]) / (self.buffer[0][1] - self.buffer[0][0]) - 1
            else:
                temp_x = [2 * (xi - bo[0]) / (bo[1] - bo[0]) - 1 for xi, bo in zip(x, self.buffer)]
        return temp_x

    def un_normalization(self, org_x):
        x = org_x
        if self.zero_one:
            if type(x) == type(1.) or type(x) == type(1):
                temp_x = self.buffer[0][0] + x * (self.buffer[0][1] - self.buffer[0][0])
            else:
                temp_x = [bo[0] + xi * (bo[1] - bo[0]) for xi, bo in zip(x, self.buffer)]
        else:
            if type(x) == type(1.) or type(x) == type(1):
                temp_x = self.buffer[0][0] + (x / 2 + 0.5) * (self.buffer[0][1] - self.buffer[0][0])
            else:
                temp_x = [bo[0] + (xi / 2 + 0.5) * (bo[1] - bo[0]) for xi, bo in zip(x, self.buffer)]
        return temp_x


class Buffer:
    def __init__(self, dm, delta):
        self.Q_list = []
        self.S_list = []
        self.dm = dm
        self.delta = delta

    def add_data(self, Q, S):
        self.Q_list.append(Q)
        self.S_list.append(S)

    def get_Q(self, state):
        index_num = 0
        Q = 0
        for si, s in enumerate(self.S_list):
            if self.dm.test(s, state) < self.delta:
                index_num += 1
                Q += float(self.Q_list[si])
        if index_num > 0:
            return float(Q)/float(index_num)
        else:
            return None

    def reset(self):
        self.Q_list = []
        self.S_list = []


class Model:
    def __init__(self, args):
        self.pred = None
        self.args = args
        self.s_dim = args.s_dim
        self.x_dim = args.x_dim

        self.P = {}
        self.s_dm = DM(self.s_dim)


        self.r_dm = DM(1, bound=[[-50., 50.]])
        self.x_dm = DM(self.x_dim, bound=[[-5., 5.] for _ in range(self.x_dim)])
        # -241.0086198778223 - --> 165.72992756180784
        self.Lt = {}
        self.Lt_qm = {}
        self.cul_t = 0
        self.buffer = Buffer(self.s_dm, self.args.delta)

        self.nki = 0
        self.t_ens = 0
        random.seed(self.args.rand_seed)
        np.random.seed(self.args.rand_seed)

    def modeling_updating(self, t, pso_dict, pred):
        self.t_ens = 0
        state = pso_dict['state']
        self.pred = pred
        time_mod1 = time.time()

        self.Lt[str(t)], self.Lt_qm[str(t)] = self.get_Lt(t, pso_dict)
        '''============================      check data       ==============================='''
        # with open(self.args.test_filename + f'Lt({str(t)})_data.csv', 'a', newline='') as f:
        #     # Lt = [{'s' : s, 'x' : x, 'r' : r, 'Q' : Q. 'Qm' : Q_max}, {...}, ...]
        #     for k, line in enumerate(self.Lt[str(t)]):
        #         row = [int(k), round(line['x'][0], 2), line['r'], line['Q'], line['Qm'], line['s'], line['x']]
        #         writer = csv.writer(f)
        #         writer.writerow(row)

        time_mod2 = time.time()
        # print(f'====Mod({self.t_ens})====Get Lt({len(self.Lt[str(t)])}) time is:{(time_mod2 - time_mod1)}')

        self.modeling(state, t)
        time_mod3 = time.time()
        # print(f'====Mod({self.t_ens})====Modeling time is:{(time_mod3 - time_mod2)}')

        if t > 3:
            self.updating(state, t)
        time_mod4 = time.time()
        # print(f'====Mod({self.t_ens})====Updating time is:{(time_mod4 - time_mod3)}')

    def modeling(self, state, t):
        # Molding
        x_lst = np.array([data['x'] for data in self.Lt[str(t)]])
        q_lst = np.array([[data['Q']] for data in self.Lt[str(t)]])

        x1_lst_ = [line[0] for line in x_lst]
        s1_lst_ = [s[0] for s in q_lst]

        plt.figure()
        s1 = plt.scatter(x1_lst_, s1_lst_, color='red', s=15, )
        plt.xlabel(r'x_1 Value', fontsize=18)
        plt.ylabel('value', fontsize=18)
        plt.title(f'model={t} ', fontsize=18)
        # plt.legend((s1), (f"s1_lst"), loc='best')
        plt.savefig(self.args.fig_filename + f'/step_model(b={self.args.time_fac})_{t}.jpg')
        plt.close()

        rbf = RBFN(self.args.numCenter)
        rbf.fit(x_lst, q_lst)
        self.P[str(t)] = [copy.deepcopy(state), rbf]

    def updating(self, state, t_cur):
        """
        Lt = [{'s' : s, 'x' : x, 'r' : r, 'Q' : Q}, {...}, ...]
        """
        # Updating
        for temp_t, lt in self.Lt.items():
            if self.s_dm.test(self.P[str(temp_t)][0],   state) > self.args.delta:
                continue

            lt_updated = []
            q_max = self.Lt_qm[str(temp_t)]
            for data in lt:
                q, qm = self.get_Q(state, data['x'], data['r'], int(temp_t), q_old=data['Q'])
                # q_max = q_max if q_max > q else q
                q_max = q_max if q_max > qm else qm

                lt_updated.append({'s': state, 'x': data['x'], 'r': data['r'], 'Q': q, 'Qm': qm})

            x_lst = np.array([data['x'] for data in lt_updated])
            q_lst = np.array([[data['Q']] for data in lt_updated])

            self.Lt[str(temp_t)] = lt_updated
            self.Lt_qm[str(temp_t)] = q_max
            self.P[str(temp_t)][1].fit(x_lst, q_lst)

            """    =============================       draw a want   =================================      """
            if self.args.if_plot:
                x1_lst_ = []
                r_lst_ = []
                q_lst_ = []
                qm_lst_ = []
                for k, line in enumerate(lt_updated):
                    x1_lst_.append(float(line['x'][0]))
                    r_lst_.append(float(line['r']))
                    q_lst_.append(float(line['Q']))
                    qm_lst_.append(float(line['Qm']))

                plt.figure()
                s1 = plt.scatter(x1_lst_, r_lst_, color='red', s=15, )
                s2 = plt.scatter(x1_lst_, q_lst_, color='blue', s=10, marker='*')
                s3 = plt.scatter(x1_lst_, qm_lst_, color='g', s=8, marker='^')
                plt.xlabel(r'x_1 Value', fontsize=18)
                plt.ylabel('value', fontsize=18)
                plt.title(f'predict step={temp_t} / {t_cur}', fontsize=18)
                plt.legend((s1, s2, s3), (f"r_lst", f"Q_lst", f'Qm_lst'), loc='best')
                plt.savefig(self.args.fig_filename + f'/step(b={self.args.time_fac})_{t_cur}_to({temp_t}).jpg')
                # plt.show()
                plt.close()
            """    =============================       save data   =================================      """
            with open(self.args.test_filename + f'/Lt(b={self.args.time_fac}_{str(temp_t)})_data.csv', 'a', newline='') as f:
                # Lt = [{'s' : s, 'x' : x, 'r' : r, 'Q' : Q. 'Qm' : Q_max}, {...}, ...]
                for k, line in enumerate(lt_updated):
                    row = [int(k), round(line['x'][0], 2), line['r'], line['Q'], line['Qm'], line['s'], line['x']]
                    writer = csv.writer(f)
                    writer.writerow(row)

    def get_Qens(self, state, x):
        self.t_ens += 1
        Qens = 0
        num = 0.
        p_list = [[v[0], v[1]] for v in self.P.values()]
        random.shuffle(p_list)
        d_list = []
        for [s, net] in p_list:
            d_ = self.s_dm.test(state, s)
            d_list.append(d_)
            if d_ <= self.args.delta:
                num += 1.
                en = float(net.predict(np.array([x]))[0])
                Qens += en
            # if num >= 10.:
            #     break
        if num != 0:
            Qensm = Qens / float(num)
        else:
            Qensm = float(p_list[int(np.argmin(d_list))][1].predict(np.array([x]))[0])

        return Qensm

    def get_Q(self, state, x, r, t, q_old=None):
        """
        input: s,x,r,t

        output: Q

        function:
            1. Q = (1 - alpha) * Q(s, a) + alpha * (r + eta * maxQ(s', x')) if t > 3 else r + rand()

        """
        if t <= 4:
            Q = r + random.random() / 150.
            r_next = r + random.random() / 150.
        else:
            s_predict = self.pred.predict(state, self.x_dm.un_normalization(x))
            r_next = self.get_max_q(s_predict)
            # if random.random() < 0.01:
            #     print(f'r_next: {r_next: < 6.2f} --> s_now: ({x[0] : < 6.2f})    '
            #           f'--> s_next: ({s_predict[0]: < 6.2f}, {s_predict[1]: < 6.2f})  ')

            alpha_ = 200. / (300. + float(t))
            if q_old is None:
                Q = (1 - alpha_) * self.get_Qens(state, x) + alpha_ * (r + self.args.eta_ * r_next)
            else:
                Q = (1 - alpha_) * q_old + alpha_ * (r + self.args.eta_ * r_next)

        return Q, r_next

    def get_Lt(self, t, pso_dict):
        """
        input: pso_dict = {'state': [np.min(fit_list), np.max(fit_list) - np.min(fit_list)],
                           'pop': pop,
                           'x_list': x_list,
                           'fit_list': fit_list,
                           'best_x': self.get_bestPosition(),
                           'best_v': self.get_bestFitnessValue()}

        output: Lt = [{'s' : s, 'x' : x, 'r' : r, 'Q' : Q}, {...}, ...]

        function:
             1. dis(xi, xj) > delta with any xi and xj in Lt
             2. size(Lt) <= Ns
             3. normalize all x
        """

        state = pso_dict['state']
        x_list = pso_dict['x_list']
        f_list = pso_dict['fit_list']

        x_lst = []
        f_lst = []
        for xi, xd in enumerate(x_list):
            perm = True
            for xl in x_lst:
                if self.x_dm.test(xd, xl) < self.args.delta:
                    perm = False
                    break
            if perm:
                x_lst.append(xd)
                f_lst.append(f_list[xi])
            if len(x_lst) >= 200:
                break
        if len(x_lst) < 200:
            for xi in np.random.permutation(np.arange(len(x_list))):
                x_lst.append(x_list[xi])
                f_lst.append(f_list[xi])
                if len(x_lst) >= 200:
                    break

        Lt = []

        Qmax = -1000.
        for x_index, x in enumerate(x_lst):
            norm_x = self.x_dm.normalization(x)
            norm_r = float(f_lst[x_index]) / 100.
            Q, qm = self.get_Q(state, norm_x, norm_r, t)
            Qmax = Qmax if Qmax >= Q else Q
            Lt.append({'s': state, 'x': norm_x, 'r': norm_r, 'Q': Q, 'Qm': qm})
        return Lt, Qmax

    def get_max_q(self, state):
        qm = 0.
        num = 0.
        d_list = []
        t_list = []
        for t_temp, lt_temp in self.Lt.items():
            data = lt_temp[0]
            d_ = self.s_dm.test(state, data['s'])
            d_list.append(d_)
            t_list.append(t_temp)
            if d_ <= self.args.delta:
                num += 1.
                qm += float(self.Lt_qm[t_temp])
        if num > 0.:
            q_max = qm / num
        else:
            q_max = float(self.Lt_qm[t_list[np.argmin(d_list)]])
        return q_max

    def get_x0(self, t, state, pop, pre):
        # epsilon-greedy 贪婪策略 epsilon = 0.1

        np.random.seed(self.args.rand_seed)
        rand_list = np.random.rand(t + 1)
        if rand_list[-1] < self.args.epsilon:
            return pop[int(len(pop) * random.random())]
        data = []

        x1_lst_ = []
        q_lst_ = []
        ri_lst_ = []

        for x in pop:
            Qens = 0.
            num = 0.
            s_next = pre.predict(state, x)

            p_list = [[v[0], v[1]] for v in self.P.values()]
            d_list = []
            for [s, net] in p_list:
                dm = self.s_dm.test(s_next, s)
                d_list.append(dm)
                if dm <= self.args.delta:
                    num += 1
                    Qens += float(net.predict(np.array([self.x_dm.normalization(x)]))[0])
            Q = Qens / num if num >= 1 else \
                float(p_list[int(np.min(d_list))][1].predict(np.array([self.x_dm.normalization(x)]))[0])

            RI = 1. / ((1. + np.min(d_list)) ** self.args.Lambda)
            data.append([Q, RI])

            x1_lst_.append(float(x[0]))
            q_lst_.append(float(Q))
            ri_lst_.append(float(RI))

        return pop[int(np.argmax(q_lst_))]

    def get_x(self, t, state, pop, pre):
        # epsilon-greedy 贪婪策略 epsilon = 0.1

        np.random.seed(self.args.rand_seed)
        rand_list = np.random.rand(t + 1)
        if rand_list[-1] < self.args.epsilon:
            return pop[int(len(pop) * random.random())]
        data = []

        x1_lst_ = []
        q_lst_ = []
        ri_lst_ = []

        for x in pop:
            Qens = 0.
            num = 0.
            s_next = pre.predict(state, x)

            p_list = [[v[0], v[1]] for v in self.P.values()]
            d_list = []
            for [s, net] in p_list:
                dm = self.s_dm.test(s_next, s)
                d_list.append(dm)
                if dm <= self.args.delta:
                    num += 1
                    Qens += float(net.predict(np.array([self.x_dm.normalization(x)]))[0])
            Q = Qens / num if num >= 1 else \
                float(p_list[int(np.min(d_list))][1].predict(np.array([self.x_dm.normalization(x)]))[0])

            RI = 1. / ((1. + np.min(d_list)) ** self.args.Lambda)
            data.append([Q, RI])

            x1_lst_.append(float(x[0]))
            q_lst_.append(float(Q))
            ri_lst_.append(float(RI))

        # 创建孤立森林模型
        clf = IsolationForest(n_estimators=100, contamination=0.25, max_samples=256, random_state=42)
        # 拟合数据
        clf.fit(data)
        # 预测样本的标签，1表示正常个体，-1表示异常个体
        labels = clf.predict(data)

        plt.figure(figsize=(8, 6))
        plt.scatter(x1_lst_, q_lst_, c=labels, cmap='coolwarm')
        plt.colorbar(label='Label (1: Normal, -1: Anomaly)')
        plt.xlabel(f'$X_1--{len([i for i in labels if i == 1])}$')
        plt.ylabel('Q')
        plt.title(f'Isolation Forest - Anomaly Detection : {len([i for i in labels if i == 1])}')
        plt.grid(True)
        plt.savefig(self.args.fig_filename + f'/IsolationForest_q(b={self.args.time_fac})_{t}.jpg')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(q_lst_, ri_lst_, c=labels, cmap='coolwarm')
        plt.colorbar(label='Label (1: Normal, -1: Anomaly)')
        plt.xlabel('Q')
        plt.ylabel('RI')
        plt.title('Isolation Forest - Anomaly Detection')
        plt.grid(True)
        plt.savefig(self.args.fig_filename + f'/IsolationForest_r(b={self.args.time_fac})_{t}.jpg')
        plt.close()

        index = np.argsort(q_lst_)[::-1]
        for i in index:
            if labels[i] == 1:
                return pop[i]

        return pop[int(np.argmax(q_lst_))]

    def test(self, config):
        if config['name'] == 'model':
            f = self.get_Qens(config['state'], self.x_dm.normalization(config['x']))
            return f

    def time_test(self, t, state, x_lst, condition=False):
        if condition:
            print(f'\n\n     ------------------------- test Q-model   -----------------------------\n')
            te_0 = time.time()
            test_list = []
            for i in range(5000):
                norm_x = self.x_dm.normalization(x_lst[i])
                test_list.append(norm_x)
            te_1 = time.time()
            print(f'                             normalizing 5000 x cost {te_1 - te_0} (s)   ')

            for i in range(5000):
                self.x_dm.test(random.choice(test_list), random.choice(test_list))
            te_2 = time.time()
            print(f'                 getting the distance of 5000 x cost {te_2 - te_1} (s)   ')

            for i in range(5000):
                float(self.P[str(t - 1)][1].predict(np.array([test_list[i]]))[0])
            te_3 = time.time()
            print(f'                   getting the Q-value of 5000 x cost {te_3 - te_2} (s)   ')

            for i in range(5000):
                float(self.get_Qens(state, test_list[i]))
            te_4 = time.time()
            print(f'                 getting the Q-ens of 5000 x cost {te_4 - te_3} (s)   ')

            print(f'\n     ------------------------- test Q-model   -----------------------------\n\n')


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
        if t < 4:
            return False
        x1_lst_ = [line[2] for line in self.Zprv]
        s1_lst_ = [s[0] for s in self.Snxt]
        s2_lst_ = [s[1] for s in self.Snxt]

        plt.figure()
        s1 = plt.scatter(x1_lst_, s1_lst_, color='red', s=15, )
        s2 = plt.scatter(x1_lst_, s2_lst_, color='blue', s=10, marker='*')
        plt.xlabel(r'x_1 Value', fontsize=18)
        plt.ylabel('value', fontsize=18)
        plt.title(f'predict step={t} -- (b={self.args.time_fac})', fontsize=18)
        plt.legend((s1, s2), (f"s1_lst", f"s2_lst"), loc='best')
        plt.savefig(self.args.fig_filename + f'/train_data(b={self.args.time_fac})_{t}.jpg')
        plt.close()

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
        self.check_pre()

    def target_func(self, x, a0, a1, a2, a3):
        return a0 + a1 * np.sin(a2 * x + a3)

    def dim_reduction(self, Zprv, Snxt, t):
        for j in range(2):
            y_list = [s[j] for s in Snxt]
            Ij = []
            NCC_ = []
            for i in range(2 + self.x_dim):
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
            z = list(s) + list(x) + [self.t]
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
        t = self.t
        random_x_1 = np.arange(-5, 5, 0.1)
        rx = [[random_x_1[i]] + [5 * random.random() for _ in range(10)] for i in range(len(random_x_1))]
        s_nxt = [self.predict(self.Snxt[-1], xi) for xi in rx]
        s1_lst_ = [s[0] for s in s_nxt]
        s2_lst_ = [s[1] for s in s_nxt]

        plt.figure()
        s1 = plt.scatter(random_x_1, s1_lst_, color='red', s=15, )
        s2 = plt.scatter(random_x_1, s2_lst_, color='blue', s=10, marker='*')
        plt.xlabel(r'$x_1 Value$', fontsize=18)
        plt.ylabel('value', fontsize=18)
        plt.title(f'predict-fl step={t} -- (b={self.args.time_fac})', fontsize=18)
        plt.legend((s1, s2), (f"$s_1 lst$", f"$s_2 lst$"), loc='best')
        plt.savefig(self.args.fig_filename + f'/train_fl(b={self.args.time_fac})_{t}.jpg')
        plt.close()





