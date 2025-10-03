# -*- coding: UTF-8 -*-
# @Date    :2023/4/18 22:43
# @Author  :高猛
# @Project :DTPs 
# @File    :utils.py
# @IDE     :PyCharm

import copy
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


class DM:
    def __init__(self, dim, bound=None):
        self.dim = dim
        if bound:
            self.bou = True
            self.bound = np.array([[bound[0], bound[1]] for _ in range(dim)])  # 存储数组各个位置的上下界
            self.max_distance = np.linalg.norm(self.bound[:, 0] - self.bound[:, 1])
        else:
            self.bou = False
            self.bound = np.array([[float('+Inf'), float('-Inf')] for _ in range(dim)])  # 存储数组各个位置的上下界
            self.max_distance = float('+Inf')

    def test(self, u, v):

        if not self.bou:
            # 更新上界、下界
            for i in range(self.dim):
                self.bound[i][0] = u[i] if u[i] < self.bound[i][0] else self.bound[i][0]
                self.bound[i][1] = u[i] if u[i] > self.bound[i][1] else self.bound[i][1]
                self.bound[i][0] = v[i] if v[i] < self.bound[i][0] else self.bound[i][0]
                self.bound[i][1] = v[i] if v[i] > self.bound[i][1] else self.bound[i][1]
            self.max_distance = np.linalg.norm(self.bound[:, 0] - self.bound[:, 1])

        # 计算dm
        dm = 0.
        for i in range(self.dim):
            lb = self.bound[i][0]
            ub = self.bound[i][1]
            ub = lb + 1 if ub == lb else ub
            dm += float(np.absolute(u[i] - v[i])) / float(ub - lb)
        dm /= self.dim
        return dm
        # return np.linalg.norm(np.array(u) - np.array(v)) / self.max_distance

    def update_bound(self, u):
        # 更新上界、下界
        for i in range(self.dim):
            self.bound[i][0] = u[i] if u[i] < self.bound[i][0] else self.bound[i][0]
            self.bound[i][1] = u[i] if u[i] > self.bound[i][1] else self.bound[i][1]
            self.bound[i][1] = self.bound[i][0] + 0.1 if self.bound[i][0] == self.bound[i][1] else self.bound[i][1]

    def get_bound(self):
        return self.bound


class buffer:
    def __init__(self):
        self.dm = DM(10, bound=[-5, 5])
        self.MAX_SIZE = 1e7
        self.memory = dict()  #{'t' : {'f_best', 'x_pso', 'model', 'f_next', 'x_best', 'f_this'}...}

    def store_data(self, t, data_dict):
        if str(t) not in self.memory.keys():
            self.memory[str(t)] = dict()
        for k, v in data_dict.items():
            self.memory[str(t)][k] = v

    def get_all_data(self):
        return self.memory

    def get_pre_data(self, t):
        x_list = []
        f_list = []
        for t_ in range(t):
            x_list.append(list(self.memory[str(t_)]["x_best"]))
            f_list.append(self.memory[str(t_)]["f_next"])
        return x_list, f_list

    def get_net_data(self, t):
        x_list = []
        f_list = []
        for t_ in range(t):
            x_list.append(self.memory[str(t_)]["x_pso"])
            f_list.append(self.memory[str(t_)]["f_best"])
        return x_list, f_list

    def data_sample(self, sample_size):
        sample_index = np.random.choice(range(len(self.memory)), sample_size)
        return np.array(copy.deepcopy(self.memory),  dtype=object)[sample_index, :]

    def get_pre_data2(self, t):
        x_list = []
        f_this = []
        f_next = []
        for t_ in range(t):
            x_list.append(list(self.memory[str(t_)]["x_best"]))
            f_this.append(self.memory[str(t_)]["f_this"])
            f_next.append(self.memory[str(t_)]["f_next"])
        return x_list, f_this, f_next

    def make_batch2(self, data_in, data_out):
        X_data = []
        y_data = []
        for i, di in enumerate(data_in):
            for j, dj in enumerate(data_in):
                if i == j:
                    pass
                else:
                    X_data.append(di + dj)
                    v = (data_out[i] - data_out[j])
                    y_data.append([v, -v])
        return X_data, y_data

    def choose_ssx(self, data, mlp, model='mlp'):
        length = len(data)
        score = np.zeros((length, ))
        d_lst = torch.FloatTensor(np.array(data))
        for i in range(length):
            for j in range(length):
                if i == j:
                    pass
                else:
                    # value = mlp(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0))
                    if model == 'rbf':
                        value = mlp.predict(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0))[0]
                    else:
                        value = mlp(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0).unsqueeze(0))[0]
                        # value = mlp(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0))
                    # if value[0] > value[1]:
                    score[i] += value[0]
        return int(np.argmax(score))

    def choose_ssx2(self, data, mlp, model='mlp'):
        length = len(data)
        score = np.zeros((length, ))
        d_lst = torch.FloatTensor(np.array(data))
        for i in range(length):
            for j in range(length):
                if i == j:
                    pass
                else:
                    # value = mlp(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0))
                    if model == 'rbf':
                        value = mlp.predict(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0))[0]
                    else:
                        value = mlp(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0).unsqueeze(0))[0]
                        # value = mlp(torch.cat([d_lst[i, :], d_lst[j, :]], dim=0))
                    if value[0] > value[1]:
                        score[i] += 1
        return int(np.argmax(score))


    def reduce(self, x_list, f_list):
        fm = np.max(f_list)
        x_choose = []
        f_choose = []
        for i, xi in enumerate(x_list):
            perm = True
            for x_ in x_choose:
                if self.dm.test(xi, x_) < 0.03:
                    perm = False
            if perm:
                x_choose.append(x_list[i])
                f_choose.append(f_list[i])

        return x_choose, f_choose

    def detection(self, t):
        # get data
        x_tra, f_this, f_next = self.get_pre_data2(t - 1)
        # using k-means to cluster
        center_ = 2
        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
        km_list = np.array([[f] for f in f_next], dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)
        # clustering f in two set according to label and checking which is positive set
        two_set = [[], []]

        for i in range(len(list(k_means_labels))):
            for k_ in range(center_):
                if int(k_means_labels[i]) == int(k_):
                    two_set[k_].append(f_next[i])
        self.positive_label = 0 if np.mean(two_set[0]) > np.mean(two_set[1]) else 1

        # calculate the difference index and make a judgment
        mu = [np.mean(f_next), np.mean(two_set[0]), np.mean(two_set[1])]
        sigma = [np.std(f_next), np.std(two_set[0]), np.std(two_set[1])]
        c_ang = sum([np.linalg.norm([mu[i] - mu[(i + 1) % 3], sigma[i] - sigma[(i + 1) % 3]]) for i in range(3)])
        l_vec = sum([np.linalg.norm([mu[i], sigma[i]]) for i in range(3)])
        dif_index = c_ang / l_vec

        if_return = True if dif_index < 0.9 else False

        return [x_tra, k_means_labels], if_return

    @classmethod
    def adaptive_kmeans(cls, data, alpha=0.13, n_max=6):
        s1, m1 = np.std(data), np.mean(data)
        std_list = [np.std(data)]
        de_cen = n_max
        im = 0
        a_list = [0.]
        a2_list = [0.5]
        s_ = [[s1]]
        m_ = [[m1]]
        for center_ in range(2, n_max):
            k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
            km_list = np.array([[x] for x in data], dtype=float)
            k_means.fit(np.array(km_list))

            k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
            k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

            # f_cen
            f_cen = [[] for _ in range(center_)]
            for i in range(len(list(k_means_labels))):
                for k_ in range(center_):
                    if int(k_means_labels[i]) == int(k_):
                        f_cen[k_].append(data[i])
            temp_means = [np.mean(f_cen[i]) for i in range(center_)]
            temp_std = [np.std(f_cen[i]) for i in range(center_)]
            m_.append(temp_means)
            s_.append(temp_std)
            ang_list = [cls.angle_between_vectors([si, mi], [s1, m1]) for si, mi in zip(temp_std, temp_means)]

            ang = sum(ang_list) / (center_ * 180)
            a_list.append(ang)
            a2_list.append(ang)

            std_list.append(np.max(temp_std))
            im = std_list[-2] - std_list[-1] if std_list[-2] - std_list[-1] > im else im
            if (a_list[-1] - a_list[-2]) / a2_list[-2] < 0.1:
            # if a_list[-1] - a_list[-2] <= alpha:
                de_cen = center_ - 1
                break

        # from sklearn.cluster import DBSCAN
        # from sklearn import metrics
        # X = [[i] for i in data]
        # db = DBSCAN(eps=0.01, min_samples=10).fit(X)
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_
        # de_cen = len(set(labels)) - (1 if -1 in labels else 0)
        #
        # print(f'         ====== center ======  :   {labels}   {de_cen}')
        print(f'==== means ===== : {[[round(j, 2) for j in i]for i in m_]}         '
              f' ==== std === : {[[round(j, 2) for j in i]for i in s_]}')
        de_cen = 4

        if de_cen > 1:
            k_means = KMeans(init='k-means++', n_clusters=de_cen, n_init=10)
            km_list = np.array([[x] for x in data], dtype=float)
            k_means.fit(np.array(km_list))

            k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
            k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

            i_cen = [[] for _ in range(de_cen)]

            for i in range(len(list(k_means_labels))):
                for k_ in range(de_cen):
                    if int(k_means_labels[i]) == int(k_):
                        i_cen[k_].append(i)
            return i_cen, de_cen
        else:
            return [list(range(len(data)))], de_cen

    @classmethod
    def angle_between_vectors(cls, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.
        cos_angle = dot_product / norm_product
        angle = np.arccos(cos_angle)
        return np.degrees(angle)















