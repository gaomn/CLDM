# -*- coding: UTF-8 -*-
# @Date    :2023/4/18 22:43
# @Author  :高猛
# @Project :DTPs 
# @File    :utils.py
# @IDE     :PyCharm

import copy
import torch
import numpy as np
from minepy import MINE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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


def Cohens_d(d1, d2):
    n1 = len(d1)
    n2 = len(d2)
    if n1 <= 1 or n2 <= 1:
        print(f'n1 = {n1}, n2 = {n2}  ERROR!!!')
        raise

    mean_1, mean_2 = np.mean(d1), np.mean(d2)
    return np.abs(mean_1 - mean_2)
    # var_1, var_2 = np.var(d1, ddof=1), np.var(d2, ddof=1)

    pooled_var = ((n1 - 1) * var_1 + (n2 - 1) * var_2) / (n1 + n2 - 2) + 1e-6
    pooled_std = np.sqrt(pooled_var)

    cohens_d = np.abs(mean_1 - mean_2) * pooled_std
    return cohens_d


def plot_all(data_dict, id_dict):
    step = 95
    c_old = {}
    c_new = {}
    kc = {}
    for id in sorted(data_dict.keys()):
        data = data_dict[id]
        x_list, _, f_next = data[0][:step], data[1][:step], data[2][:step]
        kc[id_dict[id]] = dim_reduction_ms(x_list, f_next)
        c_old[id_dict[id]] = cul_old(f_next)
        c_new[id_dict[id]] = cul_new(x_list, f_next)
    print(kc)
    keys = [f'b={i[2:]}' for i in list(c_old.keys())[:6]]
    values_old = list(c_old.values())[:6]
    values_new = list(c_new.values())[:6]

    plt.figure(figsize=(8, 5))  # 设置图形大小

    plt.plot(keys, values_old, marker='o', label='Old')  # 绘制第一个字典的折线
    plt.plot(keys, values_new, marker='o', label='New')  # 绘制第二个字典的折线

    plt.xlabel('Keys')  # x轴标签
    plt.ylabel('Values')  # y轴标签
    plt.title('Comparison of Dictionaries')  # 图表标题
    plt.legend()  # 显示图例

    plt.grid(True)  # 显示网格线
    plt.tight_layout()  # 调整布局
    plt.savefig(f'test.png', dpi=300, bbox_inches='tight')
    plt.show()  # 显示图形


def cul_old(f_next):
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

    mu = [np.mean(f_next), np.mean(f_cen[0]), np.mean(f_cen[1])]
    sigma = [np.std(f_next), np.std(f_cen[0]), np.std(f_cen[1])]
    c_ang = sum([np.linalg.norm([mu[i] - mu[(i + 1) % 3], sigma[i] - sigma[(i + 1) % 3]]) for i in range(3)])
    l_vec = sum([np.linalg.norm([mu[i], sigma[i]]) for i in range(3)])
    rd = c_ang / l_vec
    return rd


def cul_new(x_tra, f_next):
    kc = dim_reduction_ms(x_tra, f_next)

    # f_next = np.array(copy.deepcopy(f_next))
    # f_next = (f_next - np.mean(f_next)) / (np.std(f_next) + 1e-5)
    f_next = [(x) / (np.max(f_next)) for x in f_next]
    x_dec = [[xid for xii, xid in enumerate(xi) if xii in kc] for xi in x_tra]
    # x_dec = [[xid for xii, xid in enumerate(xi)] for xi in x_tra]

    n_dec = 10 if len(f_next) >= 12 else len(f_next)-2
    k_means2 = KMeans(init='k-means++', n_clusters=n_dec, n_init=10)
    k_means2.fit(np.array(x_dec))
    k_means_cluster_centers2 = np.sort(k_means2.cluster_centers_, axis=0)
    k_means_labels2 = pairwise_distances_argmin(np.array(x_dec), k_means_cluster_centers2)
    f_n = [[] for _ in range(n_dec)]
    for i, d in enumerate(k_means_labels2):
        f_n[d].append(f_next[i])

    indices_to_remove = [i for i, sublist in enumerate(f_n) if len(sublist) <= 1]
    for index in sorted(indices_to_remove, reverse=True):
        del f_n[index]

    Cohens_d_sum = []
    n_dec = len(f_n)
    for i in range(n_dec):
        for j in range(n_dec):
            if i != j:
                Cohens_d_sum.append(Cohens_d(f_n[i], f_n[j]))
    # ind = int(len(Cohens_d_sum) * 0.1)
    ind = int(len(Cohens_d_sum) * 1)
    sum_of_top_10_percent = sum(sorted(Cohens_d_sum, reverse=True)[:ind])

    # print("最大的10%数据之和为:", sum_of_top_10_percent)
    # Cohens_d_sum /= float(5 * n_dec * float(n_dec - 1))
    # sum_of_top_10_percent /= 10*100
    # sum_of_top_10_percent /= float(n_dec*(n_dec - 1))
    sum_of_top_10_percent = sum_of_top_10_percent if n_dec >= 2 else 0.
    return sum_of_top_10_percent


def cul_new_1(x_tra, f_next):
    kc = dim_reduction_ms(x_tra, f_next)

    # f_next = np.array(copy.deepcopy(f_next))
    # f_next = (f_next - np.mean(f_next)) / (np.std(f_next) + 1e-5)
    x_dec = [[xid for xii, xid in enumerate(xi) if xii in kc] for xi in x_tra]

    n_dec = 6
    k_means2 = KMeans(init='k-means++', n_clusters=n_dec, n_init=10)
    k_means2.fit(np.array(x_dec))
    k_means_cluster_centers2 = np.sort(k_means2.cluster_centers_, axis=0)
    k_means_labels2 = pairwise_distances_argmin(np.array(x_dec), k_means_cluster_centers2)
    f_n = [[] for _ in range(n_dec)]
    for i, d in enumerate(k_means_labels2):
        f_n[d].append(f_next[i])

    indices_to_remove = [i for i, sublist in enumerate(f_n) if len(sublist) <= 1]
    for index in sorted(indices_to_remove, reverse=True):
        del f_n[index]

    Cohens_d_sum = 0
    n_dec = len(f_n)
    for i in range(n_dec):
        for j in range(n_dec):
            if i != j:
                Cohens_d_sum += Cohens_d(f_n[i], f_n[j])
    Cohens_d_sum /= float(5 * n_dec * float(n_dec - 1))
    return Cohens_d_sum


def dim_reduction_ms(data_x, data_y):
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    if True:
        center1_ = 2
        k_means1 = KMeans(init='k-means++', n_clusters=center1_, n_init=10)
        km_list1 = np.array([[f] for f in data_y], dtype=float)
        k_means1.fit(np.array(km_list1))

        k_means_cluster_centers1 = np.sort(k_means1.cluster_centers_, axis=0)
        k_means_labels1 = pairwise_distances_argmin(np.array(km_list1), k_means_cluster_centers1)
        # data_y = np.array(k_means_labels1)
    index_ = []
    values_ = []
    for i in range(10):
        x_list = data_x[:, i]
        x_list = np.array(copy.deepcopy(x_list))
        data_y = np.array(copy.deepcopy(data_y))
        x_list = (x_list - np.mean(x_list)) / (np.std(x_list) + 1e-5)
        data_y = (data_y - np.mean(data_y)) / (np.std(data_y) + 1e-5)

        mine = MINE()
        mine.compute_score(x_list, data_y)

        values_.append(mine.mic())

    for i, d in enumerate(values_):
        if d > np.mean(values_) + 0.8 * np.std(values_):
        # if d > np.mean(values_):
            index_.append(i)
    if len(index_) < 1:
        index_.append(np.argmax(values_))

    return index_
    # self.Index = [0, 1]


def plot_all2(data_dict1, data_dict2, id_dict1, id_dict2):
    step = 95
    c_old = {}
    c_new = {}
    kc = {}
    for id in sorted(data_dict1.keys()):
        data = data_dict1[id]
        x_list, _, f_next = data[0][:step], data[1][:step], data[2][:step]
        kc[id_dict1[id]] = dim_reduction_ms(x_list, f_next)
        c_old[id_dict1[id]] = cul_old(f_next)
        c_new[id_dict1[id]] = cul_new(x_list, f_next)
    print(kc)
    keys1 = list(c_old.keys())
    values_old1 = list(c_old.values())
    values_new1 = list(c_new.values())

    step = 95
    c_old = {}
    c_new = {}
    kc = {}
    for id in sorted(data_dict2.keys()):
        data = data_dict2[id]
        x_list, _, f_next = data[0][:step], data[1][:step], data[2][:step]
        kc[id_dict2[id]] = dim_reduction_ms(x_list, f_next)
        c_old[id_dict2[id]] = cul_old(f_next)
        c_new[id_dict2[id]] = cul_new(x_list, f_next)
    print(kc)
    keys2 = list(c_old.keys())
    values_old2 = list(c_old.values())
    values_new2 = list(c_new.values())


    # y1, y2, y3, y4 = plot_data[0], plot_data[1], plot_data[2], plot_data[3]

    chunk_size = 6
    m_list = []
    for p_num in [1]:
        for c_type in ['con', 'dis']:
            for b_type in ['line', 'sin', 'cir']:
                m_list.append(f'p{p_num}-{c_type}-{b_type}')

    y1_chunks = [values_old1[i:i + chunk_size] for i in range(0, len(values_old1), chunk_size)]
    y2_chunks = [values_new1[i:i + chunk_size] for i in range(0, len(values_new1), chunk_size)]

    y3_chunks = [values_old2[i:i + chunk_size] for i in range(0, len(values_old2), chunk_size)]
    y4_chunks = [values_new2[i:i + chunk_size] for i in range(0, len(values_new2), chunk_size)]

    re = int(3)
    # print(f'{len(y1_chunks)}     {len(y1_chunks[0])}   {re}')
    y_data1 = zip(y1_chunks, y2_chunks)
    y_data2 = zip(y3_chunks, y4_chunks)
    m1 = m_list[:re]
    m2 = m_list[re:]

    # 绘制折线图
    x = list(range(36))
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(2, figsize=(10, 8))

    x_ticks = []
    x_offset = 0
    i = 0
    for chunk1, chunk2 in y_data1:
        # x = list(range(i * chunk_size, i * chunk_size + len(chunk1))
        x = list(range(i * chunk_size + x_offset, i * chunk_size + len(chunk1) + x_offset))
        axs[0].plot(x, chunk1, marker='o', linestyle='-', color='blue')
        axs[0].plot(x, chunk2, marker='s', linestyle='--', color='red')
        # axs[0].plot(x, chunk3, marker='^', linestyle='-.', color='green')
        # axs[0].plot(x, chunk4, marker='*', linestyle=':', color='black')
        # 在相邻的子列表之间添加 None，使它们不连接
        if i < len(y1_chunks) - 1:
            axs[0].plot([x[-1], x[-1] + 1], [chunk1[-1], None], linestyle='-', color='none')
            # plt.axvline(x[-1], linestyle='--', color='magenta', alpha=0.5)
            x_offset += 1

        i += 1
        text_x = sum(x) / len(x)
        x_ticks.append(text_x)
    axs[0].legend(['old-method', 'new-method'], loc='upper right')
    print(x_ticks, m1)
    axs[0].set_xticks(x_ticks, m1)
    axs[0].set_ylabel('values')
    axs[0].set_title('First Plot')
    axs[0].grid(True)

    x_ticks = []
    x_offset = 0
    i = 0
    for chunk1, chunk2 in y_data2:
        # x = list(range(i * chunk_size, i * chunk_size + len(chunk1))
        x = list(range(i * chunk_size + x_offset, i * chunk_size + len(chunk1) + x_offset))
        axs[1].plot(x, chunk1, marker='o', linestyle='-', color='blue')
        axs[1].plot(x, chunk2, marker='s', linestyle='--', color='red')
        # axs[1].plot(x, chunk3, marker='^', linestyle='-.', color='green')
        # axs[1].plot(x, chunk4, marker='*', linestyle=':', color='black')
        # 在相邻的子列表之间添加 None，使它们不连接
        if i < len(y1_chunks) - 1:
            axs[0].plot([x[-1], x[-1] + 1], [chunk1[-1], None], linestyle='-', color='none')
            # plt.axvline(x[-1], linestyle='--', color='magenta', alpha=0.5)
            x_offset += 1

        i += 1
        text_x = sum(x) / len(x)
        x_ticks.append(text_x)
    axs[1].legend(['old-method', 'new-method'], loc='upper right')
    axs[1].set_xticks(x_ticks, m2)
    axs[1].set_ylabel('values')
    axs[1].set_title('Second Plot')
    axs[1].grid(True)

    plt.tight_layout()  # 调整子图之间的间距，使其更加美观
    plt.savefig(f'data_save/other_fig/test.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_all3(data_lst, id_lst):
    step = 100
    c_old = {}
    c_new = {}
    kc = {}
    for data_id, data_dict in enumerate(data_lst):
        for id in sorted(data_dict.keys()):
            data = data_dict[id]
            x_list, _, f_next = data[0][:step], data[1][:step], data[2][:step]
            f_next = [(x ) / (np.max(f_next)) for x in f_next]
            kc[id_lst[data_id][id]] = dim_reduction_ms(x_list, f_next)
            c_old[str(data_id) + id_lst[data_id][id]] = cul_old(f_next)
            c_new[str(data_id) + id_lst[data_id][id]] = cul_new(x_list, f_next)

    print(kc)
    keys = list(c_old.keys())
    values_old = list(c_old.values())
    values_new = list(c_new.values())
    #
    values_old = [(x - min(values_old)) / (max(values_old) - min(values_old)+100) for x in values_old]
    # values_new = [(x - min(values_new)) / (max(values_new) - min(values_new)) for x in values_new]


    chunk_size = 6
    m_list = []
    for p_num in [1, 10]:
        for c_type in ['dis', 'con']:
            for b_type in ['line', 'sin', 'cir']:
                m_list.append(f'p{p_num}-{c_type}-{b_type}')

    leng = int(len(values_old)/2)
    y1_chunks = [values_old[i:i + chunk_size] for i in range(0, leng, chunk_size)]
    y2_chunks = [values_new[i:i + chunk_size] for i in range(0, leng, chunk_size)]

    y3_chunks = [values_old[i + leng:i + leng + chunk_size] for i in range(0, leng, chunk_size)]
    y4_chunks = [values_new[i + leng:i + leng + chunk_size] for i in range(0, leng, chunk_size)]

    re = int(6)
    # print(f'{len(y1_chunks)}     {len(y1_chunks[0])}   {re}')
    y_data1 = zip(y1_chunks, y2_chunks)
    y_data2 = zip(y3_chunks, y4_chunks)
    m1 = m_list[:re]
    m2 = m_list[re:]

    # 绘制折线图
    x = list(range(36))
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(2, figsize=(10, 8))

    x_ticks = []
    x_offset = 0
    i = 0
    for chunk1, chunk2 in y_data1:
        # x = list(range(i * chunk_size, i * chunk_size + len(chunk1))
        x = list(range(i * chunk_size + x_offset, i * chunk_size + len(chunk1) + x_offset))
        axs[0].plot(x, chunk1, marker='o', linestyle='-', color='blue')
        axs[0].plot(x, chunk2, marker='s', linestyle='--', color='red')
        # axs[0].plot(x, chunk3, marker='^', linestyle='-.', color='green')
        # axs[0].plot(x, chunk4, marker='*', linestyle=':', color='black')
        # 在相邻的子列表之间添加 None，使它们不连接
        if i < len(y1_chunks) - 1:
            axs[0].plot([x[-1], x[-1] + 1], [chunk1[-1], None], linestyle='-', color='none')
            # plt.axvline(x[-1], linestyle='--', color='magenta', alpha=0.5)
            x_offset += 1

        i += 1
        text_x = sum(x) / len(x)
        x_ticks.append(text_x)
    axs[0].legend(['old-method', 'new-method'], loc='upper right')
    print(x_ticks, m1)
    axs[0].set_xticks(x_ticks, m1)
    axs[0].set_ylabel('values')
    axs[0].set_title('First Plot')
    axs[0].grid(True)

    x_ticks = []
    x_offset = 0
    i = 0
    for chunk1, chunk2 in y_data2:
        # x = list(range(i * chunk_size, i * chunk_size + len(chunk1))
        x = list(range(i * chunk_size + x_offset, i * chunk_size + len(chunk1) + x_offset))
        axs[1].plot(x, chunk1, marker='o', linestyle='-', color='blue')
        axs[1].plot(x, chunk2, marker='s', linestyle='--', color='red')
        # axs[1].plot(x, chunk3, marker='^', linestyle='-.', color='green')
        # axs[1].plot(x, chunk4, marker='*', linestyle=':', color='black')
        # 在相邻的子列表之间添加 None，使它们不连接
        if i < len(y1_chunks) - 1:
            axs[0].plot([x[-1], x[-1] + 1], [chunk1[-1], None], linestyle='-', color='none')
            # plt.axvline(x[-1], linestyle='--', color='magenta', alpha=0.5)
            x_offset += 1

        i += 1
        text_x = sum(x) / len(x)
        x_ticks.append(text_x)
    axs[1].legend(['old-method', 'new-method'], loc='upper right')
    axs[1].set_xticks(x_ticks, m2)
    axs[1].set_ylabel('values')
    axs[1].set_title('Second Plot')
    axs[1].grid(True)

    plt.tight_layout()  # 调整子图之间的间距，使其更加美观
    plt.savefig(f'data_save/other_fig/test.png', dpi=300, bbox_inches='tight')
    plt.show()



