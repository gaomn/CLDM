# -*- coding: UTF-8 -*-
# @Date    :2023/5/11 20:21
# @Author  :高猛
# @Project :DTPs 
# @File    :test_k.py
# @IDE     :PyCharm

import os
import copy
import csv
import numpy as np
from minepy import MINE
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from utils.data_tool import list_dir
from utils.data_tool import get_data_list2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin


def add_from_str(str_list):
    data_list = []
    for si, se in enumerate(str_list):
        if se == '[' or se == ' ':
            start_i = si + 1
        if se == ',' or se == ']':
            end_i = si
            data_list.append(float(str_list[start_i:end_i]))
    return data_list


def get_data_list(filename):
    x_list = []
    a_list = []
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in list(reader)[-100:]:
            a_list.append(float(data[1]))
            x_list.append(add_from_str(data[3]))

    return a_list[1:], x_list


def get_data_list2(filename):
    a_list = []
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in list(reader)[-100:]:
            a_list.append(float(data[1]))

    return a_list[1:]


def cut_x(x_list):
    cut_list = [[] for _ in range(len(x_list[0]))]
    for x in x_list[:-1]:
        for i, xi in enumerate(x):
            cut_list[i].append(xi)
    return cut_list


def mic(x, y, alpha_=0.1):
    x = np.array(copy.deepcopy(x))
    y = np.array(copy.deepcopy(y))

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)

    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mic()


def mcn(x, y, alpha_=0.1):
    x = np.array(copy.deepcopy(x))
    y = np.array(copy.deepcopy(y))

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)

    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mcn()


def mev(x, y, alpha_=0.1):
    x = np.array(copy.deepcopy(x))
    y = np.array(copy.deepcopy(y))

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)

    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mev()


def mic(x, y, alpha_=0.8):
    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mic()


def pearsonr_correlation(x, y):
    corr, p_value = pearsonr(x, y)
    return corr


def spearmanr_correlation(x, y):
    corr, p_value = spearmanr(x, y)
    return corr


def cov_(x, y):
    x = np.array(copy.deepcopy(x))
    y = np.array(copy.deepcopy(y))

    # x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    # y = (y - np.mean(y)) / (np.std(y) + 1e-5)

    covariance = np.cov(x, y)[0][1]
    return covariance


def choose_some(in_list):
    in_array = np.array(copy.deepcopy(in_list))
    in_index = np.argsort(-in_array)
    in_sort = in_array[in_index]

    edge = [in_sort[i] - in_sort[i + 1] for i in range(len(in_sort) - 1)]
    result_lst = in_index[:np.argmax(edge) + 1]

    return result_lst


def cul_cor(filename, al=0.8):
    a_list, x_list = get_data_list(filename)

    xi_list = cut_x(x_list)
    yi_list = a_list[1:]

    n1 = 0.
    n2 = 0.
    n3 = 0.

    for t in range(4, len(yi_list)):
        cor_list = [mcn(xi_[: t], yi_list[: t], alpha_=al) for xi_ in xi_list]
        print(f'  {t}  --->  {choose_some(cor_list)}     ->     {cor_list}  ')
        n1 += 1.
        if 0 in choose_some(cor_list) and len(choose_some(cor_list)) == 1:
            n2 += 1.
        if 0 in choose_some(cor_list):
            n3 += 1.

    return n2 / n1, n3 / n1


def plot_ch(filename):
    a_list, x_list = get_data_list(filename)
    print(a_list)
    print(x_list)

    xi_list = cut_x(x_list)
    yi_list = a_list[1:]

    d0 = [x[0] for x in xi_list]
    plt.figure()
    plt.plot(d0[:20], 'ro--', label=r"$X$")
    # plt.plot(d2[:20], 'b^--', label=r"$PSO$")
    # plt.plot(d1[:20], 'go-', label=r"$Optim$")
    # plt.xlabel(f'MPB, b={int(f_n[7:10])}', fontsize=18)
    plt.ylabel('Accumulated Fitness', fontsize=12)
    plt.title(f'POC', fontsize=18)
    plt.legend(loc="best")
    # plt.savefig(f_n + f'.png')
    plt.show()
    return


def plot_fit(list_csv):
    d0 = None
    d1 = None
    d2 = None
    d3 = None

    for fi, filename in enumerate(list_csv):
        if '_POC' in filename:
            d1, x_list = get_data_list(filename)
            d0 = [5 * x[0] for x in x_list[:99]]
        elif '_PSO' in filename:
            d2 = get_data_list2(filename)
        elif '_OPT' in filename:
            d3 = get_data_list2(filename)

    print(f' 0 : {len(d0)}     1 : {len(d1)}   2 : {len(d2)}    3 : {len(d3)}')

    plt.figure()
    plt.plot(np.zeros(100)[:40], 'r-', label=r"$y=0$")
    plt.plot(d0[:40], 'k*--', label=r"$X_1$")
    plt.plot(d1[:40], 'ro--', label=r"$Our$")
    plt.plot(d2[:40], 'b^--', label=r"$PSO$")
    plt.plot(d3[:40], 'g.-', label=r"$Optim$")

    # plt.xlabel(f'MPB, b={int(f_n[7:10])}', fontsize=18)
    plt.ylabel('y', fontsize=12)
    plt.title(f'step', fontsize=18)
    plt.legend(loc="best")
    # plt.savefig(f_n + f'.png')
    plt.show()


def dis(mu1, mu2, s1, s2):
    return round(np.sqrt((mu1 - mu2) ** 2 + (s1 - s2) ** 2), 2)


def ang(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cross_product = np.cross(v1, v2)
    sin_angle = np.linalg.norm(cross_product) / (norm_v1 * norm_v2)

    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
    # dot_product = np.dot(v1, v2)
    # cos_angle = dot_product / (norm_v1 * norm_v2)
    return sin_angle


'''
 b = 100 : ang = 0.7855007861900226,  0.4419944614908296, 0.43107518712215565   === ,          mu = ([110.08, -82.02, 136.58]),          sigma = ([73.97, 26.44, 18.39])            D(478) = ([218.75, 197.89, 61.57]),        Rd1 = 3.6057533334080243,   Rd = 1.3409776011587629,   dmu = [218, 192, 26],    dsi = [8, 47, 55]        k:  1.0, 12.81 
 b = 100 : ang = 0.5014066269670798,  0.2556333017066059, 0.26356985950251366   === ,          mu = ([121.03, -42.63, 135.42]),          sigma = ([52.35, 5.01, 20.43])            D(384) = ([178.72, 170.37, 35.01]),        Rd1 = 2.9127935769794147,   Rd = 1.2321073110565062,   dmu = [178, 163, 14],    dsi = [15, 47, 31]        k:  1.0, 5.14 
 b = 50 : ang = 0.8185591312095832,  0.24735351276947262, 0.9352080201863818   === ,          mu = ([70.99, 4.23, 83.86]),          sigma = ([35.82, 29.05, 18.52])            D(168) = ([80.32, 67.1, 21.56]),        Rd1 = 2.125130719398089,   Rd = 0.8676670025010021,   dmu = [79, 66, 12],    dsi = [10, 6, 17]        k:  1.0, 2.29 
 b = 50 : ang = 0.9838374601268647,  0.2629297269801303, 0.9963023276745554   === ,          mu = ([76.63, -9.7, 88.54]),          sigma = ([37.8, 33.96, 17.23])            D(209) = ([99.65, 86.42, 23.77]),        Rd1 = 2.455823752823743,   Rd = 0.9946676856190316,   dmu = [98, 86, 11],    dsi = [16, 3, 20]        k:  1.0, 1.46 
 b = 10 : ang = 0.28629937887062235,  0.34294072630350136, 0.59752263872885   === ,          mu = ([35.61, 13.74, 49.24]),          sigma = ([20.66, 14.61, 8.74])            D(76) = ([35.98, 22.69, 18.11]),        Rd1 = 1.8649838761181754,   Rd = 0.6902514745074535,   dmu = [35, 21, 13],    dsi = [5, 6, 11]        k:  1.0, 3.06 
 b = 10 : ang = 0.6358616908837729,  0.30848185738121836, 0.8429380910128439   === ,          mu = ([37.15, 6.97, 47.35]),          sigma = ([22.13, 19.43, 10.77])            D(86) = ([41.3, 30.3, 15.27]),        Rd1 = 2.00893250377836,   Rd = 0.7725650770475938,   dmu = [40, 30, 10],    dsi = [8, 2, 11]        k:  1.0, 1.62 


'''


def norm_distance():
    path_dir = 'data_save/run_data/kxm/'
    b_list = [100, 50, 10]
    # center_ = 6
    for center_ in range(2, 6):
        for bi in b_list:
            rd1_list = []
            rd2_list = []
            path_file = path_dir + f'data_b{int(bi)}'
            list_csv = list_dir(path_file)
            mean_list = [[] for _ in range(center_ + 1)]
            std_list = [[] for _ in range(center_ + 1)]
            for fi, filename in enumerate(list_csv[:9]):
                if '_POC' in filename:
                    f_all, x_list = get_data_list(filename)
                    for s in range(6, 98):
                        f_tra = f_all[:s]

                        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
                        km_list = np.array([[x] for x in f_tra], dtype=float)
                        k_means.fit(np.array(km_list))

                        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
                        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

                        # 根据标签，将x和f存入x_cen, f_cen
                        f_cen = [[] for _ in range(center_)]

                        for i in range(len(list(k_means_labels))):
                            for k_ in range(center_):
                                if int(k_means_labels[i]) == int(k_):
                                    f_cen[k_].append(f_tra[i])
                        mu = [np.mean(f_tra)]
                        for i in range(center_):
                            mu.append(np.mean(f_cen[i]))

                        sigma = [np.std(f_tra)]
                        for i in range(center_):
                            sigma.append(np.std(f_cen[i]))

                        for i in range(len(mu)):
                            mean_list[i].append(mu[i])
                            std_list[i].append(sigma[i])

            mu_ = np.array([np.mean(np.array(mean_list)[i, :]) for i in range(center_ + 1)])
            sigma_ = np.array([np.mean(np.array(std_list)[i, :]) for i in range(center_ + 1)])
            print(f'num:({center_}) - b={bi} : mu = {mu_}   sigma = {sigma_}')
            fig, ax = plt.subplots()
            # 画出点和原点
            ax.plot(mu_, sigma_, 'ro')
            ax.plot(0, 0, 'bo')
            # 画出从原点到第一个点的向量
            for i in range(mu_.shape[0]):
                ax.arrow(0, 0, mu_[i], sigma_[i], head_width=bi / 30., head_length=bi / 30. + 0.5, fc='k', ec='k')
                ax.annotate('v{}'.format(i), xy=(mu_[i], sigma_[i]),
                            xytext=(mu_[i] + 1, sigma_[i] + 1))
            # ax.arrow(0, 0, mu_[0], sigma_[0], head_width=0.8, head_length=0.8, fc='k', ec='k')
            # ax.arrow(0, 0, mu_[1], sigma_[1], head_width=0.8, head_length=0.8, fc='k', ec='k')
            # ax.arrow(0, 0, mu_[2], sigma_[2], head_width=0.8, head_length=0.8, fc='k', ec='k')
            # 设置坐标轴范围
            plt.title(f'b = {bi}')
            # 显示图像
            plt.show()


def adaptive_km():
    path_dir = 'data_save/run_data/kxm/'
    b_list = [100, 50, 10]
    # center_ = 6
    for bi in b_list:
        n_list = []
        a_list = []
        path_file = path_dir + f'data_b{int(bi)}'
        list_csv = list_dir(path_file)
        for fi, filename in enumerate(list_csv[:3]):
            if '_POC' in filename:
                f_all, x_list = get_data_list(filename)
                for s in range(10, 98):
                    f_tra = f_all[:s]
                    al, n = adaptive_kmeans2(f_tra)
                    a_list.append(al)
                    n_list.append(n)
        print(f'b={bi:<4d} : ', end='')
        l0 = 0.
        l1 = 0.
        l2 = 0.
        for n in n_list:
            l0 += 1.
            if n == 1:
                l1 += 1.
            elif n == 2:
                l2 += 1.
        print(f' rate(n=1): {l1/l0:<4.3f}      rate(n=2): {l2/l0:<4.3f} ')


def adaptive_kmeans(data, alpha=0.45, n_max=6):
    std_list = [np.std(data)]
    de_cen = n_max
    im = 0
    for center_ in range(2, n_max + 1):
        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
        km_list = np.array([[x] for x in data], dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

        # f_cen
        f_cen = [[] for _ in range(center_)]

        '''
        b=100  :  rate(n=1): 0.000      rate(n=2): 1.000
        b=50   :  rate(n=1): 0.457      rate(n=2): 0.475
        b=10   :  rate(n=1): 0.659      rate(n=2): 0.326
        
        
        b=100  :  rate(n=1): 0.068      rate(n=2): 0.932 
        b=50   :  rate(n=1): 0.000      rate(n=2): 1.000 
        b=10   :  rate(n=1): 0.864      rate(n=2): 0.068 
        
        
        '''

        for i in range(len(list(k_means_labels))):
            for k_ in range(center_):
                if int(k_means_labels[i]) == int(k_):
                    f_cen[k_].append(data[i])
        temp_std = [np.std(f_cen[i]) for i in range(center_)]
        std_list.append(np.max(temp_std))
        im = std_list[-2] - std_list[-1] if std_list[-2] - std_list[-1] > im else im
        # print(f' c = {center_}    im = {im:<3.2f}      list : {std_list}')
        if (std_list[-2] - std_list[-1]) / std_list[-2] <= alpha:
            de_cen = center_ - 1
            break
        elif std_list[-1] < im:
            de_cen = center_
            break

    if de_cen > 1:
        k_means = KMeans(init='k-means++', n_clusters=de_cen, n_init=10)
        km_list = np.array([[x] for x in data], dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

        f_cen = [[] for _ in range(de_cen)]

        for i in range(len(list(k_means_labels))):
            for k_ in range(de_cen):
                if int(k_means_labels[i]) == int(k_):
                    f_cen[k_].append(i)
        return f_cen, de_cen
    else:
        return list(range(len(data))), de_cen


def adaptive_kmeans2(data, alpha=0.15, n_max=6):
    s1, m1 = np.std(data), np.mean(data)
    std_list = [np.std(data)]
    de_cen = n_max
    im = 0
    a_list = [0.]
    a2_list = [0.5]
    for center_ in range(2, n_max + 1):
        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
        km_list = np.array([[x] for x in data], dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

        # f_cen
        f_cen = [[] for _ in range(center_)]

        '''
        b=100  :  rate(n=1): 0.068      rate(n=2): 0.932 
        b=50   :  rate(n=1): 0.000      rate(n=2): 1.000 
        b=10   :  rate(n=1): 0.864      rate(n=2): 0.068 
        
        0.28
        
        b=100  :  rate(n=1): 0.068      rate(n=2): 0.932 
        b=50   :  rate(n=1): 0.170      rate(n=2): 0.716 
        b=10   :  rate(n=1): 0.932      rate(n=2): 0.068 
        
        '''

        for i in range(len(list(k_means_labels))):
            for k_ in range(center_):
                if int(k_means_labels[i]) == int(k_):
                    f_cen[k_].append(data[i])
        temp_means = [np.mean(f_cen[i]) for i in range(center_)]
        temp_std = [np.std(f_cen[i]) for i in range(center_)]
        ang_list = [angle_between_vectors([si, mi], [s1, m1]) for si, mi in zip(temp_std, temp_means)]

        ang = sum(ang_list)/(center_ * 180)
        # ang = np.max(ang_list) / (180.)
        a_list.append(ang)
        a2_list.append(ang)

        std_list.append(np.max(temp_std))
        im = std_list[-2] - std_list[-1] if std_list[-2] - std_list[-1] > im else im
        if (a_list[-1] - a_list[-2]) / a2_list[-2] < 0.25:
        # if a_list[-1] - a_list[-2] <= alpha:
            de_cen = center_ - 1
            break
    print(f'a1: {["{:< 6.2f}".format(num) for num in a_list]}    '
          f'a2: {["{:< 6.2f}".format(a_list[i+1] - a_list[i]) for i in range(len(a_list) - 1)]}  '
          f'a3: {["{:< 6.2f}".format((a_list[i+1] - a_list[i])/a2_list[i]) for i in range(len(a_list) - 1)]}')
    if de_cen > 1:
        k_means = KMeans(init='k-means++', n_clusters=de_cen, n_init=10)
        km_list = np.array([[x] for x in data], dtype=float)
        k_means.fit(np.array(km_list))

        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(np.array(km_list), k_means_cluster_centers)

        f_cen = [[] for _ in range(de_cen)]

        for i in range(len(list(k_means_labels))):
            for k_ in range(de_cen):
                if int(k_means_labels[i]) == int(k_):
                    f_cen[k_].append(i)
        return f_cen, de_cen
    else:
        return [list(range(len(data)))], de_cen


def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.
    cos_angle = dot_product / norm_product
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def plot_rect():
    '''
    name = 'cir'
    # m1 = [7172, 7786, 6456, 7157, 7866, 7509, 7614, 7597, 7298, 7059, 7647, 7687, 7443, ]
    # s1 = [1002, 760, 636, 1152, 933, 1088, 894, 950, 889, 1017, 859, 851, 901, ]
    # m2 = [11368, 11757, 8575, 11271, 12211, 12114, 11302, 11043, 11359, 11284, 11754, 11449, 11146, ]
    # s2 = [1511, 731, 995, 1398, 458, 656, 851, 1259, 687, 890, 602, 649, 781, ]

    # 直线边界
    name = 'cir'
    # m1 = [7172, 7786, 6456, 7157, 7866, 7509, 7614, 7687, 7504, ]
    # s1 = [1002, 760, 636, 1152, 933, 1088, 894, 851, 554, ]
    # m2 = [11368, 11757, 8575, 11271, 12211, 12114, 11302, 11449, 12342, ]
    # s2 = [1511, 731, 995, 1398, 458, 656, 851, 649, 597, ]
    '''

    # 正弦边界
    name = 'sine'
    m1 = [9304, 5417, 4750, 5662, 6853, 6942, 5237, 5978, 5918, 5853]
    s1 = [345, 744, 1394, 768, 811, 550, 1032, 810, 1074, 845]
    m2 = [14257, 7451, 4673, 7896, 10862, 9710, 5991, 7172, 8769, 7563, ]
    s2 = [588, 2893, 1550, 1798, 1147, 1263, 1780, 907, 1653, 1262]
    mm1 = [5]
    mm2 = [4]

    # 圆形边界
    # name = 'cir'
    # m0 = [5826, 4932, 4932, 4932, 4438, 3542, 4932, 4439, 3819, 4363]
    # s0 = [595, 413, 413, 413, 598, 825, 413, 693, 830, 531]
    # m1 = [9763, 5284, 3916, 4296, 5837, 5686, 3864, 5289, 4663, 4486]
    # s1 = [295, 503, 840, 532, 1119, 1150, 1214, 414, 598, 808]
    # m2 = [14694, 5740, 5359, 5582, 6599, 6952, 4624, 5977, 5829, 6307]
    # s2 = [487, 1283, 1891, 1355, 1752, 2322, 1380, 1629, 1552, 2834]
    # mm0 = [2, 3, 6]
    # mm1 = [4]
    # mm2 = [5]

    # # 方形边界
    # name = 'rect'
    # m0 = [5613, 4657, 4657, 4657, 4232, 3411, 4657, 4044, 3831, 4523]
    # s0 = [376, 260, 260, 260, 276, 259, 260, 620, 598, 893]
    # m1 = [9215, 4847, 3333, 4323, 5310, 4998, 3305, 4947, 4888, 4420]
    # s1 = [249, 279, 783, 726, 524, 791, 1394, 316, 447, 496]
    # m2 = [13814, 5372, 5507, 6867, 6517, 6941, 5976, 6096, 5667, 5990]
    # s2 = [510, 1273, 1494, 1430, 1907, 1690, 1365, 1328, 1212, 1586]
    # mm0 = [2, 3, 6]
    # mm1 = [4]
    # mm2 = [5]

    # 线性边界
    # name = 'line'
    # m1 = [9794, 4859, 7486, 7866, 7172, 7786, 7687, 6909, 6826, 6147]
    # s1 = [217, 469, 574, 933, 1002, 760, 851, 842, 824, 778]
    # m2 = [14271, 7188, 12131, 12211, 11368, 11757, 11449, 8936, 9150, 8955]
    # s2 = [594, 185, 635, 458, 1511, 731, 649, 682, 743, 2443]
    # mm1 = [3]
    # mm2 = [3]


    x = list(range(len(m1)))
    colors = ['orange' for _ in range(len(m1))]
    colors[0] = 'blue'
    colors[1] = 'green'
    # tags = ['Optimal', 'PSO', 'asscl', 'kfdim', 'pxm', "kxm", "sscl", "SVM-r", "SVM-l", "SQLPSO"]
    tags = ['Optimal', 'PSO', 'asscl', 'kfdim', 'pxm', "kxm", "sscl", "SVM-r", "SVM-l", "SQL-PSO"]
    save_path = 'data_save/other_fig/rect_fig'

    if not os.path.exists(save_path):
        os.mkdir(save_path)  # 创建一个目录。只创建一个目录文件huoze
    try:
        fig, ax = plt.subplots()
        plt.bar(x, m0, color=['red' if i in mm0 else c for i, c in enumerate(colors)])

        # 在每个方块上添加标签和标准差
        plt.errorbar(x, m0, yerr=s0, fmt='none', ecolor='r', capsize=2)
        plt.xticks(x, tags)
        for i in range(len(x)):
            plt.text(x[i], m0[i], f"{m0[i]}", ha='center', va='bottom')

        # 添加标题和标签
        ax.set_title(f'DTP (b=10)')
        ax.set_xlabel('Method')
        ax.set_ylabel('Accumulate Fitness')
        plt.savefig(save_path + f'/{name}_b10')

        # 显示图形
        plt.show()
    except Exception as e:
        print(e)

    # 绘制直方图
    fig, ax = plt.subplots()
    # colors = ['blue', 'green', 'orange', 'orange', 'orange', 'red', 'orange', 'orange', 'orange', 'orange']
    plt.bar(x, m1, color=['red' if i in mm1 else c for i, c in enumerate(colors)])


    # 在每个方块上添加标签和标准差
    plt.errorbar(x, m1, yerr=s1, fmt='none', ecolor='r', capsize=2)
    plt.xticks(x, tags)
    for i in range(len(x)):
        plt.text(x[i], m1[i], f"{m1[i]}", ha='center', va='bottom')

    # 添加标题和标签
    ax.set_title(f'DTP (b=50)')
    ax.set_xlabel('Method')
    ax.set_ylabel('Accumulate Fitness')
    plt.savefig(save_path + f'/{name}_b50')

    # 显示图形
    plt.show()

    fig, ax2 = plt.subplots()
    plt.bar(x, m2, color=['red' if i in mm2 else c for i, c in enumerate(colors)])

    # 在每个方块上添加标签和标准差
    plt.errorbar(x, m2, yerr=s2, fmt='none', ecolor='r', capsize=2)
    plt.xticks(x, tags)
    for i in range(len(x)):
        plt.text(x[i], m2[i], f"{m2[i]}", ha='center', va='bottom')

    # 添加标题和标签
    ax2.set_title(f'DTP (b=100)')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accumulate Fitness')
    plt.savefig(save_path + f'/{name}_b100')

    # 显示图形
    plt.show()


def plot_rect2():
    '''
    name = 'cir'
    # m1 = [7172, 7786, 6456, 7157, 7866, 7509, 7614, 7597, 7298, 7059, 7647, 7687, 7443, ]
    # s1 = [1002, 760, 636, 1152, 933, 1088, 894, 950, 889, 1017, 859, 851, 901, ]
    # m2 = [11368, 11757, 8575, 11271, 12211, 12114, 11302, 11043, 11359, 11284, 11754, 11449, 11146, ]
    # s2 = [1511, 731, 995, 1398, 458, 656, 851, 1259, 687, 890, 602, 649, 781, ]

    # 直线边界
    name = 'cir'
    # m1 = [7172, 7786, 6456, 7157, 7866, 7509, 7614, 7687, 7504, ]
    # s1 = [1002, 760, 636, 1152, 933, 1088, 894, 851, 554, ]
    # m2 = [11368, 11757, 8575, 11271, 12211, 12114, 11302, 11449, 12342, ]
    # s2 = [1511, 731, 995, 1398, 458, 656, 851, 649, 597, ]
    '''

    # 正弦边界


    # 圆形边界
    # name = 'cir'
    # m0 = [5826, 4932, 4439, 3819, 4363]
    # s0 = [595, 413, 693, 830, 531]
    # m1 = [9763, 5284, 5289, 4663, 4486]
    # s1 = [295, 503, 414, 598, 808]
    # m2 = [14694, 5740, 5977, 5829, 6307]
    # s2 = [487, 1283, 1629, 1552, 2834]
    # mm0 = [2]
    # mm1 = [2]
    # mm2 = [4]

    # 方形边界
    # name = 'rect'
    # m0 = [5613, 4657, 4044, 3831, 4523]
    # s0 = [376, 260, 620, 598, 893]
    # m1 = [9215, 4847, 4947, 4888, 4420]
    # s1 = [249, 279, 316, 447, 496]
    # m2 = [13814, 5372, 6096, 5667, 5990]
    # s2 = [510, 1273, 1328, 1212, 1586]
    # mm0 = [4]
    # mm1 = [2]
    # mm2 = [2]

    # 线性边界
    # name = 'line'
    # m1 = [9794, 4859, 6909, 6826, 6147]
    # s1 = [217, 469, 842, 824, 778]
    # m2 = [14271, 7188, 8936, 9150, 8955]
    # s2 = [594, 185, 682, 743, 2443]
    # mm1 = [2]
    # mm2 = [3]


    x = list(range(len(m1)))
    colors = ['orange' for _ in range(len(m1))]
    colors[0] = 'blue'
    colors[1] = 'green'
    # tags = ['Optimal', 'PSO', 'asscl', 'kfdim', 'pxm', "kxm", "sscl", "SVM-r", "SVM-l", "SQLPSO"]
    tags = ['Optimal', 'PSO', "SVM-r", "SVM-l", "SQL-PSO"]
    save_path = 'data_save/other_fig/rect_fig2'

    if not os.path.exists(save_path):
        os.mkdir(save_path)  # 创建一个目录。只创建一个目录文件huoze
    try:
        fig, ax = plt.subplots()
        plt.bar(x, m0, color=['red' if i in mm0 else c for i, c in enumerate(colors)])

        # 在每个方块上添加标签和标准差
        plt.errorbar(x, m0, yerr=s0, fmt='none', ecolor='r', capsize=2)
        plt.xticks(x, tags)
        for i in range(len(x)):
            plt.text(x[i], m0[i], f"{m0[i]}", ha='center', va='bottom')

        # 添加标题和标签
        ax.set_title(f'DTP (b=10)')
        ax.set_xlabel('Method')
        ax.set_ylabel('Accumulate Fitness')
        plt.savefig(save_path + f'/{name}_b10')

        # 显示图形
        plt.show()
    except Exception as e:
        print(e)

    # 绘制直方图
    fig, ax = plt.subplots()
    # colors = ['blue', 'green', 'orange', 'orange', 'orange', 'red', 'orange', 'orange', 'orange', 'orange']
    plt.bar(x, m1, color=['red' if i in mm1 else c for i, c in enumerate(colors)])


    # 在每个方块上添加标签和标准差
    plt.errorbar(x, m1, yerr=s1, fmt='none', ecolor='r', capsize=2)
    plt.xticks(x, tags)
    for i in range(len(x)):
        plt.text(x[i], m1[i], f"{m1[i]}", ha='center', va='bottom')

    # 添加标题和标签
    ax.set_title(f'DTP (b=50)')
    ax.set_xlabel('Method')
    ax.set_ylabel('Accumulate Fitness')
    plt.savefig(save_path + f'/{name}_b50')

    # 显示图形
    plt.show()

    fig, ax2 = plt.subplots()
    plt.bar(x, m2, color=['red' if i in mm2 else c for i, c in enumerate(colors)])

    # 在每个方块上添加标签和标准差
    plt.errorbar(x, m2, yerr=s2, fmt='none', ecolor='r', capsize=2)
    plt.xticks(x, tags)
    for i in range(len(x)):
        plt.text(x[i], m2[i], f"{m2[i]}", ha='center', va='bottom')

    # 添加标题和标签
    ax2.set_title(f'DTP (b=100)')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Accumulate Fitness')
    plt.savefig(save_path + f'/{name}_b100')

    # 显示图形
    plt.show()


def print_sigma():
    # num: (2) - b = 100:
    mu11 = [101.59220279, - 53.97134134, 138.72799601]
    sigma11 = [71.60321864, 11.00421172, 15.77434432]
    # num: (2) - b = 50:
    mu21 = [71.67579909, - 8.8287993,  91.12205169]
    sigma21 = [38.00338575, 22.30509523, 14.44396678]
    # num: (2) - b = 10:
    mu31 = [42.99669755, 22.76292578, 54.63598225]
    sigma31 = [17.85152545, 11.81050217,  8.38082208]
    # num: (3) - b = 100:
    mu12 = [101.59220279, - 54.50614511, 111.89666248, 149.82735449]
    sigma12 = [71.60321864, 10.78765654,  8.96214537,  6.81663197]
    # num: (3) - b = 50:
    mu22 = [71.67579909, - 22.80399805,  53.48122539,  97.38645975]
    sigma22 = [38.00338575, 15.1816728,  11.84759498,  8.35359184]
    # num: (3) - b = 10:
    mu32 = [42.99669755,  8.40765115, 36.02636126, 58.94258578]
    sigma32 = [17.85152545,  9.15485991,  5.83600078,  6.18454655]
    # num: (4) - b = 100:
    mu13 = [101.59220279, - 62.43196735,  53.89421669, 128.01527415, 152.07783332]
    sigma13 = [71.60321864,  8.76022946,  6.75766076,  6.20102008,  5.33206354]
    # num: (4) - b = 50:
    mu23 = [71.67579909, - 31.24039504,  32.4949191,   73.47043643, 100.72971303]
    sigm23a = [38.00338575, 11.53542117,  8.07598396,  6.99435147,  6.35082185]
    # num: (4) - b = 10:
    mu = [42.99669755,  3.13351011, 29.97804945, 47.28690391, 63.14447946]
    sigma = [17.85152545,  7.49347021,  4.64111555,  4.22833039,  4.29649123]
    # num: (5) - b = 100:
    mu = [101.59220279, - 64.2175662,   36.31475517, 108.27504733, 135.68202113, 154.20684422]
    sigma = [71.60321864,  7.60500819,  5.99183265,  4.53954904,  3.90016568,  4.1341997]
    # num: (5) - b = 50:
    mu = [71.67579909, - 35.24684349,  16.35652114,  57.13276568,  85.54425138, 103.80531233]
    sigma = [38.00338575,  9.48442283,  5.62094849,  5.67491822,  4.55583667,  4.90959585]
    # num: (5) - b = 10:
    mu = [42.99669755, - 2.14446465, 22.34533021, 36.79073278, 51.23282965, 64.77163775]
    sigma = [17.85152545,  4.71467682,  3.73987564,  3.51088656,  3.26579949,  3.58135603]

    b10_sigma = [[17.85152545, 11.81050217,  8.38082208],
                 [17.85152545,  9.15485991,  5.83600078,  6.18454655],
                 [17.85152545,  7.49347021,  4.64111555,  4.22833039,  4.29649123],
                 [17.85152545,  4.71467682,  3.73987564,  3.51088656,  3.26579949,  3.58135603]]

    b50_sigma = [[38.00338575, 22.30509523, 14.44396678],
                 [38.00338575, 15.1816728,  11.84759498,  8.35359184],
                 [38.00338575, 11.53542117,  8.07598396,  6.99435147,  6.35082185],
                 [38.00338575,  9.48442283,  5.62094849,  5.67491822,  4.55583667,  4.90959585]]

    b100_sigma = [[71.60321864, 11.00421172, 15.77434432],
                  [71.60321864, 10.78765654,  8.96214537,  6.81663197],
                  [71.60321864,  8.76022946,  6.75766076,  6.20102008,  5.33206354],
                  [71.60321864,  7.60500819,  5.99183265,  4.53954904,  3.90016568,  4.1341997]]

    b10_m = [b10_sigma[0][0]]
    for l in b10_sigma:
        b10_m.append(np.mean(l[1:]))

    b50_m = [b50_sigma[0][0]]
    for l in b50_sigma:
        b50_m.append(np.mean(l[1:]))

    b100_m = [b100_sigma[0][0]]
    for l in b100_sigma:
        b100_m.append(np.mean(l[1:]))

    print(f'b=10     {b10_m}\n'
          f'b=50     {b50_m}\n'
          f'b=100    {b100_m}\n')

    b10_m2 = [b10_sigma[0][0]]
    for i in range(len(b10_m)-1):
        b10_m2.append(b10_m[i] - b10_m[i+1])

    b50_m2 = [b50_sigma[0][0]]
    for i in range(len(b10_m)-1):
        b50_m2.append(b50_m[i] - b50_m[i+1])

    b100_m2 = [b100_sigma[0][0]]
    for i in range(len(b10_m)-1):
        b100_m2.append(b100_m[i] - b100_m[i+1])

    print(f'b=10     {b10_m2}\n'
          f'b=50     {b50_m2}\n'
          f'b=100    {b100_m2}\n')

    b10_m3 = []
    for i in range(len(b10_m)-1):
        b10_m3.append(b10_m2[i+1] / b10_m[i])

    b50_m3 = []
    for i in range(len(b10_m)-1):
        b50_m3.append(b50_m2[i+1] / b50_m[i])

    b100_m3 = []
    for i in range(len(b100_m)-1):
        b100_m3.append(b100_m2[i+1] / b100_m[i])

    print(f'b=10     {b10_m3}\n'
          f'b=50     {b50_m3}\n'
          f'b=100    {b100_m3}\n')

def plot_s():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest

    # 设置随机种子，以确保结果可复现
    np.random.seed(42)

    # 随机生成正态分布的二维数据
    num_samples = 1000
    mean = [0, 0]  # 均值
    cov = [[1, 0.5], [0.5, 1]]  # 协方差矩阵
    X = np.random.multivariate_normal(mean, cov, num_samples)

    # 创建孤立森林模型
    clf = IsolationForest(n_estimators=100, contamination='auto', max_samples=256, random_state=42)

    # 拟合数据
    clf.fit(X)

    # 预测样本的标签，1表示正常个体，-1表示异常个体
    labels = clf.predict(X)
    print(X.shape)
    print(f'{len(labels)}   {len([i for i in labels if i == 1])}   {len([i for i in labels if i == -1])}')

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='coolwarm')
    plt.colorbar(label='Label (1: Normal, -1: Anomaly)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Isolation Forest - Anomaly Detection')
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    # filename = 'data/data_b100/b=100_MPB_23224335_POC.csv'
    # filename = 'data/data_b50/b=50_MPB_20222216_POC.csv'
    path_dir = 'KD4/data/'
    b_list = [10]

    for bi in b_list:
        path_file = path_dir + f'data_b{int(bi)}'
        list_csv = list_dir(path_file)
        r1_list = []
        r2_list = []
        plot_fit(list_csv)

        # for fi, filename in enumerate(list_csv[0:2]):
        #     if '_POC' in filename:
        #         print('============================================================')
        #         print(f'           b={bi}       fi={fi}        ')
        #         plot_fit(filename)
