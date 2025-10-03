# -*- coding: UTF-8 -*-
# @Date    :2023/4/20 21:54
# @Author  :高猛
# @Project :DTPs 
# @File    :test_norm_distance.py
# @IDE     :PyCharm

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
    return round(np.sqrt((mu1 - mu2)**2 + (s1 - s2)**2), 2)


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
    center_ = 2
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

                    d = [dis(mu[2], mu[1], sigma[2], sigma[1]),
                         dis(mu[0], mu[1], sigma[0], sigma[1]),
                         dis(mu[0], mu[2], sigma[0], sigma[2])]
                    dmu = [np.abs(mu[2] - mu[1]), np.abs(mu[0] - mu[1]), np.abs(mu[0] - mu[2])]
                    dsi = [np.abs(sigma[2] - sigma[1]), np.abs(sigma[0] - sigma[1]), np.abs(sigma[0] - sigma[2])]
                    rd1 = round(sum(d) / (np.sqrt(mu[0] ** 2 + sigma[0] ** 2)), 2)
                    rd2 = round(sum(d) / (np.sqrt(mu[0] ** 2 + sigma[0] ** 2) +
                                          np.sqrt(mu[1] ** 2 + sigma[1] ** 2) + np.sqrt(mu[2] ** 2 + sigma[2] ** 2)), 2)
                    rd1_list.append(rd1)
                    rd2_list.append(rd2)

        print(f'\nb={bi}: Rd1={round(np.mean(rd1_list), 2)}~{round(np.std(rd1_list), 2)}, '
              f'{round(np.min(rd1_list), 2)} -> {round(np.max(rd1_list), 2)}')
        print(f'\nb={bi}: Rd2={round(np.mean(rd2_list), 2)}~{round(np.std(rd2_list), 2)}, '
              f'{round(np.min(rd2_list), 2)} -> {round(np.max(rd2_list), 2)}')


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
















