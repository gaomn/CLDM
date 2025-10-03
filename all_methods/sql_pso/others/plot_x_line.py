# -*- coding: UTF-8 -*-
# @Date    :2023/4/20 22:05
# @Author  :高猛
# @Project :DTPs 
# @File    :plot_x_line.py
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
