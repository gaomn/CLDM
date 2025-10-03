# -*- coding: UTF-8 -*-
# @Date    :2024/4/5 16:09
# @Author  :高猛
# @Project :cl-pso 
# @File    :Detect.py
# @IDE     :PyCharm

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


def mic(x_data, y_data, ):
    x = np.array(copy.copy(x_data))
    y = np.array(copy.copy(y_data))

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)

    mine = MINE()
    mine.compute_score(x, y)
    return mine.mic()


def pearsonr_correlation(x, y):
    corr, p_value = pearsonr(x, y)
    return corr


class detection:
    def __init__(self, args):
        self.args = args
        self.x_dim = args.x_dim
        self.Index = None

    def if_strong_signal(self, x, f):
        if self.cd(x, f) <= self.args.cd_threshold:
            return False
        return True

    def train(self, data_x_, data_y_, t):
        if t > 3:
            data_x = np.array(data_x_)
            data_y = np.array(data_y_)

            center1_ = 2
            k_means1 = KMeans(init='k-means++', n_clusters=center1_, n_init=10)
            km_list1 = np.array([[f] for f in data_y], dtype=float)
            k_means1.fit(np.array(km_list1))

            k_means_cluster_centers1 = np.sort(k_means1.cluster_centers_, axis=0)
            k_means_labels1 = pairwise_distances_argmin(np.array(km_list1), k_means_cluster_centers1)
            labels = np.array(k_means_labels1)

            mic_list = []
            for i in range(self.x_dim):
                x_list = copy.copy(data_x[:, i])

                mic_list.append(self.cal_coe(x_list, labels))

            self.Index = [np.argmax(mic_list)]
            for i, mic_value in enumerate(mic_list):
                if mic_value > np.mean(mic_list) + self.args.c * np.std(mic_list) and (i not in self.Index):
                    self.Index.append(i)

    def cal_coe(self, x_data, y_data):
        x = np.array(copy.copy(x_data))
        y = np.array(copy.copy(y_data))

        x = (x - np.mean(x)) / (np.std(x) + 1e-5)
        y = (y - np.mean(y)) / (np.std(y) + 1e-5)

        mine = MINE()
        mine.compute_score(x, y)
        return mine.mic()

    def get_kr_list(self):
        return self.Index

    def cd(self, x_tra, f_next):
        f_next = [i / (np.max(f_next)) for i in f_next]
        x_dec = [[xid / self.args.x_bound for xii, xid in enumerate(xi) if xii in self.Index] for xi in x_tra]

        n_dec = self.args.k_means_center if len(f_next) >= 12 else len(f_next) - 2
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

        dt = 0
        n_dec = len(f_n)
        for i in range(n_dec):
            for j in range(n_dec):
                if i != j:
                    dt += (np.abs(np.mean(f_n[i]) - np.mean(f_n[j])))

        dt = (np.array(dt) / float(n_dec * (n_dec - 1))) if n_dec >= 2 else 0.
        return dt
