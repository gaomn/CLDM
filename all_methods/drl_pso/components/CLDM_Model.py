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
from components.RBFN import RBFN
from components.Buffer import buffer
from sklearn.metrics.pairwise import pairwise_distances_argmin
from components.Detect import detection


class CLDM_Model:
    def __init__(self, args):
        self.args = args
        self.buffer = buffer()
        self.detection = detection(args)

    def choose_individual(self, t, pso_dict):
        # save_data
        self._train_detection(t, pso_dict)

        # when t <= 5, use PSO
        if t < 5:
            x_best = pso_dict['best_x']
        else:
            x_best = self._get_clp_solution(t, pso_dict)

        self.buffer.store_data(t, {'x_best': copy.copy(x_best)})

        return copy.copy(x_best)

    def _get_clp_solution(self, t, pso_dict):
        # 获取分类数据
        x_tra, f_this, f_next = self.buffer.get_data(t - 1)

        # 时链检测处理
        if not self.detection.if_strong_signal(x_tra, f_next):
            return pso_dict['best_x']

        # make batch data
        X_data, y_data = self._make_batch(x_tra, f_this, f_next)

        # 训练RBFN模型
        rbf_centers = 32 if y_data.shape[0] > 32 else y_data.shape[0] - 1
        model = RBFN(rbf_centers)
        model.fit(X_data, y_data)

        # 基于聚类的快速选择
        x_best = self._cqd(t, pso_dict, model)

        return x_best

    def _cqd(self, t, pso_dict, model):
        x_list = pso_dict['x_list']
        f_list = pso_dict['fit_list']
        x_tra, f_this, f_next = self.buffer.get_data(t - 1)

        f_max = np.max(f_next) if f_next else 1  # 防止空列表或除以零

        # 利用k-means, 对f值进行二分类
        k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
        km_list = np.array([[x] for x in f_list], dtype=float)
        k_means.fit(km_list)

        # 使用KMeans的labels_属性
        k_means_labels = k_means.labels_
        centers = k_means.cluster_centers_.ravel()

        # 根据标签，将x和f存入x_cen, f_cen
        i_cen = [[i for i, label in enumerate(k_means_labels) if label == k_] for k_ in range(2)]
        f_cen = [[f_list[i] for i in cen] for cen in i_cen]

        # 过滤数据
        f_this_max = np.max(f_list)
        delta_f = np.abs(np.mean(f_cen[0]) - np.mean(f_cen[1]))
        filtered_indices = [i for i, f_temp in enumerate(f_list) if f_temp < f_this_max - delta_f]
        x_list = [x_list[i] for i in filtered_indices]
        f_list = [f_list[i] for i in filtered_indices]

        # 使用k_means方法减少排序数据
        kr_list = self.detection.get_kr_list()
        # print(f'kr_list in cqd: {kr_list}')

        # 一步构建 km_list
        km_list = [[f_list[i] / f_max] + [x_list[i][kr] / self.args.x_bound for kr in kr_list] for i in
                   range(len(x_list))]

        # 计算聚类
        center_ = self.args.k_means_center
        k_means = KMeans(init='k-means++', n_clusters=center_, n_init=10)
        k_means.fit(np.array(km_list))
        k_means_labels = k_means.labels_

        # 创建集群存储列表
        pop_ = [[] for _ in range(center_)]
        fit_ = [[] for _ in range(center_)]

        # 分配到集群
        for ind_, label in enumerate(k_means_labels):
            pop_[label].append(x_list[ind_])
            fit_[label].append(f_list[ind_])

        # 选择最佳或随机选择
        x_choose = []
        f_choose = []
        for i in range(center_):
            if len(fit_[i]) <= 3:
                x_choose.extend(pop_[i])
                f_choose.extend(fit_[i])
            else:
                a_index = int(np.argmax(fit_[i]))
                selected_indices = [a_index] + list(
                    np.random.permutation([x for x in range(len(fit_[i])) if x != a_index])[:2])
                x_choose.extend([pop_[i][j] for j in selected_indices])
                f_choose.extend([fit_[i][j] for j in selected_indices])

        x_sort, f_sort = x_choose, f_choose

        model_input = [[x[ki] / self.args.x_bound for ki in kr_list] + [f / f_max] for x, f in zip(x_sort, f_sort)]
        d_lst = np.array(model_input)
        length = len(d_lst)
        score = np.zeros(length)

        # 使用向量化计算替代双层循环
        for i in range(length):
            for j in range(i + 1, length):  # 只计算一半矩阵，因为i和j的比较是对称的
                i_j_concat = np.concatenate([d_lst[i], d_lst[j]])
                j_i_concat = np.concatenate([d_lst[j], d_lst[i]])
                # np扩充一个维度

                # i_j_concat = np.expand_dims(i_j_concat, axis=0)
                # j_i_concat = np.expand_dims(j_i_concat, axis=0)
                # print(f'i_j_concat: {i_j_concat.shape}')
                value_ij = model.predict(np.array([i_j_concat]))[0]
                value_ji = model.predict(np.array([j_i_concat]))[0]

                if value_ij[0] > value_ij[1]:
                    score[i] += 1
                else:
                    score[j] += 1

                if value_ji[0] > value_ji[1]:
                    score[j] += 1
                else:
                    score[i] += 1

        # 确定最佳指标
        x_best_index = int(np.argmax(score))
        return x_sort[x_best_index]

    def _make_batch(self, x_tra, f_this, f_next):
        # 归一化处理
        f_max = np.max(f_next)
        x_norm = np.array(x_tra) / self.args.x_bound
        f_this_norm = np.array(f_this) / f_max
        f_sum_norm = f_this_norm + np.array(f_next) / f_max

        # 获取kr列表
        kr = self.detection.get_kr_list()
        # print(f'kr_list in _make_batch: {kr}')

        # 准备数据集
        train_in = []
        train_out = []

        # 创建输入输出数据对
        for i in range(len(x_norm)):
            for j in range(len(x_norm)):
                if i != j:
                    x_pair = np.concatenate([[x_norm[i][k] for k in kr], [f_this_norm[i]],
                                             [x_norm[j][k] for k in kr], [f_this_norm[j]]])
                    v_diff = f_sum_norm[i] - f_sum_norm[j]
                    train_in.append(x_pair)
                    train_out.append([v_diff, -v_diff])

        return np.array(train_in), np.array(train_out)

    def _train_detection(self, t, pso_dict):
        # save fit
        data_temp = {'f_best': copy.copy(pso_dict['best_x']),
                     'x_pso': copy.copy(pso_dict['best_v'])}

        self.buffer.store_data(t, data_temp)
        if t > 0:
            self.buffer.store_data(t - 1, {'f_next': copy.copy(pso_dict['best_v'])})

        x_tra, f_this, f_next = self.buffer.get_data(t - 1)
        self.detection.train(np.array(x_tra), np.array(f_next), t - 1)

    def save_fit(self, t, f):
        self.buffer.store_data(t, {'f_this': copy.copy(f)})
