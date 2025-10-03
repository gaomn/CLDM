# -*- coding: UTF-8 -*-
# @Date    :2023/4/18 22:22
# @Author  :高猛
# @Project :DTPs 
# @File    :test_all.py
# @IDE     :PyCharm
from utils.draw import *
from dtp_base.configs import set_param
import os
import datetime
import time
import multiprocessing as mp
# from others.test_norm_distance import norm_distance
from others.test_k import norm_distance, plot_rect, print_sigma, adaptive_km, plot_s, plot_rect2
from others.test_cor import test_cor

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norm_product
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


if __name__ == '__main__':
    # v1 = [1, 0.1]
    # v2 = [-1, 0.1]
    # print(angle_between_vectors(v1, v2))
    # r2 = (50 / np.pi)
    # print(r2)
    print(np.random.random(1)[0])

    # test_cor('data_save/run_data/pxm_26170519/')
    # plot_s()
    # plot_rect2()
    # print_sigma()
    # adaptive_km()
    # norm_distance()
    # a = np.zeros((10, ))
    # b = [1, 5, 3]
    # print(a[b])


    # # 生成两个正态分布数据
    # X1 = np.random.normal(loc=3, scale=1, size=(100, 2))
    # X2 = np.random.normal(loc=-3, scale=1, size=(100, 2))
    # X = np.concatenate((X1, X2))
    #
    # # 使用DBSCAN聚类
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # dbscan.fit(X)
    #
    # # 获取聚类结果
    # labels = dbscan.labels_
    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    #
    # # 绘制散点图
    # for i in range(n_clusters):
    #     plt.scatter(X[labels == i, 0], X[labels == i, 1], label='Cluster %d' % i)
    # plt.scatter(X[labels == -1, 0], X[labels == -1, 1], label='Noise')
    # plt.legend()
    # plt.show()





    # # 创建包含点坐标的NumPy数组
    # points = np.array([[1, 2], [3, 4], [5, 6]])
    #
    # # 创建画布
    # fig, ax = plt.subplots()
    #
    # # 画出点和原点
    # ax.plot(points[:, 0], points[:, 1], 'ro')
    # ax.plot(0, 0, 'bo')
    #
    # # 画出从原点到每个点的向量，并添加文字标注
    # for i in range(points.shape[0]):
    #     ax.arrow(0, 0, points[i, 0], points[i, 1], head_width=0.2, head_length=0.3, fc='k', ec='k')
    #     ax.annotate('v{}'.format(i + 1), xy=(points[i, 0], points[i, 1]),
    #                 xytext=(points[i, 0] + 0.2, points[i, 1] + 0.2))
    #
    # # 设置坐标轴范围
    # ax.set_xlim([-1, 6])
    # ax.set_ylim([-1, 7])
    #
    # # 显示图像
    # plt.show()




# print([1, 0, i for i in range(5)])
import torch

# from model_test.rbfn import RBFN
#
# X = np.array([[i, i+1] for i in range(50)])
# y = np.array([[xi[0], -xi[1]] for xi in X])
#
# print(X)
# print(y)
#
# rbf = RBFN(16)
#
# rbf.fit(X, y)
#
# y_pre = rbf.predict(X)
# print(y_pre)
#
# plt.figure()
# plt.plot([i[1] for i in y], 'r*-')
# plt.plot([i[1] for i in y_pre], 'b--')
# plt.xlabel('x', fontsize=18)
# plt.ylabel('y', fontsize=12)
# plt.title(f'fig', fontsize=18)
# # plt.legend(loc="best")
# plt.show()
#
# a = [1, 2, 3]
# b = [7, 8, 9]
# c = np.hstack((a, b))
#
# print('\033[32m',  f'{a}', '\033[0m')