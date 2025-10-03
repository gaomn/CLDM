# -*- coding: UTF-8 -*-
# @Date    :2023/11/7 16:40
# @Author  :高猛
# @Project :buffer
# @File    :file.py
# @IDE     :PyCharm

import os
import copy
import numpy as np

class buffer:
    def __init__(self):
        self.MAX_SIZE = 1e7
        self.memory = dict()

    def store_data(self, t, data_dict):
        if str(t) not in self.memory.keys():
            self.memory[str(t)] = dict()
        for k, v in data_dict.items():
            self.memory[str(t)][k] = v

    def get_all_data(self):
        return self.memory

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

    def get_data(self, t):
        x_list = []
        f_this = []
        f_next = []
        for t_ in range(t):
            x_list.append(list(self.memory[str(t_)]["x_best"]))
            f_this.append(self.memory[str(t_)]["f_this"])
            f_next.append(self.memory[str(t_)]["f_next"])
        return x_list, f_this, f_next


def file_init():
    data_save_path = 'data_save'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    path_list = [f'{data_save_path}/fig_data',
                 f'{data_save_path}/other_data',
                 f'{data_save_path}/other_fig',
                 f'{data_save_path}/run_data']
    for p in path_list:
        if not os.path.exists(p):
            os.mkdir(p)






