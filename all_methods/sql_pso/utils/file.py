# -*- coding: UTF-8 -*-
# @Date    :2023/11/7 16:40
# @Author  :高猛
# @Project :code_cl 
# @File    :file.py
# @IDE     :PyCharm

import os


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




