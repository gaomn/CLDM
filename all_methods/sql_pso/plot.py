# -*- coding: UTF-8 -*-
# @Date    :2023/11/10 10:08
# @Author  :高猛
# @Project :code_cl 
# @File    :plot.py.py
# @IDE     :PyCharm

from utils.draw import *
from utils.file import file_init
from dtp_base.configs import set_param
import os


def draw_fit(filename='KD2'):
    args = set_param()
    for bt_type in args.bt_type_list:
        for b in args.b_list:
            save_path = f'data_save/fig_data/{filename}'
            read_path = f'data_save/run_data/{filename}/{bt_type}_b{int(b)}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            data_dict = draw_from_file(f=read_path, b=int(b), x=f'{bt_type}_b={int(b)}',
                                       save_path=f'{save_path}/{bt_type}_b={int(b)}.png',
                                       title=filename, m=args.max_step)
            with open('plot.csv', 'a', newline='') as f:
                fields = ['title', 'bt-type', 'time_fac', 'Our method', 'Optimal', 'PSO only']
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(data_dict)


if __name__ == '__main__':

    mode = '2023-11-09_222614_asscl'
    draw_fit(mode)
