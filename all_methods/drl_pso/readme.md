

修改代码，使得在打印信息的时候，打印为像是 2h 35m 26s这样的形式，也就是说，每个部分都保留为整数而不要出现0.5h这样的表示而应该是0h 30m 0s，根据采样id和已经过去的时间（设置程序开始时间为一个全局变量，从而得到任务执行时间），再根据args的参数计算所有的任务数量，从而达到打印已经过去的时间，预估还剩下多少时间的目的,，返回完整代码给我：  print(f'===========================================================\n'
          f'sample_id: {sample_id}    '
          f't_our: {round(t_all, 3)}   ')


完整原始代码：
# -*- coding: UTF-8 -*-
# @Date    :2024/1/15 21:24
# @Author  :高猛
# @Project :code_cl_test20240109
# @File    :main_test.py
# @IDE     :PyCharm

from components.Buffer import file_init
from configs import set_param
from components.Demo import *
import os
from datetime import datetime
import time
import multiprocessing as mp


def run_data():
    t_start = time.time()
    file_init()
    args = set_param()
    name_time = str(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    num_cores = int(mp.cpu_count())
    use_cores = int(num_cores - 5)
    print(f"This computer has {num_cores} cores, we use {use_cores} cores")
    pool = mp.Pool(use_cores)

    sampler_list = []
    id_dict = {}
    sample_id = 0
    data_save_path = f'data_save/run_data/{name_time}_{args.mode}'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    for Lambda_ in args.Lambda_list:
        save_path1 = f'{data_save_path}/lambda={Lambda_}'
        if not os.path.exists(save_path1):
            os.mkdir(save_path1)
        for peak_num in args.peak_num_list:
            for bt_change in args.bt_change_list:
                save_path2 = f'{save_path1}/{bt_change[:3]}_{peak_num}'
                if not os.path.exists(save_path2):
                    os.mkdir(save_path2)

                for bt_type in args.bt_type_list:
                    for b in args.b_list:
                        rand_seed = int('123' + str(b))

                        args.Lambda_ = Lambda_
                        args.peak_num = peak_num
                        args.bt_change = bt_change
                        args.rand_seed = rand_seed
                        args.sample_id = sample_id
                        args.bt_type = bt_type
                        args.time_fac = b

                        with open(save_path2 + '/parameters.txt', 'w') as file:
                            for arg, value in vars(args).items():
                                file.write(f'{arg}: {value}\n')

                        save_path = f'{save_path2}/{bt_type}_b{int(b)}'
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)

                        id_dict[sample_id] = f'{bt_type[0]}_{b}'

                        if args.using_multiprocessing:
                            for i in range(args.sample_num):
                                sampler_list.append(
                                    pool.apply_async(run_one, args=(sample_id, save_path, copy.deepcopy(args))))
                                sample_id += 1
                        else:
                            for i in range(args.sample_num):
                                run_one(sample_id, save_path, copy.deepcopy(args))
                                sample_id += 1

    # sample
    if args.using_multiprocessing:
        for one_sampler in sampler_list:
            one_sampler.get()

    t_all = time.time() - t_start
    print(f'\n=================================================================================\n'
          f'(using_multiprocessing is {args.using_multiprocessing})  All time is '
          f'{round(t_all/3600., 2)} hours / {round(t_all/60., 2)} minutes / {round(t_all, 2)} seconds\n'
          f'=================================================================================\n')

    return f'{name_time}_{args.mode}'


def run_one(sample_id, save_path, args):
    # initialize
    f_n = f'{save_path}/b={int(args.time_fac)}_MPB_{sample_id}_'
    fig_path = f'data_save/other_fig/{args.mode}'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    save_path1 = f'data_save/other_fig/{args.mode}/step_fig'
    if not os.path.exists(save_path1):
        os.mkdir(save_path1)
    args.fig_filename = save_path1
    save_path2 = f'data_save/other_fig/{args.mode}/some_data'
    if not os.path.exists(save_path2):
        os.mkdir(save_path2)
    args.test_filename = save_path2

    # start run one loop
    t_start = time.time()

    args.filename = f_n + f'_{args.mode}.csv'
    CL_PSO(args)
    # PPO_PSO(args)

    t_end = time.time()
    t_all = float(t_end - t_start)

    # print costed time
    print(f'===========================================================\n'
          f'sample_id: {sample_id}    '
          f't_our: {round(t_all, 3)}   ')


if __name__ == '__main__':
    mode = run_data()
    print(mode)









