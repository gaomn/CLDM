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

# Global variables for time tracking
PROGRAM_START_TIME = time.time()
TOTAL_TASKS = 0

def format_time(seconds):
    """Convert seconds to a string in the format Xh Ym Zs"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"


def run_data():
    global TOTAL_TASKS
    t_start = time.time()
    file_init()
    args = set_param()
    TOTAL_TASKS = int(args.sample_num * len(args.Lambda_list) * len(args.peak_num_list) * len(args.bt_change_list) * len(args.bt_type_list) * len(args.b_list))
    name_time = str(datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    num_cores = int(mp.cpu_count())
    use_cores = int(max(1, num_cores/2))
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

    # Calculate total number of tasks for time estimation
    # TOTAL_TASKS = sample_id

    # sample
    if args.using_multiprocessing:
        for one_sampler in sampler_list:
            one_sampler.get()

    t_all = time.time() - t_start
    print(f'\n=================================================================================\n'
          f'(using_multiprocessing is {args.using_multiprocessing})  All time is '
          f'{format_time(t_all)}\n'
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
    # only_PSO(args)

    t_end = time.time()
    t_all = float(t_end - t_start)

    # Calculate elapsed and estimated time
    elapsed_time = time.time() - PROGRAM_START_TIME
    TOTAL_TASKS = int(args.sample_num * len(args.Lambda_list) * len(args.peak_num_list) * len(args.bt_change_list) * len(args.bt_type_list) * len(args.b_list))
    # Only estimate remaining time if we have total tasks calculated and at least one task completed
    if TOTAL_TASKS > 0 and sample_id > 0:
        progress = (sample_id + 1) / TOTAL_TASKS
        remaining_time = (elapsed_time / progress) - elapsed_time if progress > 0 else 0
    else:
        remaining_time = 0
        
    # print costed time with formatted times
    print(f'===========================================================\n'
          f'sample_id: {sample_id}    '
          f't_our: {format_time(t_all)}    '
          f'elapsed: {format_time(elapsed_time)}    '
          f'remaining: {format_time(remaining_time)}')


if __name__ == '__main__':
    mode = run_data()
    print(mode)