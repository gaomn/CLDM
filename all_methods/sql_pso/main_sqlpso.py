import shutil

from utils.draw import *
from utils.file import file_init
from dtp_base.configs import set_param
from dtp_base.demo import Our_method, Optimal, PSO_only, SQL_, SQL_0
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
    use_cores = int(num_cores - 1)
    print(f"本地计算机有 {num_cores} 核心, 本项目使用 {use_cores} 核心")
    pool = mp.Pool(use_cores)

    sampler_list = []
    sample_id = 0
    for bt_type in args.bt_type_list:
        for b in args.b_list:

            save_path = 'data_save/run_data'
            save_path += f'/{name_time}_{args.mode}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            with open(save_path + '/parameters.txt', 'w') as file:
                for arg, value in vars(args).items():
                    file.write(f'{arg}: {value}\n')

            save_path += f'/{bt_type}_b{int(b)}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            if args.using_multiprocessing:
                for i in range(args.sample_num):
                    # rand_seed = int('1' + str(b) + str(i))
                    rand_seed = int('123' + str(b))
                    sampler_list.append(pool.apply_async(run_one, args=(sample_id, save_path, rand_seed, b, bt_type)))
                    sample_id += 1
            else:
                for i in range(args.sample_num):
                    # rand_seed = int('1' + str(b) + str(i))
                    rand_seed = int('123' + str(b))
                    run_one(sample_id, save_path, rand_seed, b, bt_type)
                    sample_id += 1

    # 执行采样
    if args.using_multiprocessing:
        # sample
        for one_sampler in sampler_list:
            one_sampler.get()

    t_all = time.time() - t_start
    print(f'\n=================================================================================\n'
          f'(using_multiprocessing == {args.using_multiprocessing})  All time is '
          f'{round(t_all/3600., 2)} hours ---- {round(t_all/60., 2)} min\n'
          f'=================================================================================\n')

    return f'{name_time}_{args.mode}'


def run_one(sample_id, save_path, rand_seed, b, bt_type):
    args = set_param()
    args.rand_seed = rand_seed
    args.sample_id = sample_id
    args.bt_type = bt_type
    args.time_fac = b

    t_pso = 0
    t_max = 0
    t_poc = 0
    t_all = 0

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

    t0 = time.time()
    args.filename = f_n + '_OUR.csv'
    SQL_0(args)

    t1 = time.time()
    args.filename = f_n + '_OPT.csv'
    Optimal(args)

    t2 = time.time()
    args.filename = f_n + '_PSO.csv'
    PSO_only(args)

    t3 = time.time()

    t_poc += float(t1 - t0)
    t_max += float(t2 - t1)
    t_pso += float(t3 - t2)
    t_all += float(t3 - t0)

    info_dict = {
        'algo_mode': args.mode,
        'mpb_mode': args.bt_change,
        'bt_mode': bt_type,
        'peak_num': args.peak_num,
        'sample_id': sample_id,
        'rand_seed': rand_seed,
        'time_fac': b,
        'save_path': f_n + '_OUR.csv'}

    log_path = 'data_save/other_data/save_log.txt'
    with open(log_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Index'] + list(info_dict.keys()))
        if not os.path.exists(log_path):
            writer.writeheader()
            row_dict = {'Index': 0, **info_dict}  # 以索引0写入第一行数据
            writer.writerow(row_dict)
        else:
            with open(log_path, 'r') as f:
                reader = csv.reader(f)
                lines = list(reader)
                last_index = int(lines[-1][0]) if len(lines) > 1 else 0
                row_dict = {'Index': last_index + 1, **info_dict}  # 使用下一个索引值
                writer.writerow(row_dict)

    print(f'===========================================================\n'
          f'sample_id: {sample_id}    '
          f't_our: {round(t_poc, 3)}   '
          f't_max: {round(t_max, 3)}   '
          f't_pso: {round(t_pso, 3)}   '
          f't_all: {round(t_all, 3)}   ')
    return None


def draw_fit(filename='KD2'):
    args = set_param()
    for bt_type in args.bt_type_list:
        for b in args.b_list:
            save_path = f'data_save/fig_data/{filename}'
            read_path = f'data_save/run_data/{filename}/{bt_type}_b{int(b)}'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            draw_from_file(f=read_path, b=int(b), x=f'{bt_type}_b={int(b)}',
                           save_path=f'{save_path}/{bt_type}_b={int(b)}.png', title=filename, m=args.max_step)


if __name__ == '__main__':
    mode = run_data()
    draw_fit(mode)










