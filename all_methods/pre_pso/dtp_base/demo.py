# -*- encoding = utf-8 -*-
# @Time : 2022/6/17 14:24
# @Author : 高猛
# @File : demo.py
# @Software : PyCharm

import csv
import math
import os

from dtp_base import MPB
import time
from tqdm import tqdm
import copy
import random
import datetime
import argparse
import numpy as np
from dtp_base.PSO import PSO
from dtp_base.TPLPSO import TPLPSO
from dtp_base.configs import set_param
from model_test.Model import Model
from model_test.SQL_model import Model as sql_model
from dtp_base.SQL_Predictor import Predictor
from model_test.SQL_model0 import Model as sql_model0
from dtp_base.SQL_Predictor0 import Predictor as Predictor0
from model_test.PSO_Predictor import Model as PreModel


def SQL_0(args):
    pre = Predictor0(args)
    model = sql_model0(args)
    mp = MPB.MovingPeaks(dim=10, npeaks=1, number_severity=0., args=args)

    reward_sum = 0.
    reward_list = []
    h_list = []
    b_list = []
    pre_state_list = []
    real_state_list = []
    r_max = 0.
    r_min = 0.
    x_dict = {}

    # for t in tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step):
    progress_bar = tqdm(desc=f'Progress {args.sample_id + 1}', total=args.max_step)
    for t in range(args.max_step):
        time_reset = time.time()

        # PSO搜索出最优解，得到相关种群
        # print('===============print step : %d, reward :%f================:' % (t, reward_sum),
        #       round(x_best[0], 2) if t >= 1 else '00')
        pso_config = {'name': 'MP',
                      't': t,
                      'plm': mp}
        pso = PSO(args, pso_config)
        pso_dict = pso.update()
        state, pop, x_pso_best = pso_dict['state'], pso_dict['x_list'], pso_dict['best_x']

        # 记录最大最小回报值
        fit_list = pso_dict['fit_list']
        if np.max(fit_list) > r_max:
            r_max = np.max(fit_list)
        if np.min(fit_list) < r_min:
            r_min = np.min(fit_list)

        # 记录时间
        time_pso = time.time()
        # print(f'PSO cost time : {round(time_pso - time_reset, 2)}   min-->max:{round(r_min, 2)}--->{round(r_max, 2)}')

        # t>1时，
        if t >= 1:
            pre.add_data(x_best + [t], state)
            pre.train(t)

            pre_state_list.append(pre_state)
            real_state_list.append(state)

            # print(f'     pre_state : [{pre_state[0]: < 6.2f}, {pre_state[1]: < 6.2f}]    '
            #       f'real_state : [{state[0]: < 6.2f}, {state[1]: < 6.2f}]')

        time_pre = time.time()
        # print(f'Predictor cost time : {round(time_pre - time_pso, 2)}')

        # Modeling and Updating
        model.modeling_updating(t, pso_dict, pre)

        time_mod = time.time()
        # print(f'Modeling cost time : {round(time_mod - time_pre, 2)}')

        # Evaluating and dicision making
        if t >= 3:
            x_best = model.get_x(t, state, pso_dict, pre)
            # x_best = copy.deepcopy(x_pso_best)
        else:
            x_best = x_pso_best
        x_dict[str(t)] = x_best

        pre_state = pre.predict(state, x_best)
        state_last = copy.deepcopy(state)

        time_eva = time.time()
        # print(f'Evaluate cost time : {time_eva - time_mod: 0.2f}')

        mp_config = {'name': 'MP', 'x': x_best}
        re = mp.test(mp_config)
        reward_sum += re
        reward_list.append(re)

        mp.changePeaks(x_best)
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(re), float(reward_sum), x_best, x_pso_best]
            writer = csv.writer(f)
            writer.writerow(row)

        progress_bar.update()
    progress_bar.close()
    for m in model.P.values():
        m[0] = None

    # print('\n=========================over==========================\n')
    # print('\n   reward sum is : ', reward_sum,
    #       '\n   reward list is : ', reward_list,
    #       '\n   h list is : ', h_list,
    #       '\n   b list is : ', b_list)
    # print('\n=========================over==========================')


def SQL_(args):
    pre = Predictor(args)
    model = sql_model(args)
    mp = MPB.MovingPeaks(dim=10, npeaks=1, number_severity=0., args=args)

    reward_sum = 0.
    reward_list = []
    h_list = []
    b_list = []
    pre_state_list = []
    real_state_list = []
    r_max = 0.
    r_min = 0.
    x_dict = {}

    progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
    for t in range(args.max_step):
        time_reset = time.time()

        # PSO搜索出最优解，得到相关种群
        # print('===============print step : %d, reward :%f================:' % (t, reward_sum),
        #       round(x_best[0], 2) if t >= 1 else '00')
        pso_config = {'name': 'MP',
                      't': t,
                      'plm': mp}
        pso = PSO(args, pso_config)
        pso_dict = pso.update()
        state, pop, x_pso_best = pso_dict['state'], pso_dict['x_list'], pso_dict['best_x']

        # 记录最大最小回报值
        fit_list = pso_dict['fit_list']
        if np.max(fit_list) > r_max:
            r_max = np.max(fit_list)
        if np.min(fit_list) < r_min:
            r_min = np.min(fit_list)

        # 记录时间
        time_pso = time.time()
        # print(f'PSO cost time : {round(time_pso - time_reset, 2)}   min-->max:{round(r_min, 2)}--->{round(r_max, 2)}')

        # t>1时，
        if t >= 1:
            pre.add_data(x_best + [t], state)
            pre.train(t)

            pre_state_list.append(pre_state)
            real_state_list.append(state)

            # print(f'     pre_state : [{pre_state[0]: < 6.2f}, {pre_state[1]: < 6.2f}]    '
            #       f'real_state : [{state[0]: < 6.2f}, {state[1]: < 6.2f}]')

        time_pre = time.time()
        # print(f'Predictor cost time : {round(time_pre - time_pso, 2)}')

        # Modeling and Updating
        model.modeling_updating(t, pso_dict, pre)

        time_mod = time.time()
        # print(f'Modeling cost time : {round(time_mod - time_pre, 2)}')

        # Evaluating and dicision making
        if t >= 3:
            x_best = model.get_x(t, state, pop, pre)
            # x_best = copy.deepcopy(x_pso_best)
        else:
            x_best = x_pso_best
        x_dict[str(t)] = x_best

        pre_state = pre.predict(state, x_best)
        state_last = copy.deepcopy(state)

        time_eva = time.time()
        # print(f'Evaluate cost time : {time_eva - time_mod: 0.2f}')

        mp_config = {'name': 'MP', 'x': x_best}
        re = mp.test(mp_config)
        reward_sum += re
        reward_list.append(re)

        mp.changePeaks(x_best)
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(re), float(reward_sum), x_best, x_pso_best]
            writer = csv.writer(f)
            writer.writerow(row)

        progress_bar.update()
    progress_bar.close()

    # print('\n=========================over==========================\n')
    # print('\n   reward sum is : ', reward_sum,
    #       '\n   reward list is : ', reward_list,
    #       '\n   h list is : ', h_list,
    #       '\n   b list is : ', b_list)
    # print('\n=========================over==========================')


def Our_method(args):
    model = Model(args)
    mp = MPB.MovingPeaks(dim=10, npeaks=1, number_severity=0.1, args=args)
    args.mp_config = \
        {'h': args.peak_h,
         'w': args.peak_w,
         'p_num': args.peak_num,
         'sigma': args.peak_sigma}

    reward_sum = 0.
    reward_list = []

    # progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
    for t in range(args.max_step):
        time_begin = time.time()

        # Extracting and Predicting
        pso_config = {'name': 'MP',
                      't': t,
                      'plm': mp}
        pso = PSO(args, pso_config)
        pso_dict = pso.update()

        time_pso = time.time()
        # print('PSO cost time : %s' % str(time_pso - time_begin))

        # Modeling and Updating
        x_best = model.choose_individual(t, pso_dict)
        x_pso_best = pso_dict['best_x']

        time_mod = time.time()
        # print(f'\nstep:{t}   Modeling cost time : {round(time_mod - time_pso, 2)}   \n'
        #       f'         x is: {[round(xi, 2) for xi in x_best]}\n'
        #       f'     pso_x is: {[round(xi, 2) for xi in x_pso_best]}   ')

        x_best_fit = mp(x_best)[0]
        reward_sum += x_best_fit
        reward_list.append(x_best_fit)
        model.save_fit(t, x_best_fit)

        maxinum = mp.maximums()[0]
        x_real_best = maxinum[1]
        if float(x_real_best[0]) >= 0.:
            pass
        else:
            x_temp = copy.deepcopy(x_real_best)
            x_temp[0] = 0
            if mp(x_real_best)[0] > (mp(x_temp)[0] + 2 * args.time_fac):
                pass
            else:
                x_real_best = x_temp
        x_pso_best = pso_dict['best_x']
        mp.changePeaks(x_best)
        # print(f'   step : {t}     reward : {x_best_fit} // {reward_sum}   '
        #       f'\n      x_position : {[round(i, 2) for i in x_best]}'
        #       f'\n      pso_best   : {[round(i, 2) for i in x_pso_best]}   ===real===   {x_real_best[0]}\n\n\n')

        # 标准csv格式写入
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(x_best_fit), float(reward_sum), x_best, x_real_best]
            writer = csv.writer(f)
            writer.writerow(row)

        # progress_bar.update()

    x_tra, f_this, f_next = model.get_tra_data(args.max_step)
    return x_tra, f_this, f_next

    # progress_bar.close()
    # print('\n=========================over==========================\n')
    # print('   reward sum is : ', reward_sum,
    #       '\n   reward list is : ', reward_list,
    #       '\n   h list is : ', h_list,
    #       '\n   b list is : ', b_list)
    # print('\n=========================over==========================')


def Pre_only(args):
    model = Model(args)
    mp = MPB.MovingPeaks(dim=10, npeaks=1, number_severity=0.1, args=args)
    args.mp_config = \
        {'h': args.peak_h,
         'w': args.peak_w,
         'p_num': args.peak_num,
         'sigma': args.peak_sigma}

    reward_sum = 0.
    reward_list = []

    progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
    for t in range(args.max_step):
        time_begin = time.time()

        # Extracting and Predicting
        pso_config = {'name': 'MP',
                      't': t,
                      'plm': mp}
        pso = PSO(args, pso_config)
        pso_dict = pso.update()

        time_pso = time.time()
        # print('PSO cost time : %s' % str(time_pso - time_begin))

        # Modeling and Updating
        x_best = model.choose_individual(t, pso_dict)
        x_pso_best = pso_dict['best_x']

        time_mod = time.time()
        # print(f'\nstep:{t}   Modeling cost time : {round(time_mod - time_pso, 2)}   \n'
        #       f'         x is: {[round(xi, 2) for xi in x_best]}\n'
        #       f'     pso_x is: {[round(xi, 2) for xi in x_pso_best]}   ')

        x_best_fit = mp(x_best)[0]
        reward_sum += x_best_fit
        reward_list.append(x_best_fit)
        model.save_fit(t, x_best_fit)

        maxinum = mp.maximums()[0]
        x_real_best = maxinum[1]
        if float(x_real_best[0]) >= 0.:
            pass
        else:
            x_temp = copy.deepcopy(x_real_best)
            x_temp[0] = 0
            if mp(x_real_best)[0] > (mp(x_temp)[0] + 2 * args.time_fac):
                pass
            else:
                x_real_best = x_temp
        x_pso_best = pso_dict['best_x']
        mp.changePeaks(x_best)
        # print(f'   step : {t}     reward : {x_best_fit} // {reward_sum}   '
        #       f'\n      x_position : {[round(i, 2) for i in x_best]}'
        #       f'\n      pso_best   : {[round(i, 2) for i in x_pso_best]}   ===real===   {x_real_best[0]}\n\n\n')

        # 标准csv格式写入
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(x_best_fit), float(reward_sum), x_best, x_real_best]
            writer = csv.writer(f)
            writer.writerow(row)

        progress_bar.update()
    progress_bar.close()
    # print('\n=========================over==========================\n')
    # print('   reward sum is : ', reward_sum,
    #       '\n   reward list is : ', reward_list,
    #       '\n   h list is : ', h_list,
    #       '\n   b list is : ', b_list)
    # print('\n=========================over==========================')


def PSO_only(args):
    mp = MPB.MovingPeaks(dim=args.x_dim, npeaks=1, number_severity=0.1, args=args)
    f_sum = 0.

    for t in range(args.max_step):

        # use pso
        pso = PSO(args, {'plm': mp})
        pso_dict = pso.update()
        x_pso_best = pso_dict['best_x']

        # get fitness
        fit = mp.test({'x': x_pso_best})
        f_sum += fit
        mp.changePeaks(x_pso_best)

        # save data
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(fit), float(f_sum)]
            writer = csv.writer(f)
            writer.writerow(row)


def TPLPSO(args):
    mp = MPB.MovingPeaks(dim=args.x_dim, npeaks=1, number_severity=0.1, args=args)
    f_sum = 0.

    for t in range(args.max_step):

        # use pso
        pso = TPLPSO(args, {'plm': mp})
        pso_dict = pso.update()
        x_pso_best = pso_dict['best_x']

        # get fitness
        fit = mp.test({'x': x_pso_best})
        f_sum += fit
        mp.changePeaks(x_pso_best)

        # save data
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(fit), float(f_sum)]
            writer = csv.writer(f)
            writer.writerow(row)


def PSO_Pre(args):
    model = PreModel(args)
    mp = MPB.MovingPeaks(dim=10, npeaks=1, number_severity=0.1, args=args)
    args.mp_config = \
        {'h': args.peak_h,
         'w': args.peak_w,
         'p_num': args.peak_num,
         'sigma': args.peak_sigma}

    reward_sum = 0.
    reward_list = []
    time_start = time.time()
    time_this = time.time()

    # progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
    for t in tqdm(range(args.max_step)):

        # Extracting and Predicting
        pso_config = {'name': 'MP',
                      't': t,
                      'plm': mp}
        pso = PSO(args, pso_config)
        pso_dict = pso.update()

        # print('PSO cost time : %s' % str(time_pso - time_begin))

        # Modeling and Updating
        x_best = model.choose_individual(t, pso_dict)
        x_pso_best = pso_dict['best_x']

        # print(f'\nstep:{t}   Modeling cost time : {round(time_mod - time_pso, 2)}   \n'
        #       f'         x is: {[round(xi, 2) for xi in x_best]}\n'
        #       f'     pso_x is: {[round(xi, 2) for xi in x_pso_best]}   ')

        x_best_fit = mp(x_best)[0]
        reward_sum += x_best_fit
        reward_list.append(x_best_fit)
        model.save_fit(t, x_best_fit)

        maxinum = mp.maximums()[0]
        x_real_best = maxinum[1]
        if float(x_real_best[0]) >= 0.:
            pass
        else:
            x_temp = copy.deepcopy(x_real_best)
            x_temp[0] = 0
            if mp(x_real_best)[0] > (mp(x_temp)[0] + 2 * args.time_fac):
                pass
            else:
                x_real_best = x_temp
        x_pso_best = pso_dict['best_x']
        mp.changePeaks(x_best)
        # print(f'   step : {t}     reward : {x_best_fit} // {reward_sum}   '
        #       f'\n      x_position : {[round(i, 2) for i in x_best]}'
        #       f'\n      pso_best   : {[round(i, 2) for i in x_pso_best]}   ===real===   {x_real_best[0]}\n\n\n')
        t_step = time.time() - time_this
        t_all = time.time() - time_start
        # 标准csv格式写入
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(x_best_fit), float(reward_sum),t_step, t_all]
            writer = csv.writer(f)
            writer.writerow(row)
        time_this = time.time()
        # progress_bar.update()

    x_tra, f_this, f_next = model.get_tra_data(args.max_step)
    return x_tra, f_this, f_next

    # progress_bar.close()
    # print('\n=========================over==========================\n')
    # print('   reward sum is : ', reward_sum,
    #       '\n   reward list is : ', reward_list,
    #       '\n   h list is : ', h_list,
    #       '\n   b list is : ', b_list)
    # print('\n=========================over==========================')


def Optimal(args):
    mp = MPB.MovingPeaks(dim=args.x_dim, npeaks=1, number_severity=0.1, args=args)
    f_sum = 0.

    for t in range(args.max_step):

        # find optimal solution
        maxinum = mp.maximums()[0]
        x_ = maxinum[1]
        bt_type = args.bt_type
        if bt_type == 'linear':
            if x_[0] >= 0:
                x_best = x_
            else:
                x_temp = copy.deepcopy(x_)
                x_temp[0] = 0
                if mp.test({'name': 'MP', 'x': x_}) > (mp.test({'name': 'MP', 'x': x_temp}) + 2 * args.time_fac):
                    x_best = x_
                else:
                    x_best = x_temp
        elif bt_type == 'sin':
            if 2 * np.sin(0.2 * np.pi * x_[0]) >= x_[1]:
                x_best = x_
            else:
                x_temp = copy.deepcopy(x_)
                x_temp[1] = 2 * np.sin(0.2 * np.pi * x_[0])
                if mp.test({'name': 'MP', 'x': x_}) > (mp.test({'name': 'MP', 'x': x_temp}) + 2 * args.time_fac):
                    x_best = x_
                else:
                    x_best = x_temp
        elif bt_type == 'cir':
            if x_[0]**2 + x_[1]**2 <= 15.9:
                x_best = x_
            else:
                x_temp = copy.deepcopy(x_)
                sr = (15.89/(x_[0]**2 + x_[1]**2))**0.5
                x_temp[0] = x_[0] * sr
                x_temp[1] = x_[1] * sr

                if mp.test({'name': 'MP', 'x': x_}) > (mp.test({'name': 'MP', 'x': x_temp}) + 2 * args.time_fac):
                    x_best = x_
                else:
                    x_best = x_temp
        elif bt_type == 'rect':
            if -3.54 <= x_[0] <= 3.54 and -3.54 <= x_[1] <= 3.54:
                x_best = x_
            else:
                x_temp = copy.deepcopy(x_)

                # 方形边界
                dots = [[-3.54, -3.54], [-3.54, 3.54], [3.54, -3.54], [3.54, 3.54], ]
                n = np.argmin([math.sqrt((di[0] - x_[0])**2 + (di[1] - x_[1])**2) for di in dots])
                if -3.54 <= x_[0] <= 3.54 and -3.54 <= x_[1] <= 3.54:
                    x_temp = x_
                elif -3.54 <= x_[0] <= 3.54:
                    x_temp = [dots[n][1] if xi == 1 else xd for xi, xd in enumerate(x_temp)]
                elif -3.54 <= x_[1] <= 3.54:
                    x_temp = [dots[n][0] if xi == 0 else xd for xi, xd in enumerate(x_temp)]
                else:
                    x_temp = dots[n] + x_[2:]

                if mp.test({'name': 'MP', 'x': x_}) > (mp.test({'name': 'MP', 'x': x_temp}) + 2 * args.time_fac):
                    x_best = x_
                else:
                    x_best = x_temp
        elif bt_type == 'linear4':
            xb = set_param().x_bound
            x_0 = copy.deepcopy(x_)

            x_1 = copy.deepcopy(x_)
            x_1[0] = xb/2.

            x_2 = copy.deepcopy(x_)
            x_2[0] = 0.

            x_3 = copy.deepcopy(x_)
            x_3[0] = -xb/2.


            x_lst = [x_0, x_1, x_2, x_3]
            v_lst = [mp.test({'name': 'MP', 'x': x_tmp}) +
                     MPB.MovingPeaks.get_sym(x_tmp) * args.time_fac for x_tmp in x_lst]

            x_best = x_lst[int(np.argmax(v_lst))]

        else:
            print("bt_type error! ensure your bt_type: 'linear', 'sin', 'cir', 'rect'")
            raise

        # get fitness
        fit = mp.test({'x': x_best})
        f_sum += fit
        mp.changePeaks(x_best)

        # save data
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(fit), float(f_sum)]
            writer = csv.writer(f)
            writer.writerow(row)

'''
if __name__ == '__main__':
    args = set_param()

    t_list = [100, 50, 10]
    for t in t_list:

        args.time_fac = t
        save_path = args.filename

        save_path += f'/{args.mode}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path += f'/data_b{int(t)}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        t_pso = 0
        t_max = 0
        t_poc = 0
        t_all = 0

        for i in range(5):
            a = str(datetime.datetime.now())
            f_n = f'{save_path}/b={int(args.time_fac)}_MPB_{a[8:10] + a[11:13] + a[14:16] + a[17:19]}'

            args.rand_seed = int('1' + a[14:16] + a[17:19] + a[20:24])

            t0 = time.time()
            args.filename = f_n + '_POC.csv'
            POC(args)

            t1 = time.time()
            args.filename = f_n + '_OPT.csv'
            Nor(args)

            t2 = time.time()
            args.filename = f_n + '_PSO.csv'
            PSO_(args)

            t3 = time.time()

            t_poc += float(t1 - t0)
            t_max += float(t2 - t1)
            t_pso += float(t3 - t2)
            t_all += float(t3 - t0)
            print(f'===========================================================\n'
                  f'sample epoch: {i}    '
                  f't_poc: {round(float(t1 - t0), 3)}/{round(t_poc, 3)}   '
                  f't_max: {round(float(t2 - t1), 3)}/{round(t_max, 3)}   '
                  f't_pso: {round(float(t3 - t2), 3)}/{round(t_pso, 3)}   '
                  f't_all: {round(float(t3 - t0), 3)}/{round(t_all, 3)}   ')

'''




