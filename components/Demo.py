# -*- encoding = utf-8 -*-
# @Time : 2022/6/17 14:24
# @Author : 高猛
# @File : demo.py
# @Software : PyCharm

import csv
import math
import os
import time
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm
from components import MPB
from components.PSO import PSO
# from components.GA import GA as PSO
# from components.Bayesian import BayesianOptimization as PSO
# from components.CSVC_Model import CSVC_Model
from components.CLDM_Model import CLDM_Model
from components.PPO import PPOAgent


def PPO_PSO(args):
    """
    Main loop of CL-PSO with PPO agent integration.
    Uses PSO results for the first 5 steps, then switches to PPO for action selection.
    """

    ### Initialize Environment
    mp = MPB.MovingPeaks(dim=args.x_dim, npeaks=1, number_severity=0.1, args=args)
    reward_sum = 0.

    ### Initialize PPO Agent
    # State dimension: [f_min, f_range] + x_max (2 + dim)
    ppo_agent = PPOAgent(2+args.x_dim, args.x_dim)

    ### Main Loop
    progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
    for t in range(args.max_step):
        time_begin = time.time()

        # Use PSO to get population
        pso_config = {
            'name': 'MP',
            't': t,
            'plm': mp
        }
        pso = PSO(args, pso_config)
        pso_dict = pso.update()

        # Action Selection Logic
        fit_list = pso_dict['fit_list']
        f_max = np.max(fit_list)
        f_range =  f_max - np.min(fit_list)
        x_max = pso_dict['best_x']
        state = [f_max, f_range] + list(x_max)
        if t < 5 or random.random() < 0.1:
            # Use PSO result directly for the first 5 steps
            x_best = pso_dict['best_x']
        else:
            # Use PPO to select action from step 5 onward
            x_best = ppo_agent.select_action(state, is_priority=True)

        # Evaluate Fitness
        x_best_fit = mp(x_best)[0]
        reward_sum += x_best_fit

        # Update Environment
        mp.changePeaks(x_best)

        # Update PPO Agent (starting from t >= 1)
        if t >= 1:
            # Get next state by re-running PSO
            next_pso_dict = pso.update()
            next_fit_list = next_pso_dict['fit_list']
            next_f_max = np.max(next_fit_list)
            next_f_range = next_f_max - np.min(next_fit_list)
            next_x_max = next_pso_dict['best_x']
            
            next_state = [next_f_max, next_f_range] + list(next_x_max)
            # Define reward as negative fitness (minimization problem)
            reward = - x_best_fit - next_f_max
            ppo_agent.store_transition(state, x_best, reward, next_state)
            ppo_agent.update()

        # Measure Time Cost
        time_cost = time.time() - time_begin

        # Save Data
        # with open(args.filename, 'a', newline='') as f:
        #     row = [int(t), float(x_best_fit), float(reward_sum), float(time_cost)]
        #     writer = csv.writer(f)
        #     writer.writerow(row)

        with open(args.filename, 'a', newline='') as f:
            # 创建CSV行，包括原来的数据加上新的最优解信息
            row = [int(t), 
                   float(x_best_fit), 
                   float(reward_sum), 
                   float(time_cost)]               # PPO最优解(10维)
            writer = csv.writer(f)
            writer.writerow(row)

        progress_bar.update()
    progress_bar.close()


def CL_PSO(args):
    """
    main loop of cl-pso
    """

    '''  init environment '''
    model = CLDM_Model(args)
    mp = MPB.MovingPeaks(dim=10, npeaks=1, number_severity=0.1, args=args)
    reward_sum = 0.

    '''  loop star '''
    progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
    # for t in tqdm(range(args.max_step)):
    for t in range(args.max_step):
        time_begin = time.time()

        # use PSO to get pop
        pso_config = {'name': 'MP',
                      't': t,
                      'plm': mp}
        pso = PSO(args, pso_config)
        pso_dict = pso.update()

        # use CLDM to get x_best
        x_best = model.choose_individual(t, pso_dict)
        x_best_fit = mp(x_best)[0]
        reward_sum += x_best_fit
        model.save_fit(t, x_best_fit)

        mp.changePeaks(x_best)
        time_cost = time.time() - time_begin

        # save data
        with open(args.filename, 'a', newline='') as f:
            row = [int(t), float(x_best_fit), float(reward_sum), float(time_cost)]
            writer = csv.writer(f)
            writer.writerow(row)
        # if t % 99 == 0 and t != 0:
        #     print(f'ID:{args.sample_id}, step:{t}, reward_sum:{reward_sum}, time_cost:{time_cost}'
        #           f'b -> {args.bt_change} - {args.bt_type} - {args.time_fac}\n'
        #           f'===============================================================================')
        progress_bar.update()
    progress_bar.close()


def only_PSO(args):
    """
    main loop of csvc-pso
    """

    '''  init environment '''
    mp = MPB.MovingPeaks(dim=args.x_dim, npeaks=1, number_severity=0.1, args=args)
    f_sum = 0.

    progress_bar = tqdm(range(args.max_step), desc=f'ID:{args.sample_id}', total=args.max_step)
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
        progress_bar.update()
    progress_bar.close()
