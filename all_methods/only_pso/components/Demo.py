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
# from components.CSVC_Model import CSVC_Model
from components.CLDM_Model import CLDM_Model


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
