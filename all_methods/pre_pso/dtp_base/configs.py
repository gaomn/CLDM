# -*- coding: UTF-8 -*-
# @Date    :2023/4/18 22:25
# @Author  :高猛
# @Project :DTPs 
# @File    :configs.py
# @IDE     :PyCharm
import argparse


def set_param():
    parser = argparse.ArgumentParser()

    ''' mpb args '''
    parser.add_argument('--s_dim', default=2, type=int)
    parser.add_argument('--x_dim', default=200, type=int)
    parser.add_argument('--x_bound', default=5., type=float)
    parser.add_argument('--peak_h', default=50, type=int)
    parser.add_argument('--peak_w', default=5, type=int)
    parser.add_argument('--peak_sigma', default=5, type=float)
    parser.add_argument('--time_fac', default=100., type=float)
    parser.add_argument('--max_step', default=30, type=float)
    parser.add_argument('--bt_type', default='linear', type=str,  help="'linear', 'sin', 'cir'")
    parser.add_argument('--bt_change', default='continuous', type=str, help="discrete, continuous")
    parser.add_argument('--peak_num', default=1, type=int)

    ''' pso args '''
    parser.add_argument('--Population_size', default=100, type=int)
    parser.add_argument('--Iteration_number', default=100, type=int)
    parser.add_argument('--Inertia_weight', default=0.75, type=float)
    parser.add_argument('--Individual_learning_factor', default=1.4, type=float)
    parser.add_argument('--Social_learning_factor', default=1.4, type=float)
    parser.add_argument('--Max_vel', default=5., type=float)

    ''' pre args '''
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--key_variable', default='mic', type=str, help="'mic', 'pearson', 'ncc'")
    parser.add_argument('--use_label', default=True, type=bool)

    ''' model args '''
    parser.add_argument('--if_crossing', default=False, type=bool)
    parser.add_argument('--if_feedback', default=False, type=bool)
    parser.add_argument('--mode', default='cl', type=str, help="'pxm', 'kxm', 'k2xm', 'kfxim', 'kfxsm', 'kfdxim'")
    parser.add_argument('--k_means_center', default=10, type=int)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--mic_a', default=0.8, type=float)
    parser.add_argument('--svm_kernel', default='linear', type=str,  help="'linear', 'poly', 'rbf', 'sigmoid'")
    parser.add_argument('--net_model', default='rbf', type=str, help="'rbf', 'mlp_relu', 'mlp_tanh', 'gnn'")
    # 'pxm 5m', 'kxm 20.23m','k2xm_k3f5 14.44m', 'kfxim 7.17m', 'kf2dxim 6.95m',  'kfdxim 6.51m'
    # 'cl_tanh', 'clf_tanh', 'cl_rbf 2.98h' 'cl_relu3.82'

    ''' train args '''
    parser.add_argument('--rand_seed', default=10086, type=float)
    parser.add_argument('--MPB_seed', default=123, type=float)
    parser.add_argument('--filename',  default='data_save/run_data', type=str)
    parser.add_argument('--save_k_means', default='data_save/other_fig', type=str)
    parser.add_argument('--using_multiprocessing', default=False, type=bool)

    parser.add_argument('--peak_num_list', default=[1], type=list)
    parser.add_argument('--bt_change_list', default=['discrete', 'continuous'], type=list, help="discrete, continuous")
    parser.add_argument('--bt_type_list', default=['linear', 'sin', 'cir'], type=list, help="'linear', 'sin', 'cir'")
    parser.add_argument('--k_means_center_list', default=[10], type=list)
    parser.add_argument('--b_list', default=[10, 50, 100], type=list)
    parser.add_argument('--sample_num', default=1, type=int)

    ''' other args '''
    parser.add_argument('--eta_', default=0.9, type=float)
    parser.add_argument('--iso_tree', default=100, type=int)
    parser.add_argument('--ano_thr', default=0.25, type=float)
    parser.add_argument('--sub_size', default=256, type=int)
    parser.add_argument('--Lambda', default=4, type=int)
    parser.add_argument('--numCenter', default=14, type=int)
    parser.add_argument('--delta', default=0.3, type=float)
    parser.add_argument('--gamma', default=0.8, type=float)
    parser.add_argument('--if_plot', default=True, type=bool)

    args = parser.parse_args()
    return args

