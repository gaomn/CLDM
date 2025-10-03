import copy
import csv
import os
import numpy as np
from minepy import MINE
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import matplotlib.pyplot as plt


# 产生若干csv文件用于测试
def create_data(path):
    for i in range(10):
        with open(path + f'test_{str(i)}.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            row = [[100 * i + 10 * k + j for j in range(5)] for k in range(3)]
            writer.writerows(row)


# 将所有文件的路径放入到listcsv列表中
def list_dir(file_dir):
    list_filename = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        # 判断是文件夹还是文件
        if os.path.isfile(path):
            # print("{0} : is file!".format(cur_file))
            dir_files = os.path.join(file_dir, cur_file)
        # 判断是否存在.csv文件，如果存在则获取路径信息写入到list_csv列表中
        if os.path.splitext(path)[1] == '.csv':
            csv_file = os.path.join(file_dir, cur_file)
            list_filename.append(csv_file)
        if os.path.isdir(path):
            list_dir(path)
    return list_filename


# 读取和修改CSV文件
def read_and_fix(filename):
    # 读取文件为列表
    old_data = None
    with open(filename, 'r', encoding='utf-8') as f:
        old_data = list(csv.reader(f))

    # 修改列表内容，例如：全部+1
    new_data = [[int(data) + 1 for data in line]for line in old_data]

    # 将数据放回去
    with open(filename, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for line in new_data:
            writer.writerow(line)


def check_key(key_list, all_list):
    for key in key_list:
        if key not in all_list:
            return False
    return True


def get_data_list(key_list, paths, max_step):
    list_csv = list_dir(file_dir=paths)
    data_list = []
    for filename in list_csv:
        if check_key(key_list, str(filename)) and len(data_list) < 30:
            a_list = []
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for data in list(reader)[-max_step:]:
                    a_list.append(float(data[2]))
            if len(a_list) == 100:
                data_list.append(a_list)
    if len(data_list) > 1:
        np_data = np.array(data_list, dtype=float)
        mean_list = []
        std_list = []
        for i in range(np_data.shape[1]):
            # 第i列均值方差归一化
            mean_list.append(np.mean(np_data[:, i]))
            std_list.append(np.std(np_data[:, i]) + 1e-5)
        return mean_list, std_list
    elif len(data_list) == 1:
        return data_list, [0 for _ in data_list]
    else:
        print('list error !!!')
        raise


def get_data_list2(key_list, paths, max_step):
    list_csv = list_dir(file_dir=paths)
    data_list = []
    for filename in list_csv:
        if check_key(key_list, str(filename)) and len(data_list) < 30:
            a_list = []
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for data in list(reader)[-max_step:]:
                    a_list.append(float(data[1]))
            if len(a_list) == 100:
                data_list.append(a_list)
    return data_list[0]
    # if len(data_list) > 1:
    #     print(f'   {len(data_list)}      {[len(s) for s in data_list]}')
    #     np_data = np.array(data_list, dtype=float)
    #     mean_list = []
    #     std_list = []
    #     for i in range(np_data.shape[1]):
    #         # 第i列均值方差归一化
    #         mean_list.append(np.mean(np_data[:, i]))
    #         std_list.append(np.std(np_data[:, i]) + 1e-5)
    #     return mean_list, std_list
    # elif len(data_list) == 1:
    #     return data_list, [0 for _ in data_list]
    # else:
    #     print('list error !!!')
    #     raise


def get_data_list3(key_list, paths, max_step):
    list_csv = list_dir(file_dir=paths)
    data_list = []
    for filename in list_csv:
        if check_key(key_list, str(filename)) and len(data_list) < 30:
            a_list = []
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for data in list(reader)[-max_step:]:
                    a_list.append(float(data[2]))
            if len(a_list) == 100:
                data_list.append(a_list)
    return data_list[0]

def add_from_str(str_list):
    data_list = []
    for si, se in enumerate(str_list):
        if se == '[' or se == ' ':
            start_i = si + 1
        if se == ',' or se == ']':
            end_i = si
            data_list.append(float(str_list[start_i:end_i]))
    return data_list


def get_data_list(filename):
    x_list = []
    a_list = []
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in list(reader)[-100:]:
            a_list.append(float(data[1]))
            x_list.append(add_from_str(data[3]))

    return a_list, x_list


def cut_x(x_list):
    cut_list = [[] for _ in range(len(x_list[0]))]
    for x in x_list[:-1]:
        for i, xi in enumerate(x):
            cut_list[i].append(xi)
    return cut_list


def ncc(x, y):
    n = len(x)
    b = int(n ** 0.5)
    if (np.max(x) != np.min(x)) and (np.max(y) != np.min(y)):
        detax = (np.max(x) - np.min(x) + 0.00001 * (np.max(x) - np.min(x))) / float(b)
        detay = (np.max(y) - np.min(y) + 0.00001 * (np.max(y) - np.min(y))) / float(b)
        if detax != 0 and detay != 0:
            p = np.zeros((b, b))
            x1 = np.ceil((x - np.min(x) + 0.000005 * (np.max(x) - np.min(x))) / detax)
            y1 = np.ceil((y - np.min(y) + 0.000005 * (np.max(y) - np.min(y))) / detay)
            x1 = [1 if x <= 0 else x for x in x1]
            y1 = [1 if y <= 0 else y for y in y1]
            x1 = [b if x >= b else x for x in x1]
            y1 = [b if y >= b else y for y in y1]
            for i in range(n):
                p[int(x1[i]) - 1][int(y1[i]) - 1] += 1 / n
            ncc = 0
            for i in range(b):
                for j in range(b):
                    if p[i][j] != 0:
                        ncc += p[i][j] * np.log(p[i][j]) / np.log(b)
            for i in range(b):
                if np.sum(p[i]) != 0:
                    ncc -= (np.sum(p[i])) * np.log(np.sum(p[i])) / np.log(b)
            for i in range(b):
                p_i = np.sum([p_row[i] for p_row in p])
                if p_i != 0:
                    ncc -= p_i * np.log(p_i) / np.log(b)
        else:
            ncc = 0
    else:
        ncc = 0
    cor = np.cov(x, y)
    if cor[0, 1] < 0:
        ncc = -ncc
    return ncc


def mic(x, y, alpha_=0.6):
    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mic()


def mcn(x, y, alpha_=0.6):
    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mcn()


def mev(x, y, alpha_=0.6):
    mine = MINE(alpha=alpha_)
    mine.compute_score(x, y)
    return mine.mev()

def kendall_correlation(x, y):
    corr, p_value = kendalltau(x, y)
    return corr

def pearsonr_correlation(x, y):
    corr, p_value = pearsonr(x, y)
    return corr


def spearmanr_correlation(x, y):
    corr, p_value = spearmanr(x, y)
    return corr


def cov_(x, y):

    x = np.array(copy.deepcopy(x))
    y = np.array(copy.deepcopy(y))

    x = (x - np.mean(x)) / (np.std(x) + 1e-5)
    y = (y - np.mean(y)) / (np.std(y) + 1e-5)


    covariance = np.cov(x, y)[0][1]
    return covariance


def choose_some(in_list):
    in_array = np.array(copy.deepcopy(in_list))
    in_index = np.argsort(-in_array)
    in_sort = in_array[in_index]

    edge = [in_sort[i] - in_sort[i + 1] for i in range(len(in_sort) - 1)]
    result_lst = in_index[:np.argmax(edge) + 1]

    rate = np.max(edge)/(np.max(in_sort) - np.min(in_sort + 1e-5))

    return result_lst, rate


def choose_some2(in_list):
    temp_array = []
    for l in in_list:
        temp_array.append((l - np.mean(l))/(np.std(l) + 1e-5))
    in_array = [0. for _ in range(len(in_list[0]))]
    for l in temp_array:
        for i, d in enumerate(l):
            in_array[i] += d

    in_array = np.array(in_array)
    in_index = np.argsort(-in_array)
    in_sort = in_array[in_index]

    edge = [in_sort[i] - in_sort[i + 1] for i in range(len(in_sort) - 1)]
    result_lst = in_index[:np.argmax(edge) + 1]

    rate = np.max(edge)/(np.max(in_sort) - np.min(in_sort + 1e-5))

    return result_lst, rate


def cul_rate(in_list):
    in_array = np.array(copy.deepcopy(in_list))
    in_index = np.argsort(-in_array)
    in_sort = in_array[in_index]

    edge = [in_sort[i] - in_sort[i + 1] for i in range(len(in_sort) - 1)]

    rate = np.max(edge)/(np.max(in_sort) - np.min(in_sort + 1e-5))

    return rate


def cul_cor(filename, al=0.4):
    a_list, x_list = get_data_list(filename)

    xi_list = cut_x(x_list)
    yi_list = a_list[1:]

    n1 = 0.
    n2 = 0.
    n3 = 0.

    for t in range(4, len(yi_list)):
        cor_list = [ncc(xi_[: t], yi_list[: t]) for xi_ in xi_list]
        cor_list2 = [cov_(xi_[: t], yi_list[: t]) for xi_ in xi_list]
        cor_list3 = [mic(xi_[: t], yi_list[: t], alpha_=al) for xi_ in xi_list]
        cor_list4 = [kendall_correlation(xi_[: t], yi_list[: t]) for xi_ in xi_list]
        cor_list5 = [pearsonr_correlation(xi_[: t], yi_list[: t]) for xi_ in xi_list]
        cor_list6 = [spearmanr_correlation(xi_[: t], yi_list[: t]) for xi_ in xi_list]

        c_list = [cor_list, cor_list2, cor_list3, cor_list4, cor_list5, cor_list6]
        need = [[] for _ in range(5)]
        for c_ in c_list:
            n, r = choose_some(c_)
            need[0].append(1 if 0 in n and 1 in n and len(n) == 2 else 0)
            need[1].append(1 if 0 in n and 1 in n else 0)
            # need[0].append(1 if 0 in n and len(n) == 1 else 0)
            # need[1].append(1 if 0 in n else 0)
            need[2].append(r + float(np.e)**(-len(n)))
            # need[2].append(r + 1 / float(len(n)))
            need[3].append(len(n))
            need[4].append(n[0])

        # n, r = choose_some2([cor_list2, cor_list5])
        # need[0].append(1 if 0 in n and len(n) == 1 else 0)
        # need[1].append(1 if 0 in n else 0)
        # need[2].append(r)
        print(f'step {t}:   ', end='')
        for i in range(len(c_list)):
            print(f' {i + 1}: ',  end='')
            if need[0][i]:
                print('\033[32m',  f'r({need[4][i]})', '\033[0m', end='-')
            else:
                print(f' r({need[4][i]}) ', end='-')

            if need[1][i]:
                print('\033[34m',  f'f({need[3][i]})', '\033[0m', end='-')
            else:
                print(f' f({need[3][i]}) ', end='-')

            if need[2][i] == np.max(need[2]):
                print('\033[31m', 'r({:.2f})'.format(need[2][i]), '\033[0m', end='       ')
            else:
                print(' r({:.2f}) '.format(need[2][i]), end='       ')
        print('')
        # print(f'step {t}:   ', end='')
        # for i in range(3):
        #     print(f' {i + 1}: ',  end='')
        #     if need[0][i]:
        #         print('\033[32m',  f'r({need[0][i]})', '\033[0m', end='-')
        #     else:
        #         print(f' r({need[0][i]}) ', end='-')
        #
        #     if need[1][i]:
        #         print('\033[34m',  f'f({need[1][i]})', '\033[0m', end='-')
        #     else:
        #         print(f' f({need[1][i]}) ', end='-')
        #
        #     if need[2][i] == np.max(need[2]):
        #         print('\033[31m', 'r{:.2f}'.format(need[2][i]), '\033[0m', end='       ')
        #     else:
        #         print(' r{:.2f} '.format(need[2][i]), end='       ')
        # print('')

        n1_list = [0 for _ in range(len(c_list))]
        n2_list = [0 for _ in range(len(c_list))]
        n3_list = [0 for _ in range(len(c_list))]

        test_ = 4
        for i in range(len(c_list)):
            n1_list[i] += 1.
            if need[0][i]:
                n2_list[i] += 1.
            if need[1][i]:
                n3_list[i] += 1.

        r1_list = [n2_list[i] / n1_list[i] for i in range(len(c_list))]
        r2_list = [n3_list[i] / n1_list[i] for i in range(len(c_list))]

        n1 += 1.
        if need[0][test_]:
            n2 += 1.
        if need[1][test_]:
            n3 += 1.


    return r1_list, r2_list


def test_cor(path_dir):
    # filename = 'data/data_b100/b=100_MPB_23224335_POC.csv'
    # filename = 'data/data_b50/b=50_MPB_20222216_POC.csv'
    # path_dir = 'POC3/data/'
    b_list = [10, 50, 100]
    data_list1 = [[] for _ in range(6)]
    data_list2 = [[] for _ in range(6)]

    for bi in b_list:
        path_file = path_dir + f'data_b{int(bi)}'
        list_csv = list_dir(path_file)
        r1_list = [[] for _ in range(6)]
        r2_list = [[] for _ in range(6)]
        for fi, filename in enumerate(list_csv):
            if '_POC' in filename:
                print('\n\n\n============================================================')
                print(f'           b={bi}       fi={fi}        ')

                r1, r2 = cul_cor(filename)
                for i in range(6):
                    r1_list[i].append(r1[i])
                    r2_list[i].append(r2[i])
        for i in range(6):
            data_list1[i].append([np.mean(r1_list[i]), np.std(r1_list[i])])
            data_list2[i].append([np.mean(r2_list[i]), np.std(r2_list[i])])

    for i in range(6):
        for data in data_list1[i]:
            print(f'   {i}:   {round(data[0], 2)} ~ {round(data[1], 2)}')

        print()
        for data in data_list2[i]:
            print(f'   {i}:   {round(data[0], 2)} ~ {round(data[1], 2)}')
        print()
    for i in range(6):
        print(data_list1[i])
        print(data_list2[i])



if __name__ == '__main__':
    # filename = 'data/data_b100/b=100_MPB_23224335_POC.csv'
    # filename = 'data/data_b50/b=50_MPB_20222216_POC.csv'
    path_dir = 'POC3/data/'
    b_list = [10, 50, 100]
    data_list1 = []
    data_list2 = []

    for bi in b_list:
        path_file = path_dir + f'data_b{int(bi)}'
        list_csv = list_dir(path_file)
        r1_list = []
        r2_list = []
        for fi, filename in enumerate(list_csv):
            if '_POC' in filename:
                print('\n\n\n============================================================')
                print(f'           b={bi}       fi={fi}        ')

                r1, r2 = cul_cor(filename)
                r1_list.append(r1)
                r2_list.append(r2)
        data_list1.append([np.mean(r1_list), np.std(r1_list)])
        data_list2.append([np.mean(r2_list), np.std(r2_list)])

    for data in data_list1:
        print(f'        {round(data[0], 2)} ~ {round(data[1], 2)}')

    print()
    for data in data_list2:
        print(f'        {round(data[0], 2)} ~ {round(data[1], 2)}')
    print()
    print(data_list1)
    print(data_list2)
