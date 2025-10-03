import csv
import os
import numpy as np


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
        # print(f'key_list, str(filename): {key_list, str(filename)}   list_csv: {list_csv} ')
        if check_key(key_list, str(filename)) and len(data_list) < 30:
            a_list = []
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for data in list(reader)[-max_step:]:
                    if len(data) > 2:
                        a_list.append(float(data[2]))
            if len(a_list) >= max_step:
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
        return data_list[0], np.zeros((len(data_list[0]), ))
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



# if __name__ == '__main__':
#     data_file = ''
#     max_step = 100
#
#     d1 = get_data_list2([f'b={int(10)}_MPB', '_OPT'], data_file, max_step)


    # paths = r'C:/Users/12199/Desktop/test/data/'
    # # create_data(paths)
    # list_csv = list_dir(file_dir=paths)
    #
    # for filename in list_csv:
    #     read_and_fix(filename)




