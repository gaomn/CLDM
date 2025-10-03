import matplotlib.pyplot as plt
import csv
import os
import numpy as np


def get_arguments(arg_list, arg_dict):
    for arg in arg_list:
        if arg in list(arg_dict.keys()):
            return arg_dict[arg]
    print(f'Missing Argument : {arg_dict[0]}')
    raise


def check_arguments(arg_list, arg_dict, default_value):
    for arg in arg_list:
        if arg in list(arg_dict.keys()):
            return arg_dict[arg]
    return default_value


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


def check_key(key_list, all_list):
    for key in key_list:
        if key not in all_list:
            return False
    return True


def get_data_list(key_list, paths, max_step):
    list_csv = list_dir(file_dir=paths)
    data_list = []
    for filename in list_csv:
        if check_key(key_list, str(filename)):
            a_list = []
            with open(filename, encoding='utf-8') as f:
                reader = csv.reader(f)
                for data in list(reader)[-max_step:]:
                    a_list.append(float(data[2]))
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


def draw_from_file(**kwargs):
    """
    给定文件夹，获取数据绘制图像
    """
    data_file = get_arguments(['filename', 'f', 'path'], kwargs)
    max_step = check_arguments(['max_step', 'max_steps', 'm', 'max'], kwargs, 100)
    time_fac = check_arguments(['b', 'time_fac'], kwargs, 100)
    save_path = check_arguments(['save_path', 'SavePath'], kwargs, False)
    xlabel = check_arguments(['xlable', 'x'], kwargs, f'MPB, b={int(time_fac)}')
    ylabel = check_arguments(['ylable', 'y'], kwargs, 'Accumulated Fitness')
    title = check_arguments(['title', 't'], kwargs, 'data plot')

    mean_lst1, std_list1 = get_data_list([f'b={int(time_fac)}', 'OPT'], data_file, max_step)
    mean_lst2, std_list2 = get_data_list([f'b={int(time_fac)}', 'PSO'], data_file, max_step)
    mean_lst3, std_list3 = get_data_list([f'b={int(time_fac)}', 'SQN'], data_file, max_step)

    x_list = list(range(len(mean_lst1)))
    fig, ax = plt.subplots()  # 创建图实例
    ax.plot(x_list, mean_lst1, color='red', label='Optimal')
    ax.fill_between(x_list, [mean_lst1[i] + std_list1[i] for i in range(len(x_list))],
                    [mean_lst1[i] - std_list1[i] for i in range(len(x_list))],  # 上限，下限
                    facecolor='red',  # 填充颜色
                    alpha=0.3)  # 透明度

    ax.plot(x_list, mean_lst2, color='magenta', label='PSO')
    ax.fill_between(x_list, [mean_lst2[i] + std_list2[i] for i in range(len(x_list))],
                    [mean_lst2[i] - std_list2[i] for i in range(len(x_list))],  # 上限，下限
                    facecolor='magenta',  # 填充颜色
                    alpha=0.3)  # 透明度

    ax.plot(x_list, mean_lst3, color='g', label='Our')
    ax.fill_between(x_list, [mean_lst3[i] + std_list3[i] for i in range(len(x_list))],
                    [mean_lst3[i] - std_list3[i] for i in range(len(x_list))],  # 上限，下限
                    facecolor='g',  # 填充颜色
                    alpha=0.3)  # 透明度

    ax.set_xlabel(xlabel)  # 设置x轴名称 x label
    ax.set_ylabel(ylabel)  # 设置y轴名称 y label
    ax.set_title(title)  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    if save_path:
        plt.savefig(save_path)
    plt.show()  # 图形可视化