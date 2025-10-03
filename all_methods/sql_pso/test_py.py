import numpy as np
import csv
import os

log_path = 'data_save/other_data/save_log.txt'

for i in range(10):
    info_dict = {
        'algo_mode': 'TTTTTT',
        'mpb_mode': (i + 1) * 100
    }

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
                # 根据最后一行的索引自动生成逐渐增加的序号
                row_dict = {'Index': last_index + 1, **info_dict}  # 使用下一个索引值
                writer.writerow(row_dict)



