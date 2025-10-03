
'''
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from minepy import MINE



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



# 生成数据集规模
size = 60

# 生成 x1 到 x5 的随机数据
x1 = np.random.uniform(-5, 5, size)
x2 = np.random.uniform(-5, 5, size)
x3 = np.random.uniform(-5, 5, size)
x4 = np.random.uniform(-5, 5, size)
x5 = np.random.uniform(-5, 5, size)

x_list = [x1, x2, x3, x4, x5]

# 生成对应的 y
# y = np.where(x1 >= 0, 1, 0)
# y = np.where(x1**2 + x2**2 <= 15.9, 1, 0)
# y = np.where(2 * np.sin(0.2 * np.pi * x1) <= x2, 1, 0)

# y = np.zeros((size, ))
# for i in range(size):
#     if x1[i] > 2.5:
#         y[i] = 2
#     elif x1[i] > 0:
#         y[i] = 1
#     elif x1[i] > -2.5:
#         y[i] = -1
#     else:
#         y[i] = -2

y = np.zeros((size, ))
for i in range(size):
    if x1[i]**2 + x2[i]**2 <= 100 / (4*np.pi):
        y[i] = 2
    elif x1[i]**2 + x2[i]**2 <= 200 / (4*np.pi):
        y[i] = 1
    elif x1[i]**2 + x2[i]**2 <= 300 / (4*np.pi):
        y[i] = -1
    else:
        y[i] = -2


noise = np.random.uniform(-0.2, 0.2, size)
y = y + noise

# 计算互信息
I_list = []
P_list = []
S_list = []
K_list = []
N_list = []
M_list = []
svr_list = []
for i in range(5):
    I_list.append(ncc(y, x_list[i]))
    P_list.append(pearsonr(y, x_list[i])[0])
    S_list.append(spearmanr(y, x_list[i])[0])


    mine = MINE()
    mine.compute_score(y, x_list[i])
    M_list.append(mine.mic())

    data1 = y.reshape(-1, 1)
    data2 = x_list[i].reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
    kde.fit(data1)
    log_densities = kde.score_samples(data2)
    densities = np.exp(log_densities)
    K_list.append(np.mean(densities))

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(data1)
    distances, indices = knn.kneighbors(data2)
    N_list.append(np.mean(distances))

    svr = SVR(kernel='rbf')
    svr.fit(data1, data2)
    data2_predicted = svr.predict(data1)
    svr_list.append(svr.score(data1, data2))


# print(y)
print(f'ncc_list = {I_list} \n'
      f'P_list = {P_list} \n'
      f'S_list = {S_list} \n'
      f'K_list = {K_list} \n'
      f'N_list = {N_list} \n'
      f'M_list = {M_list} \n'
      f'svr_list = {svr_list} \n'
      )

'''
import math

'''
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from minepy import MINE


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


def cd(x):
    i_list_ = []
    sort_index_ = np.argsort(x)
    ncc_interval = [x[sort_index_[i + 1]] - x[sort_index_[i]] for i in range(len(x) - 1)]
    i_list_ += list(sort_index_[np.argmax(ncc_interval) + 1:])
    return i_list_

# 生成数据集规模
size = 60

# 生成 x1 到 x5 的随机数据
x1 = np.random.uniform(-5, 5, size)
x2 = np.random.uniform(-5, 5, size)
x3 = np.random.uniform(-5, 5, size)
x4 = np.random.uniform(-5, 5, size)
x5 = np.random.uniform(-5, 5, size)

x_list = [x1, x2, x3, x4, x5]

# 生成对应的 y

            # # if individual[0] >= 0.:
            # # if -3.54 <= individual[0] <= 3.54 and -3.54 <= individual[1] <= 3.54:
            # # if individual[0]**2 + individual[1]**2 <= 15.9:
            # # if 2 * np.sin(0.2 * np.pi * individual[0]) <= individual[1]:

# y = np.where(x1 >= 0, 1, 0)
y = np.where(x 1* *2 + x 2* *2 <= 15.9, 1, 0)
# y = np.where(2 * np.sin(0.2 * np.pi * x1) <= x2, 1, 0)

# y = np.zeros((size, ))
# for i in range(size):
#     if x1[i] > 2.5:
#         y[i] = 2
#     elif x1[i] > 0:
#         y[i] = 1
#     elif x1[i] > -2.5:
#         y[i] = -1
#     else:
#         y[i] = -2

# y = np.zeros((size, ))
# for i in range(size):
#     if x1[i]**2 + x2[i]**2 <= 100 / (4*np.pi):
#         y[i] = 2
#     elif x1[i]**2 + x2[i]**2 <= 200 / (4*np.pi):
#         y[i] = 1
#     elif x1[i]**2 + x2[i]**2 <= 300 / (4*np.pi):
#         y[i] = -1
#     else:
#         y[i] = -2


noise = np.random.uniform(-0.2, 0.2, size)
y = y + noise

# 计算互信息
N_list = []
P_list = []
M_list = []
S_list = []

for i in range(5):
    N_list.append(ncc(y, x_list[i]))
    P_list.append(pearsonr(y, x_list[i])[0])
    S_list.append(spearmanr(y, x_list[i])[0])

    m = MINE()
    m.compute_score(y, x_list[i])
    M_list.append(round(m.mic(), 3))


print(f'ncc_list = {cd(N_list)} \n'
      f'Pear_list = {cd(P_list)} \n'
      f'S_list = {cd(S_list)} \n'
      f'M_list = {cd(M_list)} \n'
      )
'''



import numpy as np
import random
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from minepy import MINE


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


def cd(x):
    i_list_ = []
    sort_index_ = np.argsort(x)
    ncc_interval = [x[sort_index_[i + 1]] - x[sort_index_[i]] for i in range(len(x) - 1)]
    i_list_ += list(sort_index_[np.argmax(ncc_interval) + 1:])
    return i_list_


def generate_data(sigma=0.2, size=60):
    # x1 = np.random.uniform(-5, 5, size)
    # x2 = np.random.uniform(-5, 5, size)
    # x3 = np.random.uniform(-5, 5, size)
    # x4 = np.random.uniform(-5, 5, size)
    # x5 = np.random.uniform(-5, 5, size)
    # x_list = [x1, x2, x3, x4, x5]
    min_coord = -5
    max_coord = 5
    lambda_ = 0.5
    x = [[np.random.random(1)[0] for _ in range(5)]]

    last_vector = [random.random() - 0.5 for _ in range(5)]

    for i in range(size-1):
        shift = [random.random() - 0.5 for _ in range(5)]
        shift_length = sum(s ** 2 for s in shift)
        shift_length = 7.0 / math.sqrt(shift_length) if shift_length > 0 else 0

        shift = [shift_length * (1.0 - lambda_) * s
                 + lambda_ * c for s, c in zip(shift, last_vector)]

        new_position = []
        final_shift = []
        for pp, s in zip(x[-1], shift):
            new_coord = pp + s
            if new_coord < min_coord:
                new_position.append(2.0 * min_coord - pp - s)
                final_shift.append(-1.0 * s)
            elif new_coord > max_coord:
                new_position.append(2.0 * max_coord - pp - s)
                final_shift.append(-1.0 * s)
            else:
                new_position.append(new_coord)
                final_shift.append(s)
        last_vector = final_shift
        x.append(new_position)

    x = np.array(x)
    y = np.where(x[:, 0] ** 2 + x[:, 1] ** 2 <= 15.9, 1, 0)

    num_replacements = 10    # 假设替换5个位置
    replace_indices = np.random.choice(len(y), num_replacements, replace=False)
    random_numbers = np.random.rand(num_replacements)
    y[replace_indices] = random_numbers

    noise = np.random.uniform(-sigma, sigma, size)
    y = y + noise
    return x, y


def reo(x, y):
    P_list = []
    N_list = []
    M_list = []
    for i in range(5):
        P_list.append(pearsonr(y, x[:, i])[0])
        N_list.append(ncc(y, x[:, i]))
        m = MINE()
        m.compute_score(y, x[:, i])
        M_list.append(round(m.mic(), 3))

    all_list = [P_list, N_list, M_list]
    F = [0, 0, 0]
    R = [0, 0, 0]

    for i, c in enumerate(all_list):
        c = cd(c)
        F[i] = 1 if 0 in c and 1 in c else 0
        R[i] = 1 if 0 in c and 1 in c and len(c) == 2 else 0

    return F, R

def cul(n=100, sigma=0.2, size=10):
    nf = [0., 0., 0.]
    nr = [0., 0., 0.]
    for i in range(n):
        x, y = generate_data(sigma, size)
        f, r = reo(x, y)
        for i in range(3):
            nf[i] += f[i]
            nr[i] += r[i]

    rf = [nf[i]/float(n) for i in range(3)]
    rr = [nr[i]/float(n) for i in range(3)]

    # for i in range(3):
    #     print(f'{rf[i]}-{rr[i]}')
    return rf, rr

V = [0, 0.2, 0.4, 0.6, 0.8, 1]
S = [10, 20, 40, 60, 80, 99]

for sigma in V:
    print(f'\n\nsigma={sigma}')
    sigma = 1e-3 if sigma <= 0 else sigma
    for s in S:
        rf, rr = cul(sigma=sigma, size=s)
        print(f'S={s}:  {rf[0]}-{rr[0]}    {rf[1]}-{rr[1]}    {rf[2]}-{rr[2]}')








