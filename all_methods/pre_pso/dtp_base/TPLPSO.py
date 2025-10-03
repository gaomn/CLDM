# -*- coding: UTF-8 -*-
# @Date    :2024/1/18 22:51
# @Author  :高猛
# @Project :code_cl_test20240109 
# @File    :TPLPSO.py
# @IDE     :PyCharm

# -*- encoding = utf-8 -*-
# @Time : 2022/6/28 17:39
# @Author : 高猛
# @File : my_pso.py
# @Software : PyCharm

import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy

# def fit_fun(X):  # 适应函数
#     # mp.step([0, 0], t)
#     return mp.test(X)


# def fit_fun(X):  # 适应函数
#     return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))


class Particle:
    # 初始化
    def __init__(self, x_max, max_vel, dim, config):

        # np.random.seed(int(str(time.time() % 10)[-6:]))
        # time.sleep(0.001)
        self.__pos = np.random.uniform(-x_max, x_max, dim)  # 粒子的位置
        self.__vel = np.random.uniform(-max_vel, max_vel, dim)  # 粒子的速度

        self.__bestPos = [0.0 for _ in range(dim)]  # 粒子最好的位置
        self.config = config
        self.config['x'] = self.__pos
        self.plm = self.config['plm']
        self.__fitnessValue = self.plm.test(self.config)  # 适应度函数值

    def set_pos(self, i, value):
        self.__pos[i] = value

    def set_pos_array(self, pos_values):
        self.__pos = copy.copy(pos_values)

    def get_pos(self):
        return np.array(self.__pos)

    def set_best_pos(self, i, value):
        self.__bestPos[i] = value

    def set_best_pos_array(self, value):
        self.__bestPos = copy.copy(value)

    def get_best_pos(self):
        return self.__bestPos

    def set_vel(self, i, value):
        self.__vel[i] = value

    def set_vel_array(self, vel_values):
        self.__vel = copy.copy(vel_values)

    def get_vel(self):
        return self.__vel

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class TPLPSO:
    def __init__(self, args, config):
        self.config = config
        self.C1 = args.Individual_learning_factor
        self.C2 = args.Social_learning_factor
        self.C3 = 2
        self.W = args.Inertia_weight
        self.dim = args.x_dim  # 粒子的维度
        self.size = args.Population_size  # 粒子个数
        self.iter_num = args.Iteration_number  # 迭代次数
        self.x_max = args.x_bound
        self.max_vel = args.Max_vel  # 粒子最大速度
        self.best_fitness_value = self.config['best_fitness_value'] if 'best_fitness_value' in self.config.keys() else float('-Inf')
        self.best_position = [0.0 for i in range(self.dim)]  # 种群最优位置
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.plm = self.config['plm']

        # 对种群进行初始化
        self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim, self.config) for i in range(self.size)]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def set_bestPosition_array(self, value):
        self.best_position = copy.copy(value)

    def get_bestPosition(self):
        return np.array(self.best_position)

    def update(self):
        pop = []
        x_list = []
        fit_list = []
        t1 = 0
        t2 = 0
        t3 = 0
        t4 = 0
        for i in range(self.iter_num):
            fes, fc = 0, 0
            for part in self.Particle_list:
                previous_fi = self.plm.test({'x': part.get_pos()})
                t00 = time.time()
                fes, fc = self.teaching_phase(part, fes, fc)

                t01 = time.time()
                Pg, Pi = self.get_bestPosition(), part.get_best_pos()
                self.SPS(Pg, fc, fes)
                self.config['x'] = part.get_pos()
                t02 = time.time()

                t1 += t01 - t00
                t2 += t02 - t01

                if previous_fi < self.plm.test(self.config):
                    t03 = time.time()
                    Pg, Pi = self.get_bestPosition(), part.get_best_pos()
                    fes, fc = self.peer_learning_phase(part, fes, fc)

                    t04 = time.time()
                    fc = self.SPS(Pg, fc, fes)
                    t05 = time.time()

                    t3 += t04 - t03
                    t4 += t05 - t04

                pop.append([copy.deepcopy(part.get_pos()), copy.deepcopy(part.get_fitness_value())])
                x_list.append(copy.deepcopy(part.get_pos()))
                fit_list.append(copy.deepcopy(part.get_fitness_value()))

        return_dict = {'state': [np.max(fit_list), np.max(fit_list) - np.min(fit_list)],
                       'pop': pop,
                       'x_list': x_list,
                       'fit_list': fit_list,
                       'best_x': self.get_bestPosition(),
                       'best_v': self.get_bestFitnessValue(),
                       'time': [t1, t2, t3, t4]}
        return return_dict

    def teaching_phase(self, part, fes, fc):
        fes += 1

        # 生成随机数
        random_numbers = np.random.random(size=self.dim * 2)

        # 将速度值转换为 NumPy 数组
        vel_values = np.array(part.get_vel(), dtype=float)

        # 更新速度
        vel_values += self.W * vel_values + \
                      self.C1 * random_numbers[:self.dim] * (part.get_best_pos() - part.get_pos()) + \
                      self.C2 * random_numbers[self.dim:] * (self.get_bestPosition() - part.get_pos())

        # 使用 clip 函数对速度进行截断
        vel_values = np.clip(vel_values, -self.max_vel, self.max_vel)

        # 设置粒子的新速度
        part.set_vel_array(vel_values)

        # 更新位置
        pos_values = part.get_pos() + vel_values
        part.set_pos_array(pos_values)

        # 计算适应值
        value = self.plm.test({'x': part.get_pos()})

        # 是否优于历史最优
        if value > part.get_fitness_value():
            part.set_fitness_value(value)

            # 更新历史最佳位置
            part.set_best_pos_array(part.get_pos())

            # 是否优于全局最优
            if value > self.get_bestFitnessValue():
                self.set_bestFitnessValue(value)

                # 更新全局最佳位置
                self.set_bestPosition_array(part.get_pos())
                fc = 0
            else:
                fc += 1
        else:
            fc += 1

        return fes, fc

    def teaching_phase2(self, part, fes, fc):
        fes += 1
        # 生成随机数
        random_numbers = np.random.random(size=self.dim * 2)
        # 更新速度
        vel_values = self.W * part.get_vel() + \
                     self.C1 * random_numbers[:self.dim] * (part.get_best_pos() - part.get_pos()) + \
                     self.C2 * random_numbers[self.dim:] * (self.get_bestPosition() - part.get_pos())
        # 使用 clip 函数对速度进行截断
        vel_values = np.clip(vel_values, -self.max_vel, self.max_vel)

        # 设置粒子的新速度
        part.set_vel_array(vel_values)

        # 更新位置
        pos_values = part.get_pos() + part.get_vel()
        part.set_pos_array(pos_values)

        # 计算适应值
        value = self.plm.test({'x': part.get_pos()})

        # 是否优于历史最优
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos_array(part.get_pos())

            # 是否优于全局最优
            if value > self.get_bestFitnessValue():
                self.set_bestFitnessValue(value)
                self.set_bestPosition_array(part.get_pos())
                fc = 0
            else:
                fc += 1
        else:
            fc += 1

        return fes, fc

    def exemplar_selection(self, part, Pg):
        excluded_particles = [p for p in self.Particle_list if p != part and p != Pg]
        ECi_f = np.array([p.get_fitness_value() for p in excluded_particles])

        fmax = np.max(ECi_f)
        fmin = np.min(ECi_f)

        W = (fmax - ECi_f) / (fmax - fmin)
        WCi = np.cumsum(W / np.sum(W))

        r = random.random()
        selected_index = np.searchsorted(WCi, r)

        return np.array(excluded_particles[selected_index].get_pos())

    def peer_learning_phase(self, part, fes, fc):
        Pei = self.exemplar_selection(part, self.get_bestFitnessValue())

        # 生成随机数
        random_numbers = np.random.random(size=self.dim)

        # 根据条件选择更新方式
        if self.plm.test({'x': Pei}) > part.get_fitness_value():
            vel_values = self.W * part.get_vel() + self.C3 * random_numbers * (Pei - part.get_pos())
        else:
            vel_values = self.W * part.get_vel() - self.C3 * random_numbers * (Pei - part.get_pos())

        # 使用 clip 函数对速度进行截断
        vel_values = np.clip(vel_values, -self.max_vel, self.max_vel)

        # 设置粒子的新速度
        part.set_vel_array(vel_values)

        # 更新位置
        pos_values = part.get_pos() + part.get_vel()
        part.set_pos_array(pos_values)

        # 计算适应值
        value = self.plm.test({'x': part.get_pos()})
        fes += 1

        # 更新历史最优和全局最优
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            part.set_best_pos_array(part.get_pos())

            if value > self.get_bestFitnessValue():
                self.set_bestFitnessValue(value)
                self.set_bestPosition_array(part.get_pos())
                fc = 0
            else:
                fc += 1
        else:
            fc += 1

        return fes, fc

    def teaching_phase_old(self, part, fes, fc):
        fes += 1
        # 更新速度
        for i in range(self.dim):
            vel_value = self.W * part.get_vel()[i] + \
                        self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) + \
                        self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
            vel_value = np.clip(vel_value, -self.max_vel, self.max_vel)
            part.set_vel(i, vel_value)

        # 更新位置
        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)
        value = self.plm.test({'x': part.get_pos()})
        # 是否优于历史最优
        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])

            # 是否优于全局最优
            if value > self.get_bestFitnessValue():
                self.set_bestFitnessValue(value)
                for i in range(self.dim):
                    self.set_bestPosition(i, part.get_pos()[i])
                fc = 0
            else:
                fc += 1
        else:
            fc += 1
        return fes, fc

    def exemplar_selection_old(self, part, Pg):
        ECi = np.array([p.get_pos() for idx, p in enumerate(self.Particle_list) if p != part and p != Pg])
        ECi_f = np.array([p.get_fitness_value() for idx, p in enumerate(self.Particle_list) if p != part and p != Pg])
        fmax = np.max(ECi_f)
        fmin = np.min(ECi_f)
        W = []
        for k in range(len(ECi)):
            Wk = (fmax - ECi_f[k]) / (fmax - fmin)
            W.append(Wk)
        WCi = np.cumsum(W / np.sum(W))
        r = random.random()
        selected_index = np.searchsorted(WCi, r)
        return ECi[selected_index]

    def peer_learning_phase_old(self, part, fes, fc):
        Pei = self.exemplar_selection(part, self.get_bestFitnessValue())
        if self.plm.test({'x': Pei}) > part.get_fitness_value():
            for i in range(self.dim):
                vel_value = self.W * part.get_vel()[i] + self.C3 * random.random() * (Pei[i] - part.get_pos()[i])
                vel_value = np.clip(vel_value, -self.max_vel, self.max_vel)
                part.set_vel(i, vel_value)
        else:
            for i in range(self.dim):
                vel_value = self.W * part.get_vel()[i] - self.C3 * random.random() * (Pei[i] - part.get_pos()[i])
                vel_value = np.clip(vel_value, -self.max_vel, self.max_vel)
                part.set_vel(i, vel_value)

        for i in range(self.dim):
            pos_value = part.get_pos()[i] + part.get_vel()[i]
            part.set_pos(i, pos_value)

        value = self.plm.test({'x': part.get_pos()})
        fes += 1

        if value > part.get_fitness_value():
            part.set_fitness_value(value)
            for i in range(self.dim):
                part.set_best_pos(i, part.get_pos()[i])

            if value > self.get_bestFitnessValue():
                self.set_bestFitnessValue(value)
                for i in range(self.dim):
                    self.set_bestPosition(i, part.get_pos()[i])
                fc = 0
            else:
                fc += 1
        else:
            fc += 1

        return fes, fc

    def SPS(self, Pg, fc, fes):
        # 1. if fc ≥ m then
        m = 5
        FEmax = 10000
        if fc >= m:
            # 2. Pgper = Pg;
            Pg_per = copy.deepcopy(Pg)

            # 3. Randomly select a dimension, d to perform perturbation;
            dim_to_perturb = random.randint(0, self.dim - 1)

            # 4. Calculate the range of perturbation R using Eq. (10);
            Rmax, Rmin = 1, 0.1
            R = Rmax - (Rmax - Rmin) * fes / FEmax

            # 5. Perform the perturbation on Pgd using Eq. (9);
            sgn_r5 = 1 if random.random() > 0.5 else -1
            r6 = np.random.normal(0., R)
            perturbation = sgn_r5 * r6 * (2 * self.x_max)
            Pg_per[dim_to_perturb] += perturbation
            Pg_per[dim_to_perturb] = np.clip(Pg_per[dim_to_perturb], -self.x_max, self.x_max)

            # 6. Fitness evaluation is performed on the Pgper particle;
            value = self.plm.test({'x': Pg_per})
            fes += 1

            # 7. if f(Pgper) < f(Pg) then
            if value > self.get_bestFitnessValue():
                # 8: Pg = Pgper; ( ) ( ) per g Pg f P = f ;
                self.set_bestFitnessValue(value)
                self.set_bestPosition(dim_to_perturb, Pg_per[dim_to_perturb])

            fc = 0
        return fc