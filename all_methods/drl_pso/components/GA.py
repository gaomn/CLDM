# -*- encoding = utf-8 -*-
# @Time : 2022/6/28 17:39
# @Author : 高猛
# @File : my_ga.py
# @Software : PyCharm

import numpy as np
import random
import copy


class Individual:
    # 初始化
    def __init__(self, x_max, dim, config):
        self.__chromosome = list(np.random.uniform(-x_max, x_max, dim))  # 个体的染色体
        self.__bestChromosome = [0.0 for _ in range(dim)]  # 个体最好的染色体
        self.config = config
        self.config['x'] = self.__chromosome
        self.plm = self.config['plm']
        self.__fitnessValue = self.plm.test(self.config)  # 适应度函数值
        
        # 设置初始最佳染色体
        for i in range(dim):
            self.__bestChromosome[i] = self.__chromosome[i]

    def set_chromosome(self, i, value):
        self.__chromosome[i] = value

    def get_chromosome(self):
        return self.__chromosome

    def set_best_chromosome(self, i, value):
        self.__bestChromosome[i] = value

    def get_best_chromosome(self):
        return self.__bestChromosome

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue
        
    @classmethod
    def from_chromosome(cls, chromosome, x_max, dim, config, fitness=None):
        """从现有染色体创建一个新的个体"""
        ind = cls(x_max, dim, config)
        for i in range(dim):
            ind.set_chromosome(i, chromosome[i])
        
        # 如果提供了适应度值，直接设置它，避免重新计算
        if fitness is not None:
            ind.set_fitness_value(fitness)
        else:
            # 评估新个体的适应度
            config['x'] = chromosome
            ind.set_fitness_value(config['plm'].test(config))
            
        return ind


class GA:
    def __init__(self, args, config):
        self.config = config
        self.dim = args.x_dim  # 染色体维度
        self.size = args.Population_size  # 种群大小
        self.iter_num = args.Iteration_number  # 迭代次数
        self.x_max = args.x_bound  # 染色体取值范围
        
        # 遗传算法特有参数（默认值）
        self.crossover_rate = 0.8  # 交叉概率
        self.mutation_rate = 0.1  # 变异概率
        self.tournament_size = 3  # 锦标赛选择大小
        
        self.best_fitness_value = self.config['best_fitness_value'] if 'best_fitness_value' in self.config.keys() else float('-Inf')
        self.best_chromosome = [0.0 for i in range(self.dim)]  # 种群最优染色体
        self.fitness_val_list = []  # 每次迭代最优适应值
        self.plm = self.config['plm']
        
        # 初始化种群
        self.population = [Individual(self.x_max, self.dim, self.config) for i in range(self.size)]
        
        # 寻找初始最优个体
        for individual in self.population:
            if individual.get_fitness_value() > self.best_fitness_value:
                self.best_fitness_value = individual.get_fitness_value()
                for i in range(self.dim):
                    self.best_chromosome[i] = individual.get_chromosome()[i]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestChromosome(self, i, value):
        self.best_chromosome[i] = value

    def get_bestChromosome(self):
        return self.best_chromosome
    
    # 锦标赛选择
    # def selection(self):
    #     selected = []
    #     for _ in range(self.size):
    #         tournament = random.sample(self.population, self.tournament_size)
    #         best = max(tournament, key=lambda ind: ind.get_fitness_value())
    #         selected.append(copy.deepcopy(best))
    #     return selected
    
    def selection(self):
        selected = []
        for _ in range(self.size):
            tournament = random.sample(self.population, self.tournament_size)
            best = max(tournament, key=lambda ind: ind.get_fitness_value())
            
            # 使用 from_chromosome 方法创建新个体
            new_ind = Individual.from_chromosome(
                best.get_chromosome(), 
                self.x_max, 
                self.dim, 
                self.config, 
                best.get_fitness_value()
            )
            selected.append(new_ind)
        return selected
    
    # 交叉操作（单点交叉）
    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1.get_chromosome())
        child2 = copy.deepcopy(parent2.get_chromosome())
        
        if random.random() < self.crossover_rate:
            # 单点交叉
            crossover_point = random.randint(1, self.dim - 1)
            for i in range(crossover_point, self.dim):
                child1[i], child2[i] = child2[i], child1[i]
                    
        return child1, child2
    
    # 变异操作
    def mutation(self, chromosome):
        mutated = copy.deepcopy(chromosome)
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                # 均匀变异
                mutated[i] = random.uniform(-self.x_max, self.x_max)
        return mutated
    
    # 评估适应度并更新最优解
    def evaluate_and_update(self, individual):
        # 评估适应度
        self.config['x'] = individual.get_chromosome()
        fitness = self.plm.test(self.config)
        individual.set_fitness_value(fitness)
        
        # 必要时更新全局最优
        if fitness > self.get_bestFitnessValue():
            self.set_bestFitnessValue(fitness)
            for i in range(self.dim):
                self.set_bestChromosome(i, individual.get_chromosome()[i])
    
    # 更新函数 - 主遗传算法循环
    def update(self):
        pop = []
        x_list = []
        fit_list = []
        
        # 运行指定迭代次数
        for _ in range(self.iter_num):
            # 选择
            selected = self.selection()
            
            # 通过交叉和变异创建新种群
            new_population = []
            
            # 处理父代对以创建后代
            for i in range(0, self.size, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < self.size else selected[0]
                
                # 交叉
                child1_chrom, child2_chrom = self.crossover(parent1, parent2)
                
                # 变异
                child1_chrom = self.mutation(child1_chrom)
                child2_chrom = self.mutation(child2_chrom)
                
                # 创建子代个体
                child1 = Individual(self.x_max, self.dim, self.config)
                for j in range(self.dim):
                    child1.set_chromosome(j, child1_chrom[j])
                
                child2 = Individual(self.x_max, self.dim, self.config)
                for j in range(self.dim):
                    child2.set_chromosome(j, child2_chrom[j])
                
                # 评估并更新最优
                self.evaluate_and_update(child1)
                self.evaluate_and_update(child2)
                
                # 添加到新种群
                new_population.append(child1)
                if len(new_population) < self.size:  # 确保不超过种群大小
                    new_population.append(child2)
            
            # 用新种群替换旧种群
            self.population = new_population
            
            # 收集本代所有个体的数据
            for individual in self.population:
                pop.append([copy.copy(individual.get_chromosome()), copy.copy(individual.get_fitness_value())])
                x_list.append(copy.copy(individual.get_chromosome()))
                fit_list.append(copy.copy(individual.get_fitness_value()))
        
        # 返回与PSO相同格式的字典
        return_dict = {'state': [np.min(fit_list), np.max(fit_list) - np.min(fit_list)],
                       'pop': pop,
                       'x_list': x_list,
                       'fit_list': fit_list,
                       'best_x': self.get_bestChromosome(),
                       'best_v': self.get_bestFitnessValue()}
        return return_dict