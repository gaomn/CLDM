import numpy as np
import random


# 初始化粒子群和参数
def initialize_particles(num_particles, dimension, search_space):
    particles = np.random.uniform(search_space[0], search_space[1], (num_particles, dimension))
    velocities = np.zeros((num_particles, dimension))
    personal_bests = particles.copy()
    global_best_index = np.argmin([objective_function(p) for p in particles])
    global_best = particles[global_best_index].copy()
    archive = []

    return particles, velocities, personal_bests, global_best, archive


# 目标函数
def objective_function(x):
    return np.sum(x**2)


# 主循环
def tplpso(particles, velocities, personal_bests, global_best, archive,
           max_iterations, p_learn, m, w, c1, c2, search_space):
    iteration = 0

    while iteration < max_iterations:
        for i in range(len(particles)):
            # 教学阶段
            r = random.random()
            velocities[i] = w * velocities[i] + \
                            c1 * r * (personal_bests[i] - particles[i]) + \
                            c2 * r * (global_best - particles[i])
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], search_space[0], search_space[1])
            fitness_i = objective_function(particles[i])

            if fitness_i < objective_function(personal_bests[i]):
                personal_bests[i] = particles[i]

            if fitness_i < objective_function(global_best):
                global_best = particles[i]

        # 同伴学习阶段
        for i in range(len(particles)):
            if objective_function(particles[i]) >= objective_function(personal_bests[i]):
                if random.random() < p_learn:
                    j = random.choice([idx for idx in range(len(particles)) if idx != i])
                    exemplar = personal_bests[j] if \
                        objective_function(personal_bests[j]) < objective_function(personal_bests[i]) else global_best
                    r = random.random()
                    particles[i] = particles[i] + r * (exemplar - particles[i])
                    particles[i] = np.clip(particles[i], search_space[0], search_space[1])
                    fitness_i = objective_function(particles[i])

                    if fitness_i < objective_function(personal_bests[i]):
                        personal_bests[i] = particles[i]

                    if fitness_i < objective_function(global_best):
                        global_best = particles[i]

        # 停滞预防策略
        if iteration >= m and objective_function(global_best) == objective_function(global_best_previous):
            k = random.randint(0, len(archive) - 1)
            if objective_function(personal_bests[k]) < objective_function(global_best):
                r = random.random()
                global_best = global_best + r * (personal_bests[k] - global_best)

        # 归档更新策略
        for i in range(len(particles)):
            if objective_function(personal_bests[i]) < objective_function(global_best):
                archive.append(personal_bests[i])
                if len(archive) > max_archive_size:
                    archive.sort(key=lambda x: objective_function(x))
                    archive.pop()

        global_best_previous = global_best.copy()
        iteration += 1

    return global_best


# 调用
num_particles = 20
dimension = 50
search_space = (-100, 100)
max_iterations = 100
p_learn = 0.5
m = 5
w = 0.9
c1 = 2
c2 = 2
max_archive_size = 10

particles, velocities, personal_bests, global_best, archive = \
    initialize_particles(num_particles, dimension, search_space)
result = tplpso(particles, velocities, personal_bests, global_best, archive,
                max_iterations, p_learn, m, w, c1, c2, search_space)
print("Optimal Solution:", result)
print("Objective Value:", objective_function(result))
