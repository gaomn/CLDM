import heapq
from collections import deque
from typing import Any, List, Tuple
import numpy as np

class HybridPER:
    def __init__(self, max_size: int, gamma: float, phase_threshold: float):
        """
        初始化混合优先级经验回放缓冲区。

        参数:
            max_size (int): 缓冲区最大容量
            gamma (float): 用于优先级计算的折扣因子
            phase_threshold (float): 阶段切换阈值（0到1之间的百分比）
        """
        self.max_size = max_size
        self.gamma = gamma
        self.phase_threshold = phase_threshold
        self.buffer = deque(maxlen=max_size)  # 存储经验的队列
        self.priority_heap = []  # 最大堆，存储 (-priority, step, experience)
        self.step_counter = 0  # 全局步骤计数器

    def add_experience(self, state: Any, action: Any, reward: float, next_state: Any, 
                       log_prob: float, value: float, td_error: float) -> None:
        """
        添加新经验到缓冲区和优先级堆。

        参数:
            state (Any): 当前状态
            action (Any): 采取的动作
            reward (float): 即时奖励
            next_state (Any): 下一状态
            log_prob (float): 动作的对数概率
            value (float): 当前状态的价值估计
            td_error (float): 时序差分误差
        """
        priority = -(abs(td_error) ** self.gamma)  # 负优先级模拟最大堆
        experience = (state, action, reward, next_state, log_prob, value, td_error)
        self.buffer.append(experience)
        heapq.heappush(self.priority_heap, (priority, self.step_counter, experience))
        self.step_counter += 1

    def sample(self, batch_size: int, current_step: int, n: int) -> Tuple[List[Any], List[Any], 
                                                                        List[float], List[Any], 
                                                                        List[float], List[float], 
                                                                        List[float]]:
        """
        根据当前步骤分阶段采样经验。
        先进行均匀采样，然后通过优先级采样（放回）直到达到目标样本数量。

        参数:
            batch_size (int): 期望的批量大小
            current_step (int): 当前训练步骤
            n (int): 总步骤数，用于确定采样阶段

        返回:
            Tuple[List[Any], ...]: 包含状态、动作、奖励、下一状态、对数概率、价值、TD误差的列表
        """
        # 计算当前批量大小（动态调整）
        current_batch = min(2 * n, 1 + (current_step / 1e5) * (2 * n - 1))
        target_batch_size = min(batch_size * 2, int(current_batch))
        
        # 确保有可用经验
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []
        
        experiences = []
        
        # 首先进行均匀采样（无放回）
        if len(self.buffer) <= target_batch_size:
            experiences = list(self.buffer)
        else:
            indices = np.random.choice(len(self.buffer), min(len(self.buffer), target_batch_size), replace=False)
            experiences = [self.buffer[i] for i in indices]
        
        # 如果当前处于第二阶段或样本不足，则使用优先级采样补充
        if current_step >= self.phase_threshold * n or len(experiences) < target_batch_size:
            # 准备优先级采样
            valid_heap_items = []
            min_step = self.step_counter - len(self.buffer)  # 缓冲区中有效经验的最小步骤
            
            # 获取所有有效的堆项目
            temp_heap = self.priority_heap.copy()
            while temp_heap:
                priority, step, exp = heapq.heappop(temp_heap)
                if step >= min_step:
                    valid_heap_items.append((priority, step, exp))
            
            # 如果有有效项目，使用优先级进行放回采样
            if valid_heap_items:
                # 使用负优先级（因为堆是最小堆）计算抽样概率
                priorities = np.array([-item[0] for item in valid_heap_items])
                probabilities = priorities / np.sum(priorities)
                
                # 持续采样直到达到目标样本数量
                while len(experiences) < target_batch_size:
                    idx = np.random.choice(len(valid_heap_items), p=probabilities)
                    experiences.append(valid_heap_items[idx][2])  # 添加经验
        
        # 处理经验不足的情况（虽然有放回采样不太可能发生）
        if not experiences:
            return [], [], [], [], [], [], []

        # 解包经验并返回
        states, actions, rewards, next_states, log_probs, values, td_errors = zip(*experiences)
        return (list(states), list(actions), list(rewards), list(next_states), 
                list(log_probs), list(values), list(td_errors))