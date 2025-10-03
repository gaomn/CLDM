import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from typing import List, Union
import numpy as np
from components.HybridPER import HybridPER

class PPOAgent:
    def __init__(self, input_dim: int, action_dim: int, max_size: int = 10000, gamma: float = 0.99, 
                 phase_threshold: float = 0.2, batch_size: int = 64, n: int = 100000, 
                 epsilon: float = 0.2, value_loss_coef: float = 0.5, lr: float = 3e-4,
                 fitness_stats: dict = None):
        """
        初始化PPO代理。

        参数:
            input_dim (int): 状态维度
            action_dim (int): 动作维度
            max_size (int): 经验回放缓冲区最大容量
            gamma (float): 折扣因子
            phase_threshold (float): HybridPER阶段阈值
            batch_size (int): 更新时的批量大小
            n (int): 总步骤数估计
            epsilon (float): PPO剪切参数
            value_loss_coef (float): 价值损失系数
            lr (float): 学习率
            fitness_stats (dict): 适应值统计信息，包含'min'和'max'
        """
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.n = n
        self.epsilon = epsilon
        self.value_loss_coef = value_loss_coef
        self.step_counter = 0

        # 适应值归一化所需的统计信息
        if fitness_stats is None:
            # 默认统计值，会在训练过程中动态更新
            self.fitness_stats = {
                'f_min': {'min': float('inf'), 'max': float('-inf')},
                'f_range': {'min': 0, 'max': 10000}  # 假设的初始范围
            }
        else:
            self.fitness_stats = fitness_stats
        
        # 初始化HybridPER
        self.hybrid_per = HybridPER(max_size, gamma, phase_threshold)
        
        # 新增：输入归一化模块
        self.input_normalizer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()  # 确保输入在[-1,1]范围内
        )
        
        # 新增：输出反归一化系数（可学习参数）
        self.output_scaler = nn.Parameter(torch.tensor(5.0))

        # 策略网络（修改：增加Tanh激活函数限制输出范围）
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * action_dim),
            nn.Tanh()  # 确保输出在[-1,1]范围内
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.value_net.parameters()) + 
            list(self.input_normalizer.parameters()) + 
            [self.output_scaler],  # 添加新增的可学习参数
            lr=lr
        )
    
    def normalize_input(self, state):
        """
        输入归一化处理：分别处理适应值部分和x值部分
        状态格式为: [f_min, f_range] + list(x_max)
        
        参数:
            state: 输入状态
            
        返回:
            torch.Tensor: 归一化后的状态
        """
        # 转换为张量
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        # NaN处理：将NaN值替换为0
        state = torch.where(torch.isnan(state), torch.zeros_like(state), state)
        
        # 分离适应值部分和x值部分
        if len(state.shape) > 1:  # 批处理情况
            f_values = state[:, :2]  # 前两个元素是f_min和f_range
            x_values = state[:, 2:]  # 剩余元素是x值
            
            # 适应值使用min-max归一化 (假设我们有历史统计数据)
            # 如果没有统计数据，可以使用动态更新的方法
            f_normalized = self.normalize_fitness_values(f_values)
            
            # x值使用线性缩放归一化 ([-5,5] -> [-1,1])
            x_normalized = x_values / 5.0
            
            # 重新组合
            normalized_state = torch.cat([f_normalized, x_normalized], dim=1)
        else:  # 单个状态情况
            f_values = state[:2]  # 前两个元素是f_min和f_range
            x_values = state[2:]  # 剩余元素是x值
            
            # 适应值使用min-max归一化
            f_normalized = self.normalize_fitness_values(f_values)
            
            # x值使用线性缩放归一化
            x_normalized = x_values / 5.0
            
            # 重新组合
            normalized_state = torch.cat([f_normalized, x_normalized])
        
        # 通过归一化网络进一步处理
        return self.input_normalizer(normalized_state)
    
    def denormalize_output(self, action):
        """
        输出反归一化：将[-1,1]范围映射回[-5,5]
        
        参数:
            action: 网络输出的动作
            
        返回:  
            反归一化后的动作
        """
        return action * self.output_scaler

    def select_action(self, state: List[float], is_priority: bool = True) -> np.ndarray:
        """
        根据当前状态选择动作。

        参数:
            state (List[float]): 当前状态
            is_priority (bool): 是否优先选择（兼容接口，当前未使用）

        返回:
            np.ndarray: 动作向量
        """
        # 输入归一化处理
        normalized_state = self.normalize_input(state)
        
        with torch.no_grad():
            # 通过策略网络获取动作分布参数
            policy_output = self.policy_net(normalized_state)
            
            # 分割均值和对数标准差
            mean, log_std = policy_output.chunk(2, dim=-1)
            
            # 反归一化：将[-1,1]映射回[-5,5]
            mean = self.denormalize_output(mean)
            
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
        
        return action.numpy()
    

    def store_transition(self, s, a, r, s_, log_prob=None, v=None, td_error=None):
        """
        存储单次经验。当log_prob、v和td_error为None时，自动通过当前策略计算这些值。
    
        参数:
            s (List[float]): 当前状态
            a (np.ndarray): 动作
            r (float): 即时奖励
            s_ (List[float]): 下一状态
            log_prob (torch.Tensor, optional): 动作对数概率，如果为None则自动计算
            v (torch.Tensor, optional): 价值估计，如果为None则自动计算
            td_error (torch.Tensor, optional): TD误差，如果为None则自动计算
        """
        if isinstance(s_, list) and len(s_) >= 2:
            with torch.no_grad():
                # 更新f_min统计
                f_min = s_[0]
                self.fitness_stats['f_min']['min'] = min(self.fitness_stats['f_min']['min'], f_min)
                self.fitness_stats['f_min']['max'] = max(self.fitness_stats['f_min']['max'], f_min)
                
                # 更新f_range统计
                f_range = s_[1]
                self.fitness_stats['f_range']['min'] = min(self.fitness_stats['f_range']['min'], f_range)
                self.fitness_stats['f_range']['max'] = max(self.fitness_stats['f_range']['max'], f_range)

        # 当log_prob, v, td_error为None时需要计算它们
        if log_prob is None or v is None or td_error is None:
            with torch.no_grad():
                # 转换状态为张量并进行归一化
                if not isinstance(s, torch.Tensor):
                    state_tensor = torch.tensor(s, dtype=torch.float32)
                else:
                    state_tensor = s
                normalized_state = self.normalize_input(state_tensor)
                
                # 转换动作为张量
                if not isinstance(a, torch.Tensor):
                    action_tensor = torch.tensor(a, dtype=torch.float32)
                else:
                    action_tensor = a
                
                # 如果log_prob为None，计算它
                if log_prob is None:
                    policy_output = self.policy_net(normalized_state)
                    mean, log_std = policy_output.chunk(2, dim=-1)
                    mean = self.denormalize_output(mean)
                    std = torch.exp(log_std)
                    dist = Normal(mean, std)
                    log_prob = dist.log_prob(action_tensor).sum(dim=-1)
                
                # 如果v为None，计算它
                if v is None:
                    v = self.value_net(normalized_state)
                
                # 如果td_error为None且s_不为None，计算它
                if td_error is None and s_ is not None:
                    next_state_tensor = torch.tensor(s_, dtype=torch.float32) if not isinstance(s_, torch.Tensor) else s_
                    normalized_next_state = self.normalize_input(next_state_tensor)
                    next_v = self.value_net(normalized_next_state).item()
                    # 计算TD误差：r + γ·V(s') - V(s)
                    td_error = r + self.gamma * next_v - v.item()
    
        # 添加经验到HybridPER
        self.hybrid_per.add_experience(s, a, r, s_, log_prob, v, td_error)
        self.step_counter += 1
        

    def update(self) -> None:
        """
        使用单次经验更新代理。
        """

        # 如果缓冲区足够，进行PPO更新
        if len(self.hybrid_per.buffer) >= 4:
            states, actions, rewards, next_states, log_probs, values, td_errors = \
                self.hybrid_per.sample(self.batch_size, self.step_counter, self.n)
            
            # 转换为张量并归一化状态
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            normalized_states = self.normalize_input(states_tensor)
            
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
            
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
            normalized_next_states = self.normalize_input(next_states_tensor)
            
            old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32)

            # 修复：正确处理values，确保它们是标量值
            processed_values = []
            for v in values:
                if isinstance(v, torch.Tensor):
                    # 如果是张量，提取标量值
                    processed_values.append(v.item() if v.numel() == 1 else v[0].item())
                elif hasattr(v, '__iter__'):  # 列表、元组或类似迭代对象
                    # 处理列表或嵌套列表的情况
                    if isinstance(v[0], torch.Tensor):
                        processed_values.append(v[0].item() if v[0].numel() == 1 else v[0][0].item())
                    else:
                        processed_values.append(float(v[0]))
                else:
                    # 如果已经是标量，直接转换为float
                    processed_values.append(float(v))

            values = torch.tensor(processed_values, dtype=torch.float32).view(-1, 1)


            processed_td_errors = []
            for td in td_errors:
                if isinstance(td, torch.Tensor):
                    processed_td_errors.append(td.item() if td.numel() == 1 else td[0].item())
                elif hasattr(td, '__iter__'):  # 列表、元组或类似迭代对象
                    if isinstance(td[0], torch.Tensor):
                        processed_td_errors.append(td[0].item() if td[0].numel() == 1 else td[0][0].item())
                    else:
                        processed_td_errors.append(float(td[0]))
                else:
                    processed_td_errors.append(float(td))
                    
            td_errors = torch.tensor(processed_td_errors, dtype=torch.float32)

            # 计算新策略的对数概率
            policy_output = self.policy_net(normalized_states)
            mean, log_std = policy_output.chunk(2, dim=-1)
            
            # 反归一化均值
            mean = self.denormalize_output(mean)
            
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # PPO剪切损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * td_errors  # 使用td_error作为优势估计
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * td_errors
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            with torch.no_grad():
                next_values = self.value_net(normalized_next_states)
                td_targets = rewards.view(-1, 1) + self.gamma * next_values
            new_values = self.value_net(normalized_states)
            value_loss = (new_values - td_targets).pow(2).mean()

            # 总损失
            loss = policy_loss + self.value_loss_coef * value_loss

            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def normalize_fitness_values(self, fitness_values):
        """
        适应值归一化：对f_min和f_range进行特殊处理
        
        参数:
            fitness_values: 适应值[f_min, f_range]
            
        返回:
            归一化后的适应值
        """
        if len(fitness_values.shape) > 1:  # 批处理情况
            # 分别处理f_min和f_range
            f_min = fitness_values[:, 0]
            f_range = fitness_values[:, 1]
            
            # 更新统计信息
            with torch.no_grad():
                self.fitness_stats['f_min']['min'] = min(self.fitness_stats['f_min']['min'], torch.min(f_min).item())
                self.fitness_stats['f_min']['max'] = max(self.fitness_stats['f_min']['max'], torch.max(f_min).item())
                self.fitness_stats['f_range']['min'] = min(self.fitness_stats['f_range']['min'], torch.min(f_range).item())
                self.fitness_stats['f_range']['max'] = max(self.fitness_stats['f_range']['max'], torch.max(f_range).item())
            
            # 归一化f_min
            f_min_norm = self.min_max_normalize(
                f_min, 
                self.fitness_stats['f_min']['min'],
                self.fitness_stats['f_min']['max']
            )
            
            # 归一化f_range
            f_range_norm = self.min_max_normalize(
                f_range,
                self.fitness_stats['f_range']['min'],
                self.fitness_stats['f_range']['max']
            )
            
            return torch.stack([f_min_norm, f_range_norm], dim=1)
        
        else:  # 单个样本情况
            f_min = fitness_values[0]
            f_range = fitness_values[1]
            
            # 更新统计信息
            with torch.no_grad():
                self.fitness_stats['f_min']['min'] = min(self.fitness_stats['f_min']['min'], f_min.item())
                self.fitness_stats['f_min']['max'] = max(self.fitness_stats['f_min']['max'], f_min.item())
                self.fitness_stats['f_range']['min'] = min(self.fitness_stats['f_range']['min'], f_range.item())
                self.fitness_stats['f_range']['max'] = max(self.fitness_stats['f_range']['max'], f_range.item())
            
            # 归一化f_min
            f_min_norm = self.min_max_normalize(
                f_min, 
                self.fitness_stats['f_min']['min'],
                self.fitness_stats['f_min']['max']
            )
            
            # 归一化f_range
            f_range_norm = self.min_max_normalize(
                f_range,
                self.fitness_stats['f_range']['min'],
                self.fitness_stats['f_range']['max']
            )
            
            return torch.tensor([f_min_norm, f_range_norm])
    
    def min_max_normalize(self, values, min_val, max_val):
        """
        最小-最大归一化
        
        参数:
            values: 需要归一化的值
            min_val: 最小值
            max_val: 最大值
            
        返回:
            归一化后的值
        """
        if max_val - min_val > 1e-8:
            return (values - min_val) / (max_val - min_val) * 2 - 1  # 映射到[-1,1]
        return torch.zeros_like(values)
    
    def normalize_rewards(self, rewards):
        """
        奖励归一化：使用min-max归一化
        
        参数:
            rewards: 奖励列表
            
        返回:
            归一化后的奖励
        """
        if isinstance(rewards, torch.Tensor):
            min_reward = torch.min(rewards)
            max_reward = torch.max(rewards)
            # 防止除零
            if max_reward - min_reward > 1e-8:
                return (rewards - min_reward) / (max_reward - min_reward)
            return torch.zeros_like(rewards)
        else:
            rewards = np.array(rewards)
            min_reward = np.min(rewards)
            max_reward = np.max(rewards)
            # 防止除零
            if max_reward - min_reward > 1e-8:
                return (rewards - min_reward) / (max_reward - min_reward)
            return np.zeros_like(rewards)