import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class BayesianOptimization:
    def __init__(self, args, config):
        """
        Initialize the Bayesian Optimization algorithm.
        
        Args:
            args: Arguments containing algorithm parameters
            config: Configuration dictionary
        """
        self.config = config
        self.dim = args.x_dim
        self.size = args.Population_size
        self.iter_num = args.Iteration_number
        self.x_max = args.x_bound
        self.plm = self.config['plm']
        
        # 使用Matern核，通常比RBF更鲁棒
        # nu=1.5表示一次可微，对噪声数据更好
        self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, 
                                                    length_scale_bounds=(1e-2, 1e2),
                                                    nu=1.5)
        
        # 减少n_restarts_optimizer参数，完全避免重新启动优化器
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            n_restarts_optimizer=0,  # 不使用多重启动以加速
            alpha=1e-4,             # 提高数值稳定性
            normalize_y=True,        # 标准化输出
            optimizer='fmin_l_bfgs_b'
        )
        
        # Initialize the best solution
        self.best_position = None
        self.best_fitness_value = float('inf')
        
        # Initialize data storage
        self.X_samples = []
        self.y_samples = []
        
        # 调整探索-利用平衡参数 - 降低以更快收敛
        self.kappa = 1.0
        
        # Initialize bounds for optimization
        self.bounds = [(-self.x_max, self.x_max) for _ in range(self.dim)]
        
        # 大幅减少模型重新拟合频率
        self.samples_since_last_fit = 0
        self.fit_frequency = max(5, self.iter_num // 5)  # 每5-20次迭代拟合一次模型
        
        # 增加批处理大小以减少迭代次数
        self.batch_size = min(10, max(2, self.iter_num // 5))
        
        # 局部搜索的简化参数
        self.n_random_starts = min(200, 5 * self.dim)
        
        # 是否启用简化版本
        self.use_simplified_gp = True
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the population with random samples."""
        # 简化初始化：使用纯随机采样而非拉丁超立方
        X_init = np.random.uniform(-self.x_max, self.x_max, (self.size, self.dim))
        
        # Evaluate initial samples
        y_init = np.zeros(self.size)
        for i in range(self.size):
            self.config['x'] = X_init[i]
            y_init[i] = self.plm.test(self.config)
        
        # Store initial data
        self.X_samples = X_init
        self.y_samples = y_init
        
        # Update best solution
        best_idx = np.argmin(y_init)
        self.best_position = X_init[best_idx].copy()
        self.best_fitness_value = y_init[best_idx]
        
        # 拟合初始高斯过程模型
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(self.X_samples, self.y_samples)
        except Exception as e:
            print(f"初始模型拟合错误，使用默认模型: {e}")
            # 如果拟合失败，使用默认内核
            self.kernel = C(1.0) * RBF(length_scale=1.0)
            self.gp = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=1e-4,
                normalize_y=True,
                n_restarts_optimizer=0
            )
            self.gp.fit(self.X_samples, self.y_samples)
    
    def _expected_improvement(self, x, gp, y_best):
        """
        Calculate the Expected Improvement acquisition function.
        简化版本以提高计算速度。
        
        Args:
            x: The point to evaluate
            gp: Gaussian process model
            y_best: The current best function value
            
        Returns:
            The negative expected improvement (negative because we minimize)
        """
        # 对于批量点使用向量化操作
        x = np.atleast_2d(x)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(x, return_std=True)
        
        # 如果sigma非常小，返回较大值以减少在该点采样
        mask = sigma > 1e-6
        
        # 初始化结果为一个大值
        ei = np.zeros_like(mu)
        
        # 只对有效的sigma计算EI
        if np.any(mask):
            z = (y_best - mu[mask]) / sigma[mask]
            ei[mask] = (y_best - mu[mask]) * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        
        if len(ei) == 1:
            return -ei[0]  # 单点情况
        return -ei  # 批量情况
    
    def _simplified_acquisition(self, x, gp, y_best):
        """
        更简单的采集函数，在高维度或大规模数据时速度更快。
        结合了UCB和预期均值，避免了计算密集的EI计算。
        """
        x = np.atleast_2d(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = gp.predict(x, return_std=True)
        
        # 简单的权重组合，平衡探索和利用
        # 小的mu（均值）和大的sigma（不确定性）给出更好的得分
        score = mu - self.kappa * sigma
        
        if len(score) == 1:
            return score[0]  # 单点情况
        return score  # 批量情况
    
    def _very_fast_propose_samples(self, n_samples=1):
        """
        极速版提出采样点方法 - 牺牲精度换取速度
        
        Args:
            n_samples: 要提出的采样点数量
            
        Returns:
            下一批采样点
        """
        if self.samples_since_last_fit >= self.fit_frequency or self.samples_since_last_fit == 0:
            # 如果达到重新拟合频率，重新拟合模型
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.gp.fit(self.X_samples, self.y_samples)
            except Exception as e:
                print(f"模型拟合错误: {e}")
                # 如果拟合失败，保持原模型不变
            self.samples_since_last_fit = 0
        
        # 当前最佳函数值
        y_best = np.min(self.y_samples)
        
        # 生成随机候选点 - 数量大幅减少
        n_random_points = self.n_random_starts
        X_random = np.random.uniform(-self.x_max, self.x_max, (n_random_points, self.dim))
        
        # 使用当前最佳点和其附近点
        if self.best_position is not None:
            X_near_best = self.best_position + np.random.normal(0, 0.05 * self.x_max, (5, self.dim))
            X_near_best = np.clip(X_near_best, -self.x_max, self.x_max)
            X_random = np.vstack([X_random, self.best_position.reshape(1, -1), X_near_best])
        
        # 批量评估采集函数
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.use_simplified_gp:
                acq_values = self._simplified_acquisition(X_random, self.gp, y_best)
                # 对于简化版，我们需要最小化而非最大化
                best_indices = np.argsort(acq_values)[:min(n_samples*2, 10)]
            else:
                acq_values = []
                for x in X_random:
                    acq_values.append(self._expected_improvement(x, self.gp, y_best))
                acq_values = np.array(acq_values)
                best_indices = np.argsort(acq_values)[:min(n_samples*2, 10)]
        
        # 完全跳过局部搜索以更快
        if n_samples >= len(best_indices):
            # 如果需要的样本数大于等于候选点数，直接返回所有候选点
            return X_random[best_indices]
        
        # 否则选择前n_samples个点
        return X_random[best_indices[:n_samples]]
    
    def update(self):
        """
        Update the Bayesian Optimization algorithm for all iterations.
        
        Returns:
            A dictionary containing the optimization results
        """
        x_list = [self.X_samples.copy()]
        fit_list = [self.y_samples.copy()]
        
        # 使用批处理加速优化过程
        for i in range(0, self.iter_num, self.batch_size):
            # 计算当前批次的实际大小
            current_batch_size = min(self.batch_size, self.iter_num - i)
            
            # 提出多个采样点
            next_points = self._very_fast_propose_samples(current_batch_size)
            if current_batch_size == 1:
                next_points = np.array([next_points])
            
            # 批量评估所提出的点
            next_values = np.zeros(current_batch_size)
            for j in range(current_batch_size):
                self.config['x'] = next_points[j]
                next_values[j] = self.plm.test(self.config)
            
            # 更新数据集
            self.X_samples = np.vstack((self.X_samples, next_points))
            self.y_samples = np.append(self.y_samples, next_values)
            
            # 更新计数器
            self.samples_since_last_fit += current_batch_size
            
            # 更新最佳解
            min_idx = np.argmin(next_values)
            if next_values[min_idx] < self.best_fitness_value:
                self.best_position = next_points[min_idx].copy()
                self.best_fitness_value = next_values[min_idx]
            
            # 存储当前代
            x_list.append(self.X_samples.copy())
            fit_list.append(self.y_samples.copy())
        
        # 构建返回字典
        return_dict = {
            'x_list': x_list,
            'fit_list': fit_list,
            'best_x': self.best_position,
            'best_f': self.best_fitness_value,
        }
        
        return return_dict
    
    def _upper_confidence_bound(self, x, gp):
        """
        Calculate the Upper Confidence Bound acquisition function.
        
        Args:
            x: The point to evaluate
            gp: Gaussian process model
            
        Returns:
            The negative UCB (negative because we minimize)
        """
        x = x.reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        
        # UCB = mu - kappa * sigma (negative because we're minimizing)
        return -(mu - self.kappa * sigma)[0]
    
    # 为了保持与原来的接口兼容，保留原来的_propose_next_sample方法
    def _propose_next_sample(self):
        """
        Propose the next sampling point by optimizing the acquisition function.
        
        Returns:
            The next point to sample
        """
        return self._very_fast_propose_samples(1)