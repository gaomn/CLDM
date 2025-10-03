import math
import itertools
import random
import numpy as np
from typing import List, Tuple, Callable, Optional, Union, Dict, Any

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def euclidean_distance_squared(point1: List[float], point2: List[float]) -> float:
    """计算两点之间的欧氏距离平方"""
    return sum((x - p) ** 2 for x, p in zip(point1, point2))


def cone(individual: List[float], position: List[float], height: float, width: float) -> float:
    """锥形峰函数，用于场景2和场景3
    
    :math:`f(\mathbf{x}) = h - w \sqrt{\sum_{i=1}^N (x_i - p_i)^2}`
    """
    distance_squared = euclidean_distance_squared(individual, position)
    return height - width * math.sqrt(distance_squared)


def sphere(individual: List[float], position: List[float], height: float, width: float) -> float:
    """球形函数"""
    distance_squared = euclidean_distance_squared(individual, position)
    return height * distance_squared


def function1(individual: List[float], position: List[float], height: float, width: float) -> float:
    """峰函数1，用于场景1
    
    :math:`f(\mathbf{x}) = \\frac{h}{1 + w \sqrt{\sum_{i=1}^N (x_i - p_i)^2}}`
    """
    distance_squared = euclidean_distance_squared(individual, position)
    return height / (1 + width * distance_squared)


class MovingPeaks:
    """移动峰基准问题实现"""

    def __init__(self, dim: int, random=random, **kargs):
        """初始化移动峰基准问题
        
        Args:
            dim: 问题维度
            random: 随机数生成器
            **kargs: 其他配置参数
        """
        # 复制默认场景并更新配置
        sc = SCENARIO.copy()
        sc.update(kargs)

        self.args = sc.get("args")
        pfunc = sc.get("pfunc")
        npeaks = self.args.peak_num

        # 设置随机数生成器
        random.seed(self.args.MPB_seed)
        self.random = random
        self.dim = dim

        # 初始化峰的数量范围
        self.minpeaks, self.maxpeaks = None, None
        if hasattr(npeaks, "__getitem__"):
            self.minpeaks, npeaks, self.maxpeaks = npeaks
            self.number_severity = sc.get("number_severity")

        # 设置峰函数
        self._setup_peak_functions(pfunc, npeaks)
        
        # 设置基础配置
        self.basis_function = sc.get("bfunc")
        self._setup_coordinate_bounds(sc)
        self._setup_height_params(sc, npeaks)
        self._setup_width_params(sc, npeaks)
        
        # 设置变化参数
        self.lambda_ = sc.get("lambda_")
        self.move_severity = sc.get("move_severity")
        self.height_severity = sc.get("height_severity")
        self.width_severity = sc.get("width_severity")
        
        # 设置时间因子
        self._setup_time_factor()
        
        # 初始化峰位置、高度和宽度
        self._initialize_peaks(npeaks)
        
        # 用于离线错误计算
        self.period = sc.get("period")
        self._optimum = None
        self._error = None
        self._offline_error = 0
        self.nevals = 0

    def _setup_peak_functions(self, pfunc, npeaks):
        """设置峰函数"""
        try:
            if len(pfunc) == npeaks:
                self.peaks_function = pfunc
            else:
                self.peaks_function = self.random.sample(pfunc, npeaks)
            self.pfunc_pool = tuple(pfunc)
        except TypeError:
            self.peaks_function = list(itertools.repeat(pfunc, npeaks))
            self.pfunc_pool = (pfunc,)

    def _setup_coordinate_bounds(self, sc):
        """设置坐标边界"""
        self.min_coord = sc.get("min_coord")
        self.max_coord = sc.get("max_coord")

    def _setup_height_params(self, sc, npeaks):
        """设置高度参数"""
        self.min_height = sc.get("min_height")
        self.max_height = sc.get("max_height")
        self.uniform_height = sc.get("uniform_height")

    def _setup_width_params(self, sc, npeaks):
        """设置宽度参数"""
        self.min_width = sc.get("min_width")
        self.max_width = sc.get("max_width")
        self.uniform_width = sc.get("uniform_width")

    def _setup_time_factor(self):
        """设置时间因子"""
        try:
            self.time_fac = self.args.time_fac
        except AttributeError:
            print('No time-fac!!')
            self.time_fac = 0.
        self.time_fac_sym = 0.

    def _initialize_peaks(self, npeaks):
        """初始化峰位置、高度和宽度"""
        # 初始化峰位置
        self.peaks_position = [
            [self.random.uniform(self.min_coord, self.max_coord) for _ in range(self.dim)]
            for _ in range(npeaks)
        ]
        
        # 初始化峰高度
        if self.uniform_height != 0:
            self.peaks_height = [self.uniform_height for _ in range(npeaks)]
        else:
            self.peaks_height = [
                self.random.uniform(self.min_height, self.max_height) 
                for _ in range(npeaks)
            ]
        
        # 初始化峰宽度
        if self.uniform_width != 0:
            self.peaks_width = [self.uniform_width for _ in range(npeaks)]
        else:
            self.peaks_width = [
                self.random.uniform(self.min_width, self.max_width) 
                for _ in range(npeaks)
            ]
        
        # 初始化变化向量
        self.last_change_vector = [
            [self.random.random() - 0.5 for _ in range(self.dim)] 
            for _ in range(npeaks)
        ]

    def globalMaximum(self) -> Tuple[float, List[float]]:
        """返回全局最大值及其位置"""
        potential_max = [
            (func(pos, pos, height, width), pos)
            for func, pos, height, width in zip(
                self.peaks_function, self.peaks_position, 
                self.peaks_height, self.peaks_width
            )
        ]
        return max(potential_max)

    def get_h(self) -> float:
        """返回第一个峰的高度"""
        return self.peaks_height[0]

    def maximums(self) -> List[Tuple[float, List[float]]]:
        """返回所有可见的最大值及其位置，按全局最大值优先排序"""
        maximums = []
        for func, pos, height, width in zip(
            self.peaks_function, self.peaks_position, 
            self.peaks_height, self.peaks_width
        ):
            val = func(pos, pos, height, width) + self.time_fac * self.time_fac_sym
            if val >= self.__call__(pos, count=False)[0]:
                maximums.append((val, pos))
        return sorted(maximums, reverse=True)

    def __call__(self, individual, count=False) -> Tuple[float]:
        """评估给定个体在当前基准配置下的适应度
        
        Args:
            individual: 待评估的个体
            count: 是否将此评估计入总评估次数（默认为True）
        
        Returns:
            包含适应度值的元组
        """
        possible_values = [
            func(individual, pos, height, width)
            for func, pos, height, width in zip(
                self.peaks_function, self.peaks_position, 
                self.peaks_height, self.peaks_width
            )
        ]

        if self.basis_function:
            possible_values.append(self.basis_function(individual))

        fitness = max(possible_values) + self.time_fac * self.time_fac_sym

        if count:
            self._update_error_metrics(fitness)
            self._check_period_change()

        return (fitness,)

    def test(self, individual_dict: Dict[str, List[float]], count=False) -> float:
        """评估给定个体字典在当前基准配置下的适应度
        
        Args:
            individual_dict: 包含'x'键的字典，其值为待评估的个体
            count: 是否将此评估计入总评估次数（默认为True）
            
        Returns:
            适应度值
        """
        individual = individual_dict['x']
        possible_values = [
            func(individual, pos, height, width)
            for func, pos, height, width in zip(
                self.peaks_function, self.peaks_position, 
                self.peaks_height, self.peaks_width
            )
        ]

        if self.basis_function:
            possible_values.append(self.basis_function(individual))

        fitness = max(possible_values) + self.time_fac * self.time_fac_sym

        if count:
            self._update_error_metrics(fitness)
            self._check_period_change()
            
        # 对于连续且为max类型的情况，适应值添加噪声
        if self.args.bt_change == 'continuous' and (self.args.bt_type == 'max'):
            fitness += self.random.normalvariate(0, 5)

        return fitness

    def _update_error_metrics(self, fitness: float) -> None:
        """更新错误度量"""
        self.nevals += 1
        if self._optimum is None:
            self._optimum = self.globalMaximum()[0]
            self._error = abs(fitness - self._optimum)
        self._error = min(self._error, abs(fitness - self._optimum))
        self._offline_error += self._error

    def _check_period_change(self) -> None:
        """检查是否需要更改峰"""
        if self.period > 0 and self.nevals % self.period == 0:
            self.changePeaks()

    def offlineError(self) -> float:
        """返回离线错误"""
        return self._offline_error / self.nevals

    def currentError(self) -> float:
        """返回当前错误"""
        return self._error

    def changePeaks(self, individual=None) -> None:
        """改变峰的位置、高度、宽度和数量"""
        self._change_number_of_peaks()
        self._change_peak_properties()
        self._update_time_factor(individual)
        self._optimum = None

    def _change_number_of_peaks(self) -> None:
        """改变峰的数量"""
        if self.minpeaks is None or self.maxpeaks is None:
            return
            
        npeaks = len(self.peaks_function)
        u = self.random.random()
        r = self.maxpeaks - self.minpeaks
        
        if u < 0.5:
            # 减少峰的数量
            self._remove_peaks(npeaks, r)
        else:
            # 增加峰的数量
            self._add_peaks(npeaks, r)

    def _remove_peaks(self, npeaks: int, r: int) -> None:
        """减少峰的数量"""
        u = self.random.random()
        n = min(npeaks - self.minpeaks, int(round(r * u * self.number_severity)))
        for _ in range(n):
            idx = self.random.randrange(len(self.peaks_function))
            self.peaks_function.pop(idx)
            self.peaks_position.pop(idx)
            self.peaks_height.pop(idx)
            self.peaks_width.pop(idx)
            self.last_change_vector.pop(idx)

    def _add_peaks(self, npeaks: int, r: int) -> None:
        """增加峰的数量"""
        u = self.random.random()
        n = min(self.maxpeaks - npeaks, int(round(r * u * self.number_severity)))
        for _ in range(n):
            self.peaks_function.append(self.random.choice(self.pfunc_pool))
            self.peaks_position.append(
                [self.random.uniform(self.min_coord, self.max_coord) for _ in range(self.dim)]
            )
            self.peaks_height.append(self.random.uniform(self.min_height, self.max_height))
            self.peaks_width.append(self.random.uniform(self.min_width, self.max_width))
            self.last_change_vector.append([self.random.random() - 0.5 for _ in range(self.dim)])

    def _change_peak_properties(self) -> None:
        """改变峰的属性（位置、高度、宽度）"""
        for i in range(len(self.peaks_function)):
            self._change_peak_position(i)
            self._change_peak_height(i)
            self._change_peak_width(i)

    def _change_peak_position(self, i: int) -> None:
        """改变单个峰的位置"""
        # 生成移动向量
        shift = [self.random.random() - 0.5 for _ in range(len(self.peaks_position[i]))]
        shift_length = sum(s ** 2 for s in shift)
        shift_length = self.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0

        # 结合上次的变化向量
        shift = [
            shift_length * (1.0 - self.lambda_) * s + self.lambda_ * c 
            for s, c in zip(shift, self.last_change_vector[i])
        ]

        # 规范化移动向量的长度
        shift_length = sum(s ** 2 for s in shift)
        shift_length = self.move_severity / math.sqrt(shift_length) if shift_length > 0 else 0
        shift = [s * shift_length for s in shift]

        # 应用移动，确保在边界内
        new_position = []
        final_shift = []
        for pp, s in zip(self.peaks_position[i], shift):
            new_coord = pp + s
            if new_coord < self.min_coord:
                new_position.append(2.0 * self.min_coord - pp - s)
                final_shift.append(-1.0 * s)
            elif new_coord > self.max_coord:
                new_position.append(2.0 * self.max_coord - pp - s)
                final_shift.append(-1.0 * s)
            else:
                new_position.append(new_coord)
                final_shift.append(s)

        self.peaks_position[i] = new_position
        self.last_change_vector[i] = final_shift

    def _change_peak_height(self, i: int) -> None:
        """改变单个峰的高度"""
        change = self.random.gauss(0, 1) * self.height_severity
        new_value = change + self.peaks_height[i]
        
        if new_value < self.min_height:
            self.peaks_height[i] = 2.0 * self.min_height - self.peaks_height[i] - change
        elif new_value > self.max_height:
            self.peaks_height[i] = 2.0 * self.max_height - self.peaks_height[i] - change
        else:
            self.peaks_height[i] = new_value

    def _change_peak_width(self, i: int) -> None:
        """改变单个峰的宽度"""
        change = self.random.gauss(0, 1) * self.width_severity
        new_value = change + self.peaks_width[i]
        
        if new_value < self.min_width:
            self.peaks_width[i] = 2.0 * self.min_width - self.peaks_width[i] - change
        elif new_value > self.max_width:
            self.peaks_width[i] = 2.0 * self.max_width - self.peaks_width[i] - change
        else:
            self.peaks_width[i] = new_value

    def _update_time_factor(self, individual) -> None:
        """更新时间因子"""
        if self.args.bt_change == 'continuous':
            self.time_fac_sym = self.get_sym_con(individual)
        else:
            self.time_fac_sym = self.get_sym_dis(individual)

    def get_sym_dis(self, x) -> int:
        """获取离散对称符号
        
        支持的类型:
        - linear: 线性分界
        - sin: 正弦分界
        - cir: 圆形分界
        - rect: 矩形分界
        - linear4, sin4, cir4: 4级分界
        """
        if x is None:
            return 0
            
        bt_type = self.args.bt_type
        xb = self.args.x_bound
        
        if bt_type == 'linear':
            return 1 if x[0] >= 0. else -1
        elif bt_type == 'sin':
            return 1 if 2 * np.sin(0.2 * np.pi * x[0]) >= x[1] else -1
        elif bt_type == 'cir':
            return 1 if x[0] ** 2 + x[1] ** 2 <= 2 * xb**2 / np.pi else -1
        elif bt_type == 'rect':
            return 1 if -3.54 <= x[0] <= 3.54 and -3.54 <= x[1] <= 3.54 else -1
        elif bt_type == 'linear4':
            if x[0] >= xb/2.:
                return 2
            elif x[0] >= 0:
                return 1
            elif x[0] >= -xb/2.:
                return -1
            else:
                return -2
        elif bt_type == 'sin4':
            sin_val = 2 * np.sin(0.2 * np.pi * x[0])
            if sin_val >= x[1] + xb/2.:
                return 2
            elif sin_val >= x[1]:
                return 1
            elif sin_val <= x[1] - xb/2.:
                return -2
            else:
                return -1
        elif bt_type == 'cir4':
            distance = x[0] ** 2 + x[1] ** 2
            if distance <= xb**2 / np.pi:
                return 2
            elif distance <= 2 * xb**2 / np.pi:
                return 1
            elif distance <= 3 * xb**2 / np.pi:
                return -1
            else:
                return -2
        else:
            raise ValueError(f"未知的bt_type: {bt_type}. 请确保bt_type为: 'linear', 'sin', 'cir', 'rect'等支持的类型")

    def get_sym_con(self, x) -> float:
        """获取连续对称符号
        
        支持的类型:
        - linear: 线性函数
        - sin: 正弦函数
        - cir: 圆形函数
        - tanh: 双曲正切函数
        - max: 最大值函数
        """
        if x is None:
            return 0.0
            
        bt_type = self.args.bt_type
        
        if bt_type == 'linear':
            return x[0]/5.
        elif bt_type == 'sin':
            return (x[1] - 2 * np.sin(0.2 * np.pi * x[0]))/5.
        elif bt_type == 'cir':
            return (x[0] ** 2 + x[1] ** 2)/25.
        elif bt_type == 'tanh':
            return np.tanh(x[0])/5. + self.random.normalvariate(0, 0.1)
        elif bt_type == 'max':
            return np.maximum(x[0], x[1])/5.
        else:
            raise ValueError(f"未知的bt_type: {bt_type}. 请确保bt_type为: 'linear', 'sin', 'cir'等支持的类型")


# 默认场景配置
SCENARIO = {
    "pfunc": cone,
    "npeaks": 1,
    "bfunc": None,
    "min_coord": -5.,
    "max_coord": 5.0,
    "min_height": 30.,
    "max_height": 70.,
    "uniform_height": 50.0,
    "min_width": 1.0,
    "max_width": 12.0,
    "uniform_width": 5,
    "lambda_": 0.5,
    "move_severity": 7.0,
    "height_severity": 7.0,
    "width_severity": 1.0,
    "period": 5000,
    "time_fac": 100
}


def diversity(population: List[List[float]]) -> float:
    """计算群体多样性
    
    Args:
        population: 个体列表，每个个体是一个坐标列表
        
    Returns:
        群体多样性度量
    """
    nind = len(population)
    ndim = len(population[0])
    
    # 计算每个维度的中心点
    centroid = [0.0] * ndim
    for x in population:
        centroid = [di + xi for di, xi in zip(centroid, x)]
    centroid = [di / nind for di in centroid]
    
    # 计算每个个体到中心点的平均距离
    return math.sqrt(sum((di - xi) ** 2 for x in population for di, xi in zip(centroid, x)))


# 示例使用代码
# if __name__ == "__main__":
#     rnd = random.Random()
#     rnd.seed(1254)
#     mpb = MovingPeaks(dim=10, number_severity=0.1, random=rnd)
#     res = mpb([0., 0.])
#     print(f'res : {res}   type : {type(res[0])} ')
#
#     for i in range(100):
#         m = mpb.maximums()[0]
#         print(f' {i}  ::  max_fit:{round(m[0], 2)}   x1_pos : {round(m[1][0], 2)} ')
#         mpb.changePeaks([0., 0.])