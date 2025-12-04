"""
基准测试函数集合

提供连续优化问题的标准测试函数，包括：
- 单峰函数：Sphere, Schwefel 2.22
- 多峰函数：Rastrigin, Ackley, Griewank
- 山谷函数：Rosenbrock

每个函数支持 (dim,) 或 (n, dim) 形状的输入，返回相应形状的函数值。
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class BenchmarkFunction:
    """基准测试函数封装类"""
    
    def __init__(
        self,
        name: str,
        func: Callable[[np.ndarray], np.ndarray],
        lower: float,
        upper: float,
        dim: int = 30,
        optimal_value: float = 0.0,
        description: str = "",
    ):
        """初始化基准函数
        
        Args:
            name: 函数名称
            func: 函数实现
            lower: 搜索空间下界
            upper: 搜索空间上界
            dim: 问题维度
            optimal_value: 全局最优值
            description: 函数描述
        """
        self.name = name
        self.func = func
        self.lower = lower
        self.upper = upper
        self.dim = dim
        self.optimal_value = optimal_value
        self.description = description

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """计算函数值"""
        return self.func(x)

    def __repr__(self) -> str:
        return f"BenchmarkFunction({self.name}, dim={self.dim}, bounds=[{self.lower}, {self.upper}])"


# =============================================================================
# 单峰函数（Unimodal Functions）
# =============================================================================

def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere 函数（球形函数）
    
    f(x) = sum(x_i^2)
    全局最优：f(0, ..., 0) = 0
    特点：简单的凸函数，常用于算法验证
    """
    x = np.asarray(x)
    return np.sum(x * x, axis=-1)


def schwefel_222(x: np.ndarray) -> np.ndarray:
    """Schwefel 2.22 函数
    
    f(x) = sum(|x_i|) + prod(|x_i|)
    全局最优：f(0, ..., 0) = 0
    特点：非光滑函数
    """
    x = np.asarray(x)
    abs_x = np.abs(x)
    return np.sum(abs_x, axis=-1) + np.prod(abs_x, axis=-1)


def schwefel_12(x: np.ndarray) -> np.ndarray:
    """Schwefel 1.2 函数
    
    f(x) = sum_{i=1}^{d} (sum_{j=1}^{i} x_j)^2
    全局最优：f(0, ..., 0) = 0
    特点：变量间强相关
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    cumsum = np.cumsum(x, axis=-1)
    result = np.sum(cumsum ** 2, axis=-1)
    return result.squeeze() if result.shape[0] == 1 else result


def sum_squares(x: np.ndarray) -> np.ndarray:
    """Sum of Squares 函数（加权球形函数）
    
    f(x) = sum(i * x_i^2)
    全局最优：f(0, ..., 0) = 0
    """
    x = np.asarray(x)
    d = x.shape[-1]
    weights = np.arange(1, d + 1)
    return np.sum(weights * x * x, axis=-1)


# =============================================================================
# 多峰函数（Multimodal Functions）
# =============================================================================

def rastrigin(x: np.ndarray) -> np.ndarray:
    """Rastrigin 函数
    
    f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))
    全局最优：f(0, ..., 0) = 0
    特点：高度多峰，有大量局部最优
    """
    x = np.asarray(x)
    d = x.shape[-1]
    return 10 * d + np.sum(x * x - 10 * np.cos(2 * np.pi * x), axis=-1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock 函数（香蕉函数）
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2)
    全局最优：f(1, ..., 1) = 0
    特点：山谷形状，优化路径弯曲
    """
    x = np.asarray(x)
    return np.sum(
        100.0 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (x[..., :-1] - 1) ** 2,
        axis=-1
    )


def ackley(x: np.ndarray) -> np.ndarray:
    """Ackley 函数
    
    f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/d)) - exp(sum(cos(2*pi*x_i))/d) + 20 + e
    全局最优：f(0, ..., 0) = 0
    特点：多峰 + 平坦区域
    """
    x = np.asarray(x)
    d = x.shape[-1]
    a, b, c = 20, 0.2, 2 * np.pi
    
    sum_sq = np.sum(x * x, axis=-1)
    sum_cos = np.sum(np.cos(c * x), axis=-1)
    
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    
    return term1 + term2 + a + np.e


def griewank(x: np.ndarray) -> np.ndarray:
    """Griewank 函数
    
    f(x) = sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))) + 1
    全局最优：f(0, ..., 0) = 0
    特点：多峰，高维时局部最优减少
    """
    x = np.asarray(x)
    d = x.shape[-1]
    
    sum_term = np.sum(x * x, axis=-1) / 4000
    
    # 处理 prod 项
    i_vals = np.sqrt(np.arange(1, d + 1))
    prod_term = np.prod(np.cos(x / i_vals), axis=-1)
    
    return sum_term - prod_term + 1


def levy(x: np.ndarray) -> np.ndarray:
    """Levy 函数
    
    全局最优：f(1, ..., 1) = 0
    特点：多峰
    """
    x = np.asarray(x)
    d = x.shape[-1]
    
    w = 1 + (x - 1) / 4
    
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    term2 = np.sum(
        (w[..., :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[..., :-1] + 1) ** 2),
        axis=-1
    )
    term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)
    
    return term1 + term2 + term3


def michalewicz(x: np.ndarray, m: float = 10) -> np.ndarray:
    """Michalewicz 函数
    
    全局最优取决于维度，约为 -d * 0.966
    特点：有多个深度不同的局部最优
    """
    x = np.asarray(x)
    d = x.shape[-1]
    
    i_vals = np.arange(1, d + 1)
    return -np.sum(
        np.sin(x) * np.sin(i_vals * x ** 2 / np.pi) ** (2 * m),
        axis=-1
    )


def schwefel(x: np.ndarray) -> np.ndarray:
    """Schwefel 函数
    
    f(x) = 418.9829*d - sum(x_i * sin(sqrt(|x_i|)))
    全局最优：f(420.9687, ..., 420.9687) ≈ 0
    特点：全局最优远离局部最优
    """
    x = np.asarray(x)
    d = x.shape[-1]
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=-1)


def alpine(x: np.ndarray) -> np.ndarray:
    """Alpine 函数
    
    f(x) = sum(|x_i * sin(x_i) + 0.1 * x_i|)
    全局最优：f(0, ..., 0) = 0
    特点：多峰，非对称
    """
    x = np.asarray(x)
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=-1)


def styblinski_tang(x: np.ndarray) -> np.ndarray:
    """Styblinski-Tang 函数
    
    f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)
    全局最优：f(-2.903534, ..., -2.903534) ≈ -39.16599 * d
    """
    x = np.asarray(x)
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=-1)


# =============================================================================
# 基准函数注册表
# =============================================================================

BENCHMARKS = {
    # 单峰函数
    "sphere": BenchmarkFunction(
        name="sphere",
        func=sphere,
        lower=-100.0,
        upper=100.0,
        optimal_value=0.0,
        description="球形函数（单峰，简单凸函数）",
    ),
    "schwefel_222": BenchmarkFunction(
        name="schwefel_222",
        func=schwefel_222,
        lower=-10.0,
        upper=10.0,
        optimal_value=0.0,
        description="Schwefel 2.22 函数（非光滑）",
    ),
    "sum_squares": BenchmarkFunction(
        name="sum_squares",
        func=sum_squares,
        lower=-10.0,
        upper=10.0,
        optimal_value=0.0,
        description="加权球形函数",
    ),
    
    # 多峰函数
    "rastrigin": BenchmarkFunction(
        name="rastrigin",
        func=rastrigin,
        lower=-5.12,
        upper=5.12,
        optimal_value=0.0,
        description="Rastrigin 函数（高度多峰）",
    ),
    "rosenbrock": BenchmarkFunction(
        name="rosenbrock",
        func=rosenbrock,
        lower=-30.0,
        upper=30.0,
        optimal_value=0.0,
        description="Rosenbrock 香蕉函数（山谷形状）",
    ),
    "ackley": BenchmarkFunction(
        name="ackley",
        func=ackley,
        lower=-32.0,
        upper=32.0,
        optimal_value=0.0,
        description="Ackley 函数（多峰+平坦区域）",
    ),
    "griewank": BenchmarkFunction(
        name="griewank",
        func=griewank,
        lower=-600.0,
        upper=600.0,
        optimal_value=0.0,
        description="Griewank 函数（多峰）",
    ),
    "levy": BenchmarkFunction(
        name="levy",
        func=levy,
        lower=-10.0,
        upper=10.0,
        optimal_value=0.0,
        description="Levy 函数（多峰）",
    ),
    "schwefel": BenchmarkFunction(
        name="schwefel",
        func=schwefel,
        lower=-500.0,
        upper=500.0,
        optimal_value=0.0,  # 实际约为 0，但最优点不在原点
        description="Schwefel 函数（全局最优远离局部最优）",
    ),
    "alpine": BenchmarkFunction(
        name="alpine",
        func=alpine,
        lower=-10.0,
        upper=10.0,
        optimal_value=0.0,
        description="Alpine 函数（多峰，非对称）",
    ),
    "styblinski_tang": BenchmarkFunction(
        name="styblinski_tang",
        func=styblinski_tang,
        lower=-5.0,
        upper=5.0,
        optimal_value=-39.16599 * 30,  # 30维时的最优值
        description="Styblinski-Tang 函数",
    ),
}


# 按类别分组的基准函数
UNIMODAL_BENCHMARKS = ["sphere", "schwefel_222", "sum_squares"]
MULTIMODAL_BENCHMARKS = ["rastrigin", "rosenbrock", "ackley", "griewank", "levy", "schwefel", "alpine", "styblinski_tang"]


def get_benchmark(name: str, dim: int = 30) -> BenchmarkFunction:
    """获取指定的基准函数
    
    Args:
        name: 函数名称
        dim: 问题维度
    
    Returns:
        BenchmarkFunction 实例
    
    Raises:
        ValueError: 如果函数名称无效
    """
    if name not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"未知基准函数: {name}。可用函数: {available}")
    
    bench = BENCHMARKS[name]
    # 创建新实例并设置维度
    return BenchmarkFunction(
        name=bench.name,
        func=bench.func,
        lower=bench.lower,
        upper=bench.upper,
        dim=dim,
        optimal_value=bench.optimal_value,
        description=bench.description,
    )


def list_benchmarks() -> None:
    """打印所有可用的基准函数"""
    print("\n可用的基准测试函数：")
    print("=" * 60)
    
    print("\n【单峰函数】")
    for name in UNIMODAL_BENCHMARKS:
        bench = BENCHMARKS[name]
        print(f"  - {name}: {bench.description}")
        print(f"    bounds=[{bench.lower}, {bench.upper}], optimal={bench.optimal_value}")
    
    print("\n【多峰函数】")
    for name in MULTIMODAL_BENCHMARKS:
        bench = BENCHMARKS[name]
        print(f"  - {name}: {bench.description}")
        print(f"    bounds=[{bench.lower}, {bench.upper}], optimal={bench.optimal_value}")
    
    print("=" * 60)
