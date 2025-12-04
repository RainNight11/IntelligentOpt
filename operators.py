"""
遗传算子：选择、交叉、变异

提供向量化的 numpy 实现，支持批量操作以提升性能。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# =============================================================================
# 选择算子
# =============================================================================

def roulette_wheel_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    num_parents: int,
) -> np.ndarray:
    """轮盘赌选择（适用于最小化问题）
    
    Args:
        population: 种群 (pop_size, dim)
        fitness: 适应度值 (pop_size,)
        num_parents: 需要选择的父代数量
    
    Returns:
        选中的父代个体 (num_parents, dim)
    """
    # 最小化问题：适应度越小，选择概率越大
    inv_f = 1.0 / (1.0 + fitness - fitness.min() + 1e-12)
    probs = inv_f / inv_f.sum()
    idx = np.random.choice(len(population), size=num_parents, replace=True, p=probs)
    return population[idx]


def tournament_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    num_parents: int,
    k: int = 3,
) -> np.ndarray:
    """锦标赛选择
    
    Args:
        population: 种群 (pop_size, dim)
        fitness: 适应度值 (pop_size,)
        num_parents: 需要选择的父代数量
        k: 锦标赛候选个体数量
    
    Returns:
        选中的父代个体 (num_parents, dim)
    """
    pop_size = len(population)
    # 向量化：一次性生成所有锦标赛的候选索引
    candidates = np.random.randint(0, pop_size, size=(num_parents, k))
    # 获取每个锦标赛中适应度最小的个体索引
    candidate_fitness = fitness[candidates]  # (num_parents, k)
    best_in_tournament = np.argmin(candidate_fitness, axis=1)  # (num_parents,)
    selected_idx = candidates[np.arange(num_parents), best_in_tournament]
    return population[selected_idx]


def rank_selection(
    population: np.ndarray,
    fitness: np.ndarray,
    num_parents: int,
) -> np.ndarray:
    """排序选择
    
    Args:
        population: 种群 (pop_size, dim)
        fitness: 适应度值 (pop_size,)
        num_parents: 需要选择的父代数量
    
    Returns:
        选中的父代个体 (num_parents, dim)
    """
    pop_size = len(population)
    ranks = np.argsort(np.argsort(fitness))  # 排名：0 是最好的
    # 概率与排名成反比
    probs = (pop_size - ranks).astype(float)
    probs /= probs.sum()
    idx = np.random.choice(pop_size, size=num_parents, replace=True, p=probs)
    return population[idx]


# =============================================================================
# 交叉算子
# =============================================================================

def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    p_c: float,
) -> np.ndarray:
    """均匀交叉（单个体）
    
    Args:
        parent1: 父代1 (dim,)
        parent2: 父代2 (dim,)
        p_c: 交叉概率
    
    Returns:
        子代个体 (dim,)
    """
    if np.random.rand() > p_c:
        return parent1.copy()
    mask = np.random.rand(parent1.shape[0]) < 0.5
    return np.where(mask, parent1, parent2)


def uniform_crossover_batch(
    parents1: np.ndarray,
    parents2: np.ndarray,
    p_c: float,
) -> np.ndarray:
    """均匀交叉（批量向量化版本）
    
    Args:
        parents1: 父代1组 (n, dim)
        parents2: 父代2组 (n, dim)
        p_c: 交叉概率
    
    Returns:
        子代个体组 (n, dim)
    """
    n, dim = parents1.shape
    # 决定哪些个体进行交叉
    do_crossover = np.random.rand(n) < p_c
    # 生成交叉掩码
    mask = np.random.rand(n, dim) < 0.5
    # 默认返回 parent1，交叉的位置使用 mask 混合
    children = parents1.copy()
    children[do_crossover] = np.where(
        mask[do_crossover],
        parents1[do_crossover],
        parents2[do_crossover]
    )
    return children


def two_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    p_c: float,
) -> np.ndarray:
    """两点交叉（单个体）
    
    Args:
        parent1: 父代1 (dim,)
        parent2: 父代2 (dim,)
        p_c: 交叉概率
    
    Returns:
        子代个体 (dim,)
    """
    if np.random.rand() > p_c:
        return parent1.copy()
    d = parent1.shape[0]
    pt1, pt2 = sorted(np.random.choice(d, size=2, replace=False))
    child = parent1.copy()
    child[pt1:pt2] = parent2[pt1:pt2]
    return child


def two_point_crossover_batch(
    parents1: np.ndarray,
    parents2: np.ndarray,
    p_c: float,
) -> np.ndarray:
    """两点交叉（批量向量化版本）
    
    Args:
        parents1: 父代1组 (n, dim)
        parents2: 父代2组 (n, dim)
        p_c: 交叉概率
    
    Returns:
        子代个体组 (n, dim)
    """
    n, dim = parents1.shape
    children = parents1.copy()
    
    # 决定哪些个体进行交叉
    do_crossover = np.random.rand(n) < p_c
    num_crossover = do_crossover.sum()
    
    if num_crossover > 0:
        # 为每个交叉生成两个交叉点
        points = np.sort(np.random.randint(0, dim, size=(num_crossover, 2)), axis=1)
        # 创建掩码
        idx = np.arange(dim)
        masks = (idx >= points[:, 0:1]) & (idx < points[:, 1:2])  # (num_crossover, dim)
        # 应用交叉
        children[do_crossover] = np.where(
            masks,
            parents2[do_crossover],
            parents1[do_crossover]
        )
    
    return children


# =============================================================================
# 变异算子
# =============================================================================

def uniform_mutation(
    vector: np.ndarray,
    p_m: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """均匀变异（单个体）
    
    Args:
        vector: 个体 (dim,)
        p_m: 变异概率
        lower: 下界
        upper: 上界
    
    Returns:
        变异后的个体 (dim,)
    """
    mask = np.random.rand(vector.shape[0]) < p_m
    random_vals = np.random.uniform(lower, upper, size=vector.shape[0])
    result = vector.copy()
    result[mask] = random_vals[mask]
    return np.clip(result, lower, upper)


def uniform_mutation_batch(
    population: np.ndarray,
    p_m: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    """均匀变异（批量向量化版本）
    
    Args:
        population: 种群 (n, dim)
        p_m: 变异概率
        lower: 下界
        upper: 上界
    
    Returns:
        变异后的种群 (n, dim)
    """
    mask = np.random.rand(*population.shape) < p_m
    random_vals = np.random.uniform(lower, upper, size=population.shape)
    result = population.copy()
    result[mask] = random_vals[mask]
    return np.clip(result, lower, upper)


def gaussian_mutation(
    vector: np.ndarray,
    p_m: float,
    lower: float,
    upper: float,
    sigma: float = 0.1,
) -> np.ndarray:
    """高斯变异（单个体）
    
    Args:
        vector: 个体 (dim,)
        p_m: 变异概率
        lower: 下界
        upper: 上界
        sigma: 标准差系数，实际 std = sigma * (upper - lower)
    
    Returns:
        变异后的个体 (dim,)
    """
    mask = np.random.rand(vector.shape[0]) < p_m
    noise = np.random.randn(vector.shape[0]) * sigma * (upper - lower)
    result = vector.copy()
    result[mask] += noise[mask]
    return np.clip(result, lower, upper)


def gaussian_mutation_batch(
    population: np.ndarray,
    p_m: float,
    lower: float,
    upper: float,
    sigma: float = 0.1,
) -> np.ndarray:
    """高斯变异（批量向量化版本）
    
    Args:
        population: 种群 (n, dim)
        p_m: 变异概率
        lower: 下界
        upper: 上界
        sigma: 标准差系数
    
    Returns:
        变异后的种群 (n, dim)
    """
    mask = np.random.rand(*population.shape) < p_m
    noise = np.random.randn(*population.shape) * sigma * (upper - lower)
    result = population.copy()
    result[mask] += noise[mask]
    return np.clip(result, lower, upper)


def adaptive_gaussian_mutation(
    vector: np.ndarray,
    p_m: float,
    lower: float,
    upper: float,
    diversity: float,
) -> np.ndarray:
    """自适应高斯变异（单个体）
    
    根据种群多样性动态调整变异强度：
    - 多样性高时：减小变异强度，加强开发
    - 多样性低时：增大变异强度，加强探索
    
    Args:
        vector: 个体 (dim,)
        p_m: 变异概率
        lower: 下界
        upper: 上界
        diversity: 归一化的种群多样性 [0, 1]
    
    Returns:
        变异后的个体 (dim,)
    """
    # 多样性低时增大 sigma，多样性高时减小 sigma
    sigma_scale = max(0.05, min(0.5, diversity * 1.5))
    return gaussian_mutation(vector, p_m, lower, upper, sigma=sigma_scale)


def adaptive_gaussian_mutation_batch(
    population: np.ndarray,
    p_m: float,
    lower: float,
    upper: float,
    diversity: float,
) -> np.ndarray:
    """自适应高斯变异（批量向量化版本）
    
    Args:
        population: 种群 (n, dim)
        p_m: 变异概率
        lower: 下界
        upper: 上界
        diversity: 归一化的种群多样性 [0, 1]
    
    Returns:
        变异后的种群 (n, dim)
    """
    sigma_scale = max(0.05, min(0.5, diversity * 1.5))
    return gaussian_mutation_batch(population, p_m, lower, upper, sigma=sigma_scale)


# =============================================================================
# 批量繁殖（向量化）
# =============================================================================

def batch_reproduce(
    population: np.ndarray,
    fitness: np.ndarray,
    p_c: float,
    p_m: float,
    lower: float,
    upper: float,
    selection_fn: str = "tournament",
    crossover_fn: str = "two_point",
    mutation_fn: str = "gaussian",
    sigma: float = 0.1,
    tournament_k: int = 3,
    diversity: float = 0.5,
) -> np.ndarray:
    """批量繁殖新种群（完全向量化）
    
    Args:
        population: 当前种群 (pop_size, dim)
        fitness: 适应度 (pop_size,)
        p_c: 交叉概率
        p_m: 变异概率
        lower: 下界
        upper: 上界
        selection_fn: 选择算子名称
        crossover_fn: 交叉算子名称
        mutation_fn: 变异算子名称
        sigma: 高斯变异标准差系数
        tournament_k: 锦标赛 k 值
        diversity: 种群多样性（用于自适应变异）
    
    Returns:
        新种群 (pop_size, dim)
    """
    pop_size = population.shape[0]
    
    # 选择父代
    selection_map = {
        "roulette": roulette_wheel_selection,
        "tournament": lambda p, f, n: tournament_selection(p, f, n, k=tournament_k),
        "rank": rank_selection,
    }
    select = selection_map.get(selection_fn, selection_map["tournament"])
    
    parents1 = select(population, fitness, pop_size)
    parents2 = select(population, fitness, pop_size)
    
    # 交叉
    crossover_map = {
        "uniform": uniform_crossover_batch,
        "two_point": two_point_crossover_batch,
    }
    crossover = crossover_map.get(crossover_fn, crossover_map["two_point"])
    children = crossover(parents1, parents2, p_c)
    
    # 变异
    mutation_map = {
        "uniform": lambda p: uniform_mutation_batch(p, p_m, lower, upper),
        "gaussian": lambda p: gaussian_mutation_batch(p, p_m, lower, upper, sigma=sigma),
        "adaptive_gaussian": lambda p: adaptive_gaussian_mutation_batch(p, p_m, lower, upper, diversity),
    }
    mutate = mutation_map.get(mutation_fn, mutation_map["gaussian"])
    children = mutate(children)
    
    return children
