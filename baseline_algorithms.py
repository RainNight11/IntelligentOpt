"""
对比算法实现

包含以下优化算法（连续优化）：
- Standard GA: 使用 GARunner/GAConfig，不在本文件中实现，由主框架调用
- 基础算法：PSO, DE
- 进化策略：ES
- 模拟退火：SA
- 蚁群算法：ACO (ACOR 连续版本)
- 先进 DE：CMA-ES, SHADE, L-SHADE
- 混合/自适应算法：AGSK, HHO

所有算法接口统一：
    run(benchmark, population_size, num_generations, seed) -> Dict
    返回: {"best_curve": np.ndarray, "final_best": float, "diversity_curve": np.ndarray}
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from benchmarks import BenchmarkFunction


def compute_diversity(pop: np.ndarray, lower: float, upper: float) -> float:
    """计算种群多样性（归一化的平均标准差）"""
    std = np.std(pop, axis=0)
    norm_std = std / (upper - lower + 1e-12)
    return float(np.mean(norm_std))


# =============================================================================
# PSO - 粒子群优化
# =============================================================================


class PSO:
    """粒子群优化算法（向量化实现）

    参数:
        w_start, w_end: 惯性权重线性衰减
        c1, c2: 个体/社会学习因子
    """

    def __init__(
        self,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max_ratio: float = 0.2,
    ):
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper
        v_max = self.v_max_ratio * (upper - lower)

        positions = np.random.uniform(lower, upper, (population_size, dim))
        velocities = np.random.uniform(-v_max, v_max, (population_size, dim))

        fitness = benchmark(positions)
        pbest = positions.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(pbest_fit)
        gbest = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        best_curve = [gbest_fit]
        diversity_curve = [compute_diversity(positions, lower, upper)]

        for g in range(num_generations):
            w = self.w_start - (self.w_start - self.w_end) * g / max(1, num_generations)

            r1 = np.random.rand(population_size, dim)
            r2 = np.random.rand(population_size, dim)
            velocities = (
                w * velocities
                + self.c1 * r1 * (pbest - positions)
                + self.c2 * r2 * (gbest - positions)
            )
            velocities = np.clip(velocities, -v_max, v_max)

            positions = positions + velocities
            positions = np.clip(positions, lower, upper)

            fitness = benchmark(positions)

            improved = fitness < pbest_fit
            pbest[improved] = positions[improved]
            pbest_fit[improved] = fitness[improved]

            min_idx = np.argmin(pbest_fit)
            if pbest_fit[min_idx] < gbest_fit:
                gbest = pbest[min_idx].copy()
                gbest_fit = pbest_fit[min_idx]

            best_curve.append(gbest_fit)
            diversity_curve.append(compute_diversity(positions, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": gbest_fit,
        }


# =============================================================================
# DE - 差分进化（向量化）
# =============================================================================


class DE:
    """差分进化算法 (DE/rand/1/bin) 向量化实现"""

    def __init__(self, F: float = 0.8, CR: float = 0.9):
        self.F = F
        self.CR = CR

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper
        NP = population_size

        population = np.random.uniform(lower, upper, (NP, dim))
        fitness = benchmark(population)

        best_fit = float(np.min(fitness))

        best_curve = [best_fit]
        diversity_curve = [compute_diversity(population, lower, upper)]

        for _ in range(num_generations):
            indices = np.arange(NP)
            r1 = np.random.randint(0, NP, NP)
            r2 = np.random.randint(0, NP, NP)
            r3 = np.random.randint(0, NP, NP)

            mask = r1 == indices
            r1[mask] = (r1[mask] + 1) % NP
            mask = (r2 == indices) | (r2 == r1)
            r2[mask] = (r2[mask] + 2) % NP
            mask = (r3 == indices) | (r3 == r1) | (r3 == r2)
            r3[mask] = (r3[mask] + 3) % NP

            mutants = population[r1] + self.F * (population[r2] - population[r3])
            mutants = np.clip(mutants, lower, upper)

            cross_mask = np.random.rand(NP, dim) < self.CR
            j_rand = np.random.randint(0, dim, NP)
            cross_mask[np.arange(NP), j_rand] = True
            trials = np.where(cross_mask, mutants, population)

            trial_fit = benchmark(trials)

            improved = trial_fit < fitness
            population[improved] = trials[improved]
            fitness[improved] = trial_fit[improved]

            current_best = float(np.min(fitness))
            if current_best < best_fit:
                best_fit = current_best

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(population, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


# =============================================================================
# CMA-ES
# =============================================================================


class CMAES:
    """简化版 CMA-ES"""

    def __init__(self, sigma0: float = 0.5):
        self.sigma0 = sigma0

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper

        lam = population_size
        mu = lam // 2

        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        mean = np.random.uniform(lower, upper, dim)
        sigma = self.sigma0 * (upper - lower)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        best_fit = np.inf
        best_curve = []
        diversity_curve = []

        for g in range(num_generations):
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

            z = np.random.randn(lam, dim)
            y = z @ np.diag(D) @ B.T
            offspring = mean + sigma * y
            offspring = np.clip(offspring, lower, upper)

            fitness = benchmark(offspring)
            order = np.argsort(fitness)
            offspring = offspring[order]
            y = y[order]
            z = z[order]

            if fitness[order[0]] < best_fit:
                best_fit = float(fitness[order[0]])

            mean_old = mean.copy()
            mean = weights @ offspring[:mu]

            y_w = weights @ y[:mu]
            z_w = weights @ z[:mu]

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_w)
            hsig = (
                np.linalg.norm(ps)
                / np.sqrt(1 - (1 - cs) ** (2 * (g + 1)))
                / chiN
                < 1.4 + 2 / (dim + 1)
            )

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

            artmp = (1 / sigma) * (offspring[:mu] - mean_old)
            C = (
                (1 - c1 - cmu) * C
                + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                + cmu * artmp.T @ np.diag(weights) @ artmp
            )
            C = (C + C.T) / 2

            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
            sigma = min(sigma, (upper - lower))

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(offspring, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


# =============================================================================
# ES - 进化策略 (Evolution Strategy)
# =============================================================================


class ES:
    """(μ, λ)-进化策略

    使用自适应步长高斯变异，截断选择。
    """

    def __init__(self, sigma0: float = 0.3, tau: float | None = None):
        self.sigma0 = sigma0
        self.tau = tau

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper

        lam = population_size              # 子代数量
        mu = max(2, lam // 4)              # 父代数量

        tau = self.tau if self.tau is not None else 1.0 / np.sqrt(2 * dim)
        tau_prime = 1.0 / np.sqrt(2 * np.sqrt(dim))

        population = np.random.uniform(lower, upper, (lam, dim))
        sigmas = np.full((lam, dim), self.sigma0 * (upper - lower))

        fitness = benchmark(population)
        best_fit = float(np.min(fitness))

        best_curve = [best_fit]
        diversity_curve = [compute_diversity(population, lower, upper)]

        for _ in range(num_generations):
            # 选择最优 μ 个个体作为父代
            order = np.argsort(fitness)[:mu]
            parents = population[order]
            parent_sigmas = sigmas[order]

            # 生成 λ 个子代
            z_global = np.random.randn(lam)          # 全局因子
            z_local = np.random.randn(lam, dim)      # 局部因子

            parent_idx = np.random.randint(0, mu, lam)
            base = parents[parent_idx]
            base_sigma = parent_sigmas[parent_idx]

            global_factor = np.exp(tau_prime * z_global)[:, None]
            local_factor = np.exp(tau * z_local)
            new_sigma = base_sigma * global_factor * local_factor
            new_sigma = np.clip(new_sigma, 1e-10, (upper - lower))

            offspring = base + new_sigma * np.random.randn(lam, dim)
            offspring = np.clip(offspring, lower, upper)

            off_fit = benchmark(offspring)

            # 选择最优 λ 个作为新种群（这里采用 (μ, λ) → 全部来自子代）
            order_off = np.argsort(off_fit)[:lam]
            population = offspring[order_off]
            sigmas = new_sigma[order_off]
            fitness = off_fit[order_off]

            current_best = float(np.min(fitness))
            if current_best < best_fit:
                best_fit = current_best

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(population, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


# =============================================================================
# SA - 模拟退火 (向量化多起点)
# =============================================================================


class SA:
    """模拟退火算法（多起点并行，向量化实现）"""

    def __init__(
        self,
        T0: float = 1000.0,
        T_min: float = 1e-8,
        alpha: float = 0.95,
    ) -> None:
        self.T0 = T0
        self.T_min = T_min
        self.alpha = alpha

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper

        n_chains = population_size
        solutions = np.random.uniform(lower, upper, (n_chains, dim))
        fitness = benchmark(solutions)

        best_solutions = solutions.copy()
        best_fitness = fitness.copy()

        best_idx = int(np.argmin(best_fitness))
        global_best = best_solutions[best_idx].copy()
        global_best_fit = float(best_fitness[best_idx])

        best_curve = [global_best_fit]
        diversity_curve = [compute_diversity(solutions, lower, upper)]

        T = self.T0
        for _ in range(num_generations):
            step_size = 0.1 * (upper - lower) * (T / self.T0)

            neighbors = solutions + step_size * np.random.randn(n_chains, dim)
            neighbors = np.clip(neighbors, lower, upper)

            neighbor_fit = benchmark(neighbors)

            delta = neighbor_fit - fitness
            accept_prob = np.exp(-delta / (T + 1e-12))
            accept = (delta < 0) | (np.random.rand(n_chains) < accept_prob)

            solutions[accept] = neighbors[accept]
            fitness[accept] = neighbor_fit[accept]

            improved = fitness < best_fitness
            best_solutions[improved] = solutions[improved]
            best_fitness[improved] = fitness[improved]

            best_idx = int(np.argmin(best_fitness))
            if best_fitness[best_idx] < global_best_fit:
                global_best = best_solutions[best_idx].copy()
                global_best_fit = float(best_fitness[best_idx])

            T = max(self.T_min, T * self.alpha)

            best_curve.append(global_best_fit)
            diversity_curve.append(compute_diversity(solutions, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": global_best_fit,
        }


# =============================================================================
# ACO - 连续域蚁群优化 (ACOR，向量化)
# =============================================================================


class ACO:
    """连续域蚁群优化 (ACOR)"""

    def __init__(
        self,
        q: float = 0.5,   # 选择压力
        xi: float = 0.85, # 局部搜索强度
    ) -> None:
        self.q = q
        self.xi = xi

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper

        k = population_size   # 存档大小
        m = population_size   # 每代生成新解数

        archive = np.random.uniform(lower, upper, (k, dim))
        archive_fit = benchmark(archive)

        order = np.argsort(archive_fit)
        archive = archive[order]
        archive_fit = archive_fit[order]

        best_fit = float(archive_fit[0])
        best_curve = [best_fit]
        diversity_curve = [compute_diversity(archive, lower, upper)]

        idx = np.arange(1, k + 1)
        weights = (1.0 / (self.q * k * np.sqrt(2 * np.pi))) * np.exp(
            -(idx**2) / (2 * (self.q * k) ** 2)
        )
        weights = weights / weights.sum()

        for _ in range(num_generations):
            # 选择 m 个参考解
            sel_idx = np.random.choice(k, size=m, p=weights)
            selected = archive[sel_idx]  # (m, dim)

            # 计算 sigma: 对每个 selected 与整个存档的平均绝对差
            diff = np.abs(archive[np.newaxis, :, :] - selected[:, np.newaxis, :])  # (m,k,dim)
            sigma = self.xi * np.mean(diff, axis=1)  # (m,dim)

            new_solutions = selected + sigma * np.random.randn(m, dim)
            new_solutions = np.clip(new_solutions, lower, upper)

            new_fit = benchmark(new_solutions)

            combined = np.vstack([archive, new_solutions])
            combined_fit = np.concatenate([archive_fit, new_fit])

            order = np.argsort(combined_fit)[:k]
            archive = combined[order]
            archive_fit = combined_fit[order]

            if archive_fit[0] < best_fit:
                best_fit = float(archive_fit[0])

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(archive, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


# =============================================================================
# SHADE / L-SHADE 省略注释，只保留实现（与会话中版本一致简化）
# =============================================================================


class SHADE:
    """SHADE: Success-History based Adaptive Differential Evolution"""

    def __init__(self, H: int = 100):
        self.H = H

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper
        NP = population_size

        population = np.random.uniform(lower, upper, (NP, dim))
        fitness = benchmark(population)

        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        k = 0

        best_fit = float(np.min(fitness))
        best_curve = [best_fit]
        diversity_curve = [compute_diversity(population, lower, upper)]

        archive: list[np.ndarray] = []
        archive_max = NP

        for _ in range(num_generations):
            S_F = []
            S_CR = []
            delta_f = []

            new_pop = population.copy()
            new_fit = fitness.copy()

            for i in range(NP):
                r = np.random.randint(self.H)

                F = -1.0
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                CR = float(np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0))

                p = max(0.05, 0.2)
                pbest_size = max(1, int(NP * p))
                pbest_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(pbest_indices)

                candidates = list(range(NP))
                candidates.remove(i)
                r1 = np.random.choice(candidates)

                pool = list(range(NP)) + list(range(NP, NP + len(archive)))
                if r1 in pool:
                    pool.remove(r1)
                if i in pool:
                    pool.remove(i)
                r2 = np.random.choice(pool)

                x_r2 = population[r2] if r2 < NP else archive[r2 - NP]

                mutant = (
                    population[i]
                    + F * (population[pbest_idx] - population[i])
                    + F * (population[r1] - x_r2)
                )
                mutant = np.clip(mutant, lower, upper)

                j_rand = np.random.randint(dim)
                trial = population[i].copy()
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]

                trial_fit = float(benchmark(trial))
                if trial_fit < fitness[i]:
                    if len(archive) < archive_max:
                        archive.append(population[i].copy())
                    else:
                        archive[np.random.randint(archive_max)] = population[i].copy()

                    new_pop[i] = trial
                    new_fit[i] = trial_fit

                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_fit)

                    if trial_fit < best_fit:
                        best_fit = trial_fit

            population = new_pop
            fitness = new_fit

            if S_F:
                weights = np.array(delta_f) / (np.sum(delta_f) + 1e-12)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)
                M_F[k] = np.sum(weights * S_F_arr**2) / (
                    np.sum(weights * S_F_arr) + 1e-12
                )
                M_CR[k] = float(np.sum(weights * S_CR_arr))
                k = (k + 1) % self.H

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(population, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


class LSHADE:
    """L-SHADE: SHADE with Linear Population Size Reduction"""

    def __init__(self, H: int = 100, NP_min: int = 4):
        self.H = H
        self.NP_min = NP_min

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper
        NP_init = population_size
        NP = NP_init
        max_nfe = NP_init * num_generations
        nfe = 0

        population = np.random.uniform(lower, upper, (NP, dim))
        fitness = benchmark(population)
        nfe += NP

        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        k = 0

        best_fit = float(np.min(fitness))
        best_curve = [best_fit]
        diversity_curve = [compute_diversity(population, lower, upper)]

        archive: list[np.ndarray] = []

        for _ in range(num_generations):
            if NP <= self.NP_min:
                break

            S_F = []
            S_CR = []
            delta_f = []

            new_pop_list = []
            new_fit_list = []

            for i in range(NP):
                r = np.random.randint(self.H)

                F = -1.0
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                CR = float(np.clip(np.random.normal(M_CR[r], 0.1), 0.0, 1.0))

                p = max(2 / NP, 0.2 - 0.15 * nfe / max_nfe)
                pbest_size = max(1, int(NP * p))
                pbest_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(pbest_indices)

                candidates = list(range(NP))
                candidates.remove(i)
                r1 = np.random.choice(candidates)

                pool_size = NP + len(archive)
                pool = [j for j in range(pool_size) if j not in (i, r1)]
                r2 = np.random.choice(pool)

                x_r2 = population[r2] if r2 < NP else archive[r2 - NP]

                mutant = (
                    population[i]
                    + F * (population[pbest_idx] - population[i])
                    + F * (population[r1] - x_r2)
                )
                mutant = np.clip(mutant, lower, upper)

                j_rand = np.random.randint(dim)
                trial = population[i].copy()
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial[mask] = mutant[mask]

                trial_fit = float(benchmark(trial))
                nfe += 1

                if trial_fit < fitness[i]:
                    if len(archive) < NP:
                        archive.append(population[i].copy())
                    else:
                        archive[np.random.randint(len(archive))] = population[i].copy()

                    new_pop_list.append(trial)
                    new_fit_list.append(trial_fit)

                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_fit)

                    if trial_fit < best_fit:
                        best_fit = trial_fit
                else:
                    new_pop_list.append(population[i])
                    new_fit_list.append(fitness[i])

            population = np.array(new_pop_list)
            fitness = np.array(new_fit_list)

            if S_F:
                weights = np.array(delta_f) / (np.sum(delta_f) + 1e-12)
                S_F_arr = np.array(S_F)
                S_CR_arr = np.array(S_CR)
                M_F[k] = np.sum(weights * S_F_arr**2) / (
                    np.sum(weights * S_F_arr) + 1e-12
                )
                M_CR[k] = float(np.sum(weights * S_CR_arr))
                k = (k + 1) % self.H

            NP_new = int(
                round(NP_init - (NP_init - self.NP_min) * nfe / max(1, max_nfe))
            )
            NP_new = max(NP_new, self.NP_min)

            if NP_new < NP:
                order = np.argsort(fitness)
                population = population[order[:NP_new]]
                fitness = fitness[order[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    archive = archive[:NP]

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(population, lower, upper))

        while len(best_curve) < num_generations + 1:
            best_curve.append(best_fit)
            diversity_curve.append(diversity_curve[-1] if diversity_curve else 0.0)

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


# =============================================================================
# AGSK - 自知识自适应遗传算法（基于 DE 的自适应）
# =============================================================================


class AGSK:
    """Adaptive GA with Self-Knowledge（这里实现为多策略自适应 DE）"""

    def __init__(self) -> None:
        pass

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper
        NP = population_size

        population = np.random.uniform(lower, upper, (NP, dim))
        fitness = benchmark(population)

        best_idx = int(np.argmin(fitness))
        best_fit = float(fitness[best_idx])
        best_ind = population[best_idx].copy()

        F = 0.5 * np.ones(NP)
        CR = 0.9 * np.ones(NP)

        best_curve = [best_fit]
        diversity_curve = [compute_diversity(population, lower, upper)]

        for g in range(num_generations):
            progress = g / max(1, num_generations)

            new_pop = population.copy()
            new_fit = fitness.copy()

            for i in range(NP):
                if np.random.rand() < 0.1:
                    F[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR[i] = np.random.rand()

                if progress < 0.3:
                    strategy = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
                elif progress < 0.7:
                    strategy = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
                else:
                    strategy = np.random.choice([0, 1, 2], p=[0.2, 0.3, 0.5])

                candidates = [j for j in range(NP) if j != i]

                if strategy == 0:
                    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                    mutant = population[r1] + F[i] * (population[r2] - population[r3])
                elif strategy == 1:
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                    mutant = best_ind + F[i] * (population[r1] - population[r2])
                else:
                    r1, r2 = np.random.choice(candidates, 2, replace=False)
                    mutant = (
                        population[i]
                        + F[i] * (best_ind - population[i])
                        + F[i] * (population[r1] - population[r2])
                    )

                mutant = np.clip(mutant, lower, upper)

                j_rand = np.random.randint(dim)
                trial = population[i].copy()
                mask = np.random.rand(dim) < CR[i]
                mask[j_rand] = True
                trial[mask] = mutant[mask]

                trial_fit = float(benchmark(trial))
                if trial_fit < fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    if trial_fit < best_fit:
                        best_fit = trial_fit
                        best_ind = trial.copy()

            population = new_pop
            fitness = new_fit

            best_curve.append(best_fit)
            diversity_curve.append(compute_diversity(population, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": best_fit,
        }


# =============================================================================
# HHO - 哈里斯鹰优化
# =============================================================================


class HHO:
    """Harris Hawks Optimization（连续优化）"""

    def __init__(self) -> None:
        pass

    def _levy_flight(self, dim: int, beta: float = 1.5) -> np.ndarray:
        sigma = (
            np.math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step

    def run(
        self,
        benchmark: BenchmarkFunction,
        population_size: int = 100,
        num_generations: int = 500,
        seed: int = 0,
    ) -> Dict:
        np.random.seed(seed)
        dim = benchmark.dim
        lower, upper = benchmark.lower, benchmark.upper
        N = population_size

        X = np.random.uniform(lower, upper, (N, dim))
        fitness = benchmark(X)

        best_idx = int(np.argmin(fitness))
        rabbit = X[best_idx].copy()
        rabbit_energy = float(fitness[best_idx])

        best_curve = [rabbit_energy]
        diversity_curve = [compute_diversity(X, lower, upper)]

        for g in range(num_generations):
            E0 = 2 * np.random.rand() - 1
            E = 2 * E0 * (1 - g / max(1, num_generations))

            for i in range(N):
                q = np.random.rand()
                r = np.random.rand()

                if abs(E) >= 1:
                    if q >= 0.5:
                        rand_idx = np.random.randint(N)
                        X_rand = X[rand_idx]
                        X[i] = X_rand - r * np.abs(X_rand - 2 * r * X[i])
                    else:
                        X_m = np.mean(X, axis=0)
                        X[i] = (rabbit - X_m) - r * (lower + r * (upper - lower))
                else:
                    J = 2 * (1 - np.random.rand())
                    if r >= 0.5:
                        if abs(E) >= 0.5:
                            X[i] = rabbit - E * np.abs(J * rabbit - X[i])
                        else:
                            X[i] = rabbit - E * np.abs(rabbit - X[i])
                    else:
                        if abs(E) >= 0.5:
                            Y = rabbit - E * np.abs(J * rabbit - X[i])
                            Y = np.clip(Y, lower, upper)
                            if benchmark(Y) < fitness[i]:
                                X[i] = Y
                            else:
                                LF = self._levy_flight(dim)
                                Z = Y + np.random.rand(dim) * LF
                                Z = np.clip(Z, lower, upper)
                                if benchmark(Z) < fitness[i]:
                                    X[i] = Z
                        else:
                            Y = rabbit - E * np.abs(
                                J * rabbit - np.mean(X, axis=0)
                            )
                            Y = np.clip(Y, lower, upper)
                            if benchmark(Y) < fitness[i]:
                                X[i] = Y
                            else:
                                LF = self._levy_flight(dim)
                                Z = Y + np.random.rand(dim) * LF
                                Z = np.clip(Z, lower, upper)
                                if benchmark(Z) < fitness[i]:
                                    X[i] = Z

                X[i] = np.clip(X[i], lower, upper)

            fitness = benchmark(X)
            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < rabbit_energy:
                rabbit = X[best_idx].copy()
                rabbit_energy = float(fitness[best_idx])

            best_curve.append(rabbit_energy)
            diversity_curve.append(compute_diversity(X, lower, upper))

        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": rabbit_energy,
        }


# =============================================================================
# ES - 进化策略（在上方已有实现）
# SA - 模拟退火（向量化）
# ACO - 连续蚁群优化（向量化）
# 统一接口
# =============================================================================

BASELINE_ALGORITHMS = {
    "PSO": PSO,
    "DE": DE,
    "ES": ES,
    "SA": SA,
    "ACO": ACO,
    "CMA-ES": CMAES,
    "SHADE": SHADE,
    "L-SHADE": LSHADE,
    "AGSK": AGSK,
    "HHO": HHO,
}


def get_baseline_algorithm(name: str):
    """获取对比算法实例"""
    if name not in BASELINE_ALGORITHMS:
        raise ValueError(f"未知算法: {name}。可用: {list(BASELINE_ALGORITHMS.keys())}")
    return BASELINE_ALGORITHMS[name]()


def list_baseline_algorithms() -> list:
    """列出所有可用的对比算法"""
    return list(BASELINE_ALGORITHMS.keys())


