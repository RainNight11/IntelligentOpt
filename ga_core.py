"""
遗传算法核心模块

实现了以下算法变体：
- GA_baseline: 基础遗传算法
- GA_MP: 多种群遗传算法（含救援迁移）
- GA_RL: RL增强遗传算法（Q-learning自适应算子选择）
- GA_MP_RL: 混合算法（多种群 + RL）

主要特性：
- 精英保留策略
- 早停机制
- 向量化操作
- 可配置参数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmarks import BenchmarkFunction
from operators import (
    batch_reproduce,
    adaptive_gaussian_mutation,
    gaussian_mutation,
    rank_selection,
    roulette_wheel_selection,
    tournament_selection,
    two_point_crossover,
    uniform_crossover,
    uniform_mutation,
)
from rl_controller import RLController


@dataclass
class GAConfig:
    """GA 配置参数"""
    # 基础参数
    population_size: int = 200
    num_generations: int = 500
    p_c: float = 0.9
    p_m: float = 0.05
    dim: int = 30
    # 精英保留和早停
    elitism_count: int = 2
    early_stop_patience: int = 50
    early_stop_tolerance: float = 1e-8
    gaussian_sigma: float = 0.1
    tournament_k: int = 3
    # 变异自适应尺度（配合 RL 区分探索/开发）
    p_m_min: float = 0.005
    p_m_max: float = 0.3
    explore_p_m_scale: float = 1.6
    exploit_p_m_scale: float = 0.6
    explore_sigma_scale: float = 1.4
    exploit_sigma_scale: float = 0.7
    # 算法开关
    use_multipop: bool = False
    use_rescue_migration: bool = False
    use_rl: bool = False
    # 多种群参数
    num_subpops: int = 4
    migration_period: int = 10
    improvement_window: int = 10
    migration_count: int = 3
    migration_good_threshold: float = 0.01
    migration_bad_threshold: float = 0.001
    migration_noise_sigma: float = 0.02
    # 停滞注入
    stagnation_div_threshold: float = 0.15
    stagnation_imp_threshold: float = 0.001
    stagnation_replace_frac: float = 0.15
    # RL 参数可选
    rl_params: Optional[Dict[str, object]] = None


def compute_diversity(pop: np.ndarray, lower: float, upper: float) -> float:
    """计算种群多样性（归一化的平均标准差）
    
    Args:
        pop: 种群 (pop_size, dim)
        lower: 搜索空间下界
        upper: 搜索空间上界
    
    Returns:
        归一化的多样性值 [0, 1]
    """
    std = np.std(pop, axis=0)
    norm_std = std / (upper - lower + 1e-12)
    return float(np.mean(norm_std))


class GARunner:
    """遗传算法运行器"""
    
    # 动作到算子组合的映射
    ACTION_MAP = {
        0: ("roulette", "uniform", "uniform"),
        1: ("tournament", "uniform", "uniform"),
        2: ("tournament", "two_point", "gaussian"),
        3: ("roulette", "two_point", "gaussian"),
        4: ("rank", "two_point", "adaptive_gaussian"),
        5: ("rank", "uniform", "adaptive_gaussian"),
    }
    ACTION_STYLE = {
        0: "explore",
        1: "explore",
        2: "balanced",
        3: "balanced",
        4: "exploit",
        5: "exploit",
    }
    
    def __init__(self, config: GAConfig):
        self.config = config

    def _default_action(self) -> int:
        """默认动作：锦标赛选择 + 两点交叉 + 高斯变异"""
        return 2

    def _apply_elitism(
        self,
        old_pop: np.ndarray,
        old_fit: np.ndarray,
        new_pop: np.ndarray,
        new_fit: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用精英保留策略
        
        将旧种群中最优的 elitism_count 个个体替换新种群中最差的个体。
        
        Args:
            old_pop: 旧种群
            old_fit: 旧种群适应度
            new_pop: 新种群
            new_fit: 新种群适应度
        
        Returns:
            (更新后的新种群, 更新后的适应度)
        """
        elite_count = self.config.elitism_count
        if elite_count <= 0:
            return new_pop, new_fit
        
        # 找到旧种群中最好的个体
        elite_idx = np.argsort(old_fit)[:elite_count]
        # 找到新种群中最差的个体
        worst_idx = np.argsort(new_fit)[-elite_count:]
        
        # 替换
        new_pop = new_pop.copy()
        new_fit = new_fit.copy()
        new_pop[worst_idx] = old_pop[elite_idx]
        new_fit[worst_idx] = old_fit[elite_idx]
        
        return new_pop, new_fit
    
    def _check_early_stop(
        self,
        best_history: List[float],
        patience: int,
        tolerance: float,
    ) -> bool:
        """检查是否满足早停条件
        
        Args:
            best_history: 历史最优适应度列表
            patience: 耐心值（连续无改进的代数）
            tolerance: 容忍度（改进小于此值视为无改进）
        
        Returns:
            是否应该早停
        """
        if patience <= 0 or len(best_history) < patience:
            return False
        
        # 检查最近 patience 代是否有显著改进
        recent = best_history[-patience:]
        improvement = recent[0] - recent[-1]
        return improvement < tolerance

    def _select_and_reproduce_vectorized(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        action_id: int,
        bounds: Tuple[float, float],
        div_norm: float,
    ) -> np.ndarray:
        """向量化的选择和繁殖操作
        
        Args:
            population: 当前种群
            fitness: 适应度
            action_id: RL 动作 ID
            bounds: 搜索空间边界
            div_norm: 归一化多样性
        
        Returns:
            新种群
        """
        lower, upper = bounds
        selection_name, crossover_name, mutation_name = self.ACTION_MAP[action_id]
        # 动态调整变异强度：探索动作更激进，开发动作更保守；多样性越低，变异越强
        style = self.ACTION_STYLE.get(action_id, "balanced")
        p_m = self.config.p_m
        sigma = self.config.gaussian_sigma
        if style == "explore":
            p_m *= self.config.explore_p_m_scale
            sigma *= self.config.explore_sigma_scale
        elif style == "exploit":
            p_m *= self.config.exploit_p_m_scale
            sigma *= self.config.exploit_sigma_scale
        # 多样性低时额外提升变异概率
        p_m *= (1.0 + 0.6 * max(0.0, 0.5 - div_norm))
        p_m = min(self.config.p_m_max, max(self.config.p_m_min, p_m))
        sigma = max(1e-4, sigma)

        return batch_reproduce(
            population=population,
            fitness=fitness,
            p_c=self.config.p_c,
            p_m=p_m,
            lower=lower,
            upper=upper,
            selection_fn=selection_name,
            crossover_fn=crossover_name,
            mutation_fn=mutation_name,
            sigma=sigma,
            tournament_k=self.config.tournament_k,
            diversity=div_norm,
        )

    def _rescue_migration(
        self,
        subpops: List[np.ndarray],
        fitness_list: List[np.ndarray],
        best_history: List[List[float]],
        g: int,
        bounds: Tuple[float, float],
    ) -> None:
        """救援迁移：将优秀子种群的精英迁移到停滞子种群
        
        Args:
            subpops: 子种群列表
            fitness_list: 各子种群适应度列表
            best_history: 各子种群历史最优记录
            g: 当前代数
            bounds: 搜索空间边界
        """
        cfg = self.config
        
        if g == 0 or (g % cfg.migration_period) != 0:
            return
        
        # 迁移概率随代数增加
        p_mig = 0.2 + 0.6 * (g / max(1, cfg.num_generations))
        if np.random.rand() >= p_mig:
            return

        lower, upper = bounds
        L = cfg.improvement_window
        
        # 计算各子种群的改进率和方差
        delta_best = []
        variances = []
        for k, fit in enumerate(fitness_list):
            if len(best_history[k]) <= L:
                delta_best.append(-np.inf)
            else:
                past = best_history[k][-L]
                now = best_history[k][-1]
                delta = (past - now) / (abs(past) + 1e-12)
                delta_best.append(delta)
            variances.append(np.var(fit))

        # 识别优秀和停滞的子种群
        good_candidates = [i for i, d in enumerate(delta_best) if d > cfg.migration_good_threshold]
        bad_candidates = [i for i, d in enumerate(delta_best) if d < cfg.migration_bad_threshold]

        if not good_candidates or not bad_candidates:
            return

        # 选择方差最大的优秀子种群（多样性好）
        good_idx = max(good_candidates, key=lambda idx: variances[idx])
        # 选择方差最小的停滞子种群（多样性差）
        bad_idx = min(bad_candidates, key=lambda idx: variances[idx])

        if good_idx == bad_idx:
            return
        
        pop_good, fit_good = subpops[good_idx], fitness_list[good_idx]
        pop_bad, fit_bad = subpops[bad_idx], fitness_list[bad_idx]

        # 迁移精英个体
        n_migrate = min(cfg.migration_count, len(pop_good), len(pop_bad))
        elite_indices = np.argsort(fit_good)[:n_migrate]
        worst_indices = np.argsort(fit_bad)[-n_migrate:]

        pop_bad = pop_bad.copy()
        fit_bad = fit_bad.copy()
        # 复制精英并注入小噪声，防止完全克隆导致多样性塌陷
        noise = np.random.randn(*pop_good[elite_indices].shape) * cfg.migration_noise_sigma * (upper - lower)
        migrated = np.clip(pop_good[elite_indices] + noise, lower, upper)
        pop_bad[worst_indices] = migrated
        fit_bad[worst_indices] = fit_good[elite_indices]

        subpops[bad_idx] = np.clip(pop_bad, lower, upper)
        fitness_list[bad_idx] = fit_bad

    def run(
        self,
        benchmark: BenchmarkFunction,
        seed: int = 0,
        rl_controller: Optional[RLController] = None,
    ) -> Dict:
        """运行遗传算法
        
        Args:
            benchmark: 基准测试函数
            seed: 随机种子
            rl_controller: RL 控制器（可选）
        
        Returns:
            包含运行结果的字典：
            - best_curve: 每代最优适应度
            - diversity_curve: 每代多样性
            - final_best: 最终最优适应度
            - rl_counts: RL 动作统计（如果使用 RL）
            - early_stopped: 是否早停
            - actual_generations: 实际运行的代数
        """
        np.random.seed(seed)
        cfg = self.config
        bounds = (benchmark.lower, benchmark.upper)
        dim = benchmark.dim
        
        # 初始化种群
        if cfg.use_multipop:
            sub_size = cfg.population_size // cfg.num_subpops
            subpops = [
                np.random.uniform(benchmark.lower, benchmark.upper, size=(sub_size, dim))
                for _ in range(cfg.num_subpops)
            ]
        else:
            subpops = [
                np.random.uniform(benchmark.lower, benchmark.upper, size=(cfg.population_size, dim))
            ]

        # 初始化 RL 控制器
        rl_settings = cfg.rl_params or {}
        controller_kwargs = RLController.filter_kwargs(rl_settings)
        multi_agent = bool(rl_settings.get("per_subpop_controller", cfg.use_multipop))
        sync_interval = int(rl_settings.get("sync_q_interval", 0))
        sync_tau = float(rl_settings.get("sync_q_tau", 0.0))
        sync_tau = min(1.0, max(0.0, sync_tau))
        rl_controllers: List[RLController] = []
        if rl_controller is not None:
            rl_controllers = [rl_controller]
            multi_agent = False
        elif cfg.use_rl:
            if multi_agent and cfg.use_multipop:
                rl_controllers = [RLController(**controller_kwargs) for _ in subpops]
            else:
                rl_controllers = [RLController(**controller_kwargs)]
                multi_agent = False

        # 状态跟踪
        best_history = [[] for _ in subpops]
        prev_mean = [None for _ in subpops]
        last_mp_signals: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0) for _ in subpops]
        global_best_history: List[float] = []
        div_min, div_max = np.inf, 0.0

        # 结果记录
        best_curve = []
        diversity_curve = []
        early_stopped = False
        actual_generations = cfg.num_generations

        # 初始适应度评估
        fitness_list = [benchmark(pop) for pop in subpops]

        for g in range(cfg.num_generations):
            # 收集当代统计信息
            stats = []
            divs = []
            for idx, (pop, fit) in enumerate(zip(subpops, fitness_list)):
                best = float(np.min(fit))
                mean = float(np.mean(fit))
                div = compute_diversity(pop, benchmark.lower, benchmark.upper)
                
                # 更新多样性范围（用于归一化）
                div_min = min(div_min, div)
                div_max = max(div_max, div)
                div_norm = 0.0 if div_max == div_min else (div - div_min) / (div_max - div_min)
                
                best_history[idx].append(best)
                stats.append((best, mean, fit.var(), div, div_norm))
                divs.append(div_norm)

            # 全局最优和多样性
            global_best = min(s[0] for s in stats)
            global_div = float(np.mean(divs)) if divs else 0.0
            best_curve.append(global_best)
            diversity_curve.append(global_div)
            global_best_history.append(global_best)
            
            # 早停检查
            if self._check_early_stop(
                global_best_history,
                cfg.early_stop_patience,
                cfg.early_stop_tolerance,
            ):
                early_stopped = True
                actual_generations = g + 1
                break
            
            # 救援迁移（多种群模式）
            if cfg.use_multipop and cfg.use_rescue_migration:
                self._rescue_migration(subpops, fitness_list, best_history, g, bounds)

            # 进化各子种群
            next_subpops = []
            next_fitness = []
            next_mp_signals: List[Tuple[float, float, float]] = []
            global_pop = np.concatenate(subpops) if cfg.use_multipop else None
            global_fit = np.concatenate(fitness_list) if cfg.use_multipop else None

            for idx, (pop, fit, stat) in enumerate(zip(subpops, fitness_list, stats)):
                best_val, mean, _, _, div_norm = stat
                
                # 计算改进率
                if prev_mean[idx] is None:
                    improvement = 0.0
                else:
                    improvement = (prev_mean[idx] - mean) / (abs(prev_mean[idx]) + 1e-12)
                prev_mean[idx] = mean

                # 停滞+低多样性时，注入随机个体以拉高多样性（只在多种群+RL时启用）
                if (
                    cfg.use_multipop
                    and cfg.use_rl
                    and div_norm < cfg.stagnation_div_threshold
                    and improvement <= cfg.stagnation_imp_threshold
                ):
                    replace_count = max(1, int(len(pop) * cfg.stagnation_replace_frac))
                    worst_idx = np.argsort(fit)[-replace_count:]
                    pop = pop.copy()
                    pop[worst_idx] = np.random.uniform(
                        benchmark.lower,
                        benchmark.upper,
                        size=(replace_count, dim),
                    )
                    fit = benchmark(pop)
                    best_val = float(np.min(fit))
                    mean = float(np.mean(fit))
                    div = compute_diversity(pop, benchmark.lower, benchmark.upper)
                    div_min = min(div_min, div)
                    div_max = max(div_max, div)
                    div_norm = 0.0 if div_max == div_min else (div - div_min) / (div_max - div_min)

                # 选择动作
                state = 0
                action = self._default_action()
                ctrl = None
                if cfg.use_rl and rl_controllers:
                    ctrl = rl_controllers[idx] if (multi_agent and idx < len(rl_controllers)) else rl_controllers[0]
                    state = ctrl.encode_state(div_norm, improvement, last_mp_signals[idx])
                    action = ctrl.select_action(state, g, cfg.num_generations, improvement)

                # 繁殖新种群（向量化）
                # GA offspring（本子群）
                new_pop_ga = self._select_and_reproduce_vectorized(pop, fit, action, bounds, div_norm)
                new_fit_ga = benchmark(new_pop_ga)
                mp_signal = (0.0, 0.0, 0.0)

                # 仅在 MP+RL 时尝试全局池候选
                if cfg.use_multipop and cfg.use_rl and ctrl is not None and global_pop is not None and global_fit is not None:
                    global_div_norm = global_div
                    new_pop_mp_full = self._select_and_reproduce_vectorized(global_pop, global_fit, action, bounds, global_div_norm)
                    new_pop_mp = new_pop_mp_full[: len(pop)]  # 与当前子群等长
                    new_fit_mp = benchmark(new_pop_mp)

                    # 多样性与相对收益
                    mp_div = compute_diversity(new_pop_mp, benchmark.lower, benchmark.upper)
                    div_min = min(div_min, mp_div)
                    div_max = max(div_max, mp_div)
                    mp_div_norm = 0.0 if div_max == div_min else (mp_div - div_min) / (div_max - div_min)
                    mp_div_delta = mp_div_norm - div_norm
                    mean_ga = float(np.mean(new_fit_ga))
                    mean_mp = float(np.mean(new_fit_mp))
                    mp_fit_gain = (mean_ga - mean_mp) / (abs(mean_ga) + 1e-12)  # >0 表示 MP 更好
                    mp_diff_norm = float(
                        np.mean(np.abs(new_pop_mp - new_pop_ga)) / (abs(benchmark.upper - benchmark.lower) + 1e-12)
                    )
                    mp_signal = (mp_div_delta, mp_diff_norm, mp_fit_gain)

                    # 选择更优个体（逐个体择优，避免将劣质 MP 混入）
                    use_mp_mask = new_fit_mp <= new_fit_ga
                    new_pop = np.where(use_mp_mask[:, None], new_pop_mp, new_pop_ga)
                    new_fit = np.where(use_mp_mask, new_fit_mp, new_fit_ga)

                    # 若整体 MP 明显更优且多样性提升，可追加一层平滑混合以保留差异性
                    if mp_fit_gain > 0 and mp_div_delta > 0:
                        alpha = ctrl.get_alpha(state, g, cfg.num_generations)
                        new_pop = np.clip(alpha * new_pop_mp + (1.0 - alpha) * new_pop, benchmark.lower, benchmark.upper)
                        new_fit = benchmark(new_pop)
                else:
                    new_pop = new_pop_ga
                    new_fit = new_fit_ga

                # 精英保留
                new_pop, new_fit = self._apply_elitism(pop, fit, new_pop, new_fit)
                
                # RL 更新
                if cfg.use_rl and rl_controllers and ctrl is not None:
                    new_div = compute_diversity(new_pop, benchmark.lower, benchmark.upper)
                    div_min = min(div_min, new_div)
                    div_max = max(div_max, new_div)
                    new_div_norm = 0.0 if div_max == div_min else (new_div - div_min) / (div_max - div_min)

                    f_old = mean
                    f_new = float(np.mean(new_fit))
                    div_old, div_new = div_norm, new_div_norm
                    best_new = float(np.min(new_fit))
                    
                    # 计算奖励
                    r_fit = (f_old - f_new) / (abs(f_old) + 1e-12)
                    delta_d = (div_new - div_old) / (abs(div_old) + 1e-12 + 1e-9)
                    if delta_d > 0:
                        r_div = min(0.2, delta_d)
                    elif delta_d < -0.1:
                        r_div = max(-0.3, delta_d)
                    else:
                        r_div = 0.0

                    r_best = (best_val - best_new) / (abs(best_val) + 1e-12)
                    reward = (
                        ctrl.reward_fitness_weight * r_fit
                        + ctrl.reward_diversity_weight * r_div
                        + ctrl.reward_best_weight * r_best
                    )
                    reward = ctrl.shape_reward(reward)

                    next_state = ctrl.encode_state(
                        new_div_norm,
                        (f_old - f_new) / (abs(f_old) + 1e-12),
                        mp_signal,
                    )
                    ctrl.update(state, action, reward, next_state)

                next_subpops.append(new_pop)
                next_fitness.append(new_fit)
                next_mp_signals.append(mp_signal)

            subpops = next_subpops
            fitness_list = next_fitness
            last_mp_signals = next_mp_signals

            # RL 控制器同步
            if (
                cfg.use_rl
                and multi_agent
                and rl_controllers
                and sync_interval > 0
                and sync_tau > 0
                and ((g + 1) % sync_interval == 0)
            ):
                mean_q = np.mean([ctrl.q_table for ctrl in rl_controllers], axis=0)
                for ctrl in rl_controllers:
                    ctrl.q_table = (1.0 - sync_tau) * ctrl.q_table + sync_tau * mean_q

        # 最终结果
        final_best = min(float(np.min(fit)) for fit in fitness_list)
        rl_counts = None
        if cfg.use_rl and rl_controllers:
            rl_counts = np.sum([ctrl.counts for ctrl in rl_controllers], axis=0)
        
        return {
            "best_curve": np.array(best_curve),
            "diversity_curve": np.array(diversity_curve),
            "final_best": final_best,
            "rl_counts": rl_counts,
            "early_stopped": early_stopped,
            "actual_generations": actual_generations,
        }
