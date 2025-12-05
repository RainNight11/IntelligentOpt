"""
Q-learning 控制器

自适应选择遗传算子组合，根据多样性与改进信号动态调整。
新增特性：
    - 自适应状态离散：阈值随历史统计更新，提升对不同基准的敏感度
    - 奖励形状化：支持最优值奖励、裁剪与滑动平均，降低噪声
    - 停滞探索增强：改进率过低时自动提升探索概率

默认状态空间 (9个状态):
    div_level ∈ {低, 中, 高}, imp_level ∈ {停滞, 缓慢, 快速}
    状态编码: div_level * 3 + imp_level

动作空间 (6个动作):
    0: 轮盘选择 + 均匀交叉 + 均匀变异
    1: 锦标赛选择 + 均匀交叉 + 均匀变异
    2: 锦标赛选择 + 两点交叉 + 高斯变异 (默认)
    3: 轮盘选择 + 两点交叉 + 高斯变异
    4: 排序选择 + 两点交叉 + 自适应高斯变异
    5: 排序选择 + 均匀交叉 + 自适应高斯变异
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np


class RLController:
    """Q-learning 强化学习控制器"""
    
    # 状态离散化阈值
    DIV_LOW_THRESHOLD = 0.3
    DIV_HIGH_THRESHOLD = 0.7
    IMP_STAGNANT_THRESHOLD = 0.001
    IMP_FAST_THRESHOLD = 0.01
    CONTROLLER_KWARGS = {
        "num_states",
        "num_actions",
        "alpha",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "reward_fitness_weight",
        "reward_diversity_weight",
        "reward_best_weight",
        "reward_clip",
        "reward_smoothing",
        "use_adaptive_thresholds",
        "adaptive_window",
        "adaptive_min_samples",
        "stagnation_eps_boost",
        "div_adaptive_quantiles",
        "imp_adaptive_quantiles",
        "mp_effect_quantiles",
        "alpha_init",
        "alpha_min",
        "alpha_max",
        "alpha_lr",
        "alpha_schedule_pow",
    }
    
    @classmethod
    def filter_kwargs(cls, params: Optional[Dict[str, object]]) -> Dict[str, object]:
        """提取 RLController 可识别的参数"""
        if not params:
            return {}
        return {k: params[k] for k in cls.CONTROLLER_KWARGS if k in params}
    
    def __init__(
        self,
        num_states: int = 9,
        num_actions: int = 6,
        alpha: float = 0.3,
        gamma: float = 0.9,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        reward_fitness_weight: float = 0.8,
        reward_diversity_weight: float = 0.2,
        reward_best_weight: float = 0.3,
        reward_clip: float = 1.0,
        reward_smoothing: float = 0.15,
        use_adaptive_thresholds: bool = True,
        adaptive_window: int = 60,
        adaptive_min_samples: int = 15,
        stagnation_eps_boost: float = 0.2,
        div_adaptive_quantiles: Tuple[float, float] = (0.3, 0.7),
        imp_adaptive_quantiles: Tuple[float, float] = (0.25, 0.7),
        mp_effect_quantiles: Tuple[float, float] = (0.3, 0.7),
        alpha_init: float = 0.8,
        alpha_min: float = 0.05,
        alpha_max: float = 1.0,
        alpha_lr: float = 0.05,
        alpha_schedule_pow: float = 1.5,
    ):
        """初始化 RL 控制器
        
        Args:
            num_states: 状态空间大小
            num_actions: 动作空间大小
            alpha: 学习率 (0-1)，控制 Q 值更新步长
            gamma: 折扣因子 (0-1)，控制未来奖励权重
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            reward_fitness_weight: 适应度奖励权重
            reward_diversity_weight: 多样性奖励权重
            reward_best_weight: 最优值改进奖励权重
            reward_clip: 奖励裁剪幅度（abs<=reward_clip）
            reward_smoothing: 奖励滑动平均系数 (0=禁用)
            use_adaptive_thresholds: 是否使用自适应状态离散
            adaptive_window: 自适应阈值的滑动窗口长度
            adaptive_min_samples: 开始自适应所需的样本数量
            stagnation_eps_boost: 停滞状态下额外探索度
            div_adaptive_quantiles: 多样性阈值分位数
            imp_adaptive_quantiles: 改进率阈值分位数
            mp_effect_quantiles: MP 影响度分位，用于状态离散
            alpha_init: α 初始值（偏探索）
            alpha_min: α 最小值
            alpha_max: α 最大值
            alpha_lr: α 更新学习率
            alpha_schedule_pow: α 随迭代衰减的幂次，控制“前高后低”
        """
        if num_states < 9:
            raise ValueError("num_states 至少为 9，以覆盖 (div, improvement) 的 3×3 组合")
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.reward_fitness_weight = reward_fitness_weight
        self.reward_diversity_weight = reward_diversity_weight
        self.reward_best_weight = reward_best_weight
        self.reward_clip = reward_clip
        self.reward_smoothing = reward_smoothing
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.adaptive_window = adaptive_window
        self.adaptive_min_samples = adaptive_min_samples
        self.stagnation_eps_boost = stagnation_eps_boost
        self.div_adaptive_quantiles = div_adaptive_quantiles
        self.imp_adaptive_quantiles = imp_adaptive_quantiles
        self.mp_effect_quantiles = mp_effect_quantiles
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_lr = alpha_lr
        self.alpha_schedule_pow = alpha_schedule_pow
        
        # Q 表和动作计数
        self.q_table = np.zeros((num_states, num_actions), dtype=float)
        self.counts = np.zeros_like(self.q_table, dtype=int)
        
        # 学习历史（用于分析）
        self.reward_history: list[float] = []
        self.action_history: list[int] = []
        self._last_reward = 0.0
        self._div_history: deque[float] = deque(maxlen=adaptive_window)
        self._imp_history: deque[float] = deque(maxlen=adaptive_window)
        self._mp_effect_history: deque[float] = deque(maxlen=adaptive_window)
        self._div_bins = (self.DIV_LOW_THRESHOLD, self.DIV_HIGH_THRESHOLD)
        self._imp_bins = (self.IMP_STAGNANT_THRESHOLD, self.IMP_FAST_THRESHOLD)
        self._mp_effect_bins = (-0.05, 0.05)
        self.alpha_table = np.full(self.num_states, self.alpha_init, dtype=float)

    def epsilon(self, g: int, g_max: int, improvement: Optional[float] = None) -> float:
        """计算当前探索率（线性衰减）
        
        Args:
            g: 当前代数
            g_max: 最大代数
            improvement: 当前改进率，用于停滞加成
        
        Returns:
            当前探索率
        """
        progress = g / max(1, g_max)
        eps = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        eps = float(max(self.epsilon_end, eps))
        if (
            self.stagnation_eps_boost > 0
            and improvement is not None
            and improvement <= self._imp_bins[0]
        ):
            eps = min(1.0, eps + self.stagnation_eps_boost)
        return eps

    def _update_bins(self, value: float, history: deque, current: Tuple[float, float],
                     quantiles: Tuple[float, float]) -> Tuple[float, float]:
        """根据历史样本更新阈值"""
        history.append(float(value))
        if (
            not self.use_adaptive_thresholds
            or len(history) < self.adaptive_min_samples
        ):
            return current
        arr = np.asarray(history, dtype=float)
        q_low, q_high = np.quantile(arr, quantiles)
        if not np.isfinite(q_low) or not np.isfinite(q_high) or np.isclose(q_low, q_high):
            return current
        return float(q_low), float(q_high)

    def encode_state(
        self,
        div_norm: float,
        improvement: float,
        mp_signal: Tuple[float, float, float] | None = None,
    ) -> int:
        """将连续状态编码为离散状态 ID
        
        Args:
            div_norm: 归一化的多样性值 [0, 1]
            improvement: 适应度改进率
            mp_signal: (mp_div_delta, mp_diff_norm, mp_fitness_gain)
        
        Returns:
            状态 ID
        """
        self._div_bins = self._update_bins(
            div_norm,
            self._div_history,
            self._div_bins,
            self.div_adaptive_quantiles,
        )
        self._imp_bins = self._update_bins(
            improvement,
            self._imp_history,
            self._imp_bins,
            self.imp_adaptive_quantiles,
        )
        mp_div_delta = 0.0
        mp_diff_norm = 0.0
        mp_fit_gain = 0.0
        if mp_signal is not None:
            mp_div_delta, mp_diff_norm, mp_fit_gain = mp_signal
        mp_effect_val = 0.5 * mp_fit_gain + 0.3 * mp_div_delta + 0.2 * mp_diff_norm
        self._mp_effect_bins = self._update_bins(
            mp_effect_val,
            self._mp_effect_history,
            self._mp_effect_bins,
            self.mp_effect_quantiles,
        )

        div_low, div_high = self._div_bins
        imp_low, imp_high = self._imp_bins
        mp_low, mp_high = self._mp_effect_bins

        # 多样性等级
        if div_norm < div_low:
            div_level = 0  # 低多样性
        elif div_norm <= div_high:
            div_level = 1  # 中等多样性
        else:
            div_level = 2  # 高多样性

        # 改进等级
        if improvement <= imp_low:
            imp_level = 0  # 停滞
        elif improvement <= imp_high:
            imp_level = 1  # 缓慢改进
        else:
            imp_level = 2  # 快速改进

        # MP 影响等级
        if mp_effect_val <= mp_low:
            mp_level = 0  # MP 负面或无益
        elif mp_effect_val <= mp_high:
            mp_level = 1  # 中性
        else:
            mp_level = 2  # MP 有益

        base_state = div_level * 9 + imp_level * 3 + mp_level
        state_id = int(base_state % self.num_states)
        return state_id

    @staticmethod
    def decode_state(state: int) -> tuple[int, int]:
        """将状态 ID 解码为多样性等级和改进等级
        
        Args:
            state: 状态 ID
        
        Returns:
            (多样性等级, 改进等级)
        """
        div_level = state // 3
        imp_level = state % 3
        return div_level, imp_level
    
    def select_action(self, state: int, g: int, g_max: int, improvement: Optional[float] = None) -> int:
        """选择动作（epsilon-greedy 策略）
        
        Args:
            state: 当前状态
            g: 当前代数
            g_max: 最大代数
            improvement: 当前改进率（用于自适应探索）
        
        Returns:
            选择的动作 ID
        """
        eps = self.epsilon(g, g_max, improvement)
        
        if np.random.rand() < eps:
            # 探索：随机选择
            action = np.random.randint(self.num_actions)
        else:
            # 利用：选择 Q 值最大的动作
            action = int(np.argmax(self.q_table[state]))
        
        # 记录
        self.counts[state, action] += 1
        self.action_history.append(action)
        
        return action

    def shape_reward(self, reward: float) -> float:
        """对奖励进行裁剪与平滑，降低噪声"""
        if self.reward_clip is not None:
            reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))
        if self.reward_smoothing > 0:
            reward = (
                (1.0 - self.reward_smoothing) * self._last_reward
                + self.reward_smoothing * reward
            )
        self._last_reward = reward
        self.reward_history.append(reward)
        return reward

    def get_alpha(self, state: int, g: int, g_max: int) -> float:
        """获取当前融合权重 α（0-1），前期高，后期低"""
        progress = g / max(1, g_max)
        decay = 1.0 - progress ** self.alpha_schedule_pow
        base = float(self.alpha_table[state])
        alpha = self.alpha_min + decay * (base - self.alpha_min)
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """更新 Q 表（Q-learning 更新规则）
        
        Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
        """
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        # 同步调整 alpha_table（鼓励正奖励时保持高 alpha，负奖励时降低）
        self.alpha_table[state] = np.clip(
            self.alpha_table[state] + self.alpha_lr * reward,
            self.alpha_min,
            self.alpha_max,
        )
        
    def get_policy(self) -> np.ndarray:
        """获取当前贪婪策略
        
        Returns:
            每个状态的最优动作 (num_states,)
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_state_action_visits(self) -> np.ndarray:
        """获取状态-动作访问次数
        
        Returns:
            访问计数矩阵 (num_states, num_actions)
        """
        return self.counts.copy()
    
    def get_statistics(self) -> dict:
        """获取学习统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        return {
            "total_updates": len(self.reward_history),
            "mean_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "std_reward": np.std(self.reward_history) if self.reward_history else 0.0,
            "q_table_mean": np.mean(self.q_table),
            "q_table_max": np.max(self.q_table),
            "most_selected_action": int(np.argmax(np.sum(self.counts, axis=0))),
            "state_visit_counts": np.sum(self.counts, axis=1).tolist(),
            "div_bins": self._div_bins,
            "imp_bins": self._imp_bins,
            "mp_effect_bins": self._mp_effect_bins,
            "alpha_mean": float(np.mean(self.alpha_table)),
        }
    
    def reset(self) -> None:
        """重置控制器状态（保留参数）"""
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.counts = np.zeros_like(self.q_table, dtype=int)
        self.reward_history = []
        self.action_history = []
        self._last_reward = 0.0
        self._div_history = deque(maxlen=self.adaptive_window)
        self._imp_history = deque(maxlen=self.adaptive_window)
        self._mp_effect_history = deque(maxlen=self.adaptive_window)
        self._div_bins = (self.DIV_LOW_THRESHOLD, self.DIV_HIGH_THRESHOLD)
        self._imp_bins = (self.IMP_STAGNANT_THRESHOLD, self.IMP_FAST_THRESHOLD)
        self._mp_effect_bins = (-0.05, 0.05)
        self.alpha_table = np.full(self.num_states, self.alpha_init, dtype=float)
    
    def save(self, path: str) -> None:
        """保存 Q 表和统计信息
        
        Args:
            path: 保存路径（.npz 格式）
        """
        np.savez(
            path,
            q_table=self.q_table,
            counts=self.counts,
            alpha_table=self.alpha_table,
            params={
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "reward_fitness_weight": self.reward_fitness_weight,
                "reward_diversity_weight": self.reward_diversity_weight,
                "reward_best_weight": self.reward_best_weight,
                "reward_clip": self.reward_clip,
                "reward_smoothing": self.reward_smoothing,
                "use_adaptive_thresholds": self.use_adaptive_thresholds,
                "adaptive_window": self.adaptive_window,
                "adaptive_min_samples": self.adaptive_min_samples,
                "stagnation_eps_boost": self.stagnation_eps_boost,
                "div_adaptive_quantiles": self.div_adaptive_quantiles,
                "imp_adaptive_quantiles": self.imp_adaptive_quantiles,
                "mp_effect_quantiles": self.mp_effect_quantiles,
                "alpha_init": self.alpha_init,
                "alpha_min": self.alpha_min,
                "alpha_max": self.alpha_max,
                "alpha_lr": self.alpha_lr,
                "alpha_schedule_pow": self.alpha_schedule_pow,
            },
        )
    
    @classmethod
    def load(cls, path: str) -> "RLController":
        """从文件加载 RL 控制器
        
        Args:
            path: 文件路径
        
        Returns:
            加载的 RLController 实例
        """
        data = np.load(path, allow_pickle=True)
        params = data["params"].item()
        
        controller = cls(
            num_states=data["q_table"].shape[0],
            num_actions=data["q_table"].shape[1],
            **params,
        )
        controller.q_table = data["q_table"]
        controller.counts = data["counts"]
        if "alpha_table" in data:
            controller.alpha_table = data["alpha_table"]
        else:
            controller.alpha_table = np.full(controller.num_states, controller.alpha_init, dtype=float)
        
        return controller
