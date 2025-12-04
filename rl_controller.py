"""
Q-learning 控制器

用于自适应选择遗传算子组合，根据种群状态（多样性、改进程度）
动态调整选择、交叉、变异算子的组合。

状态空间 (9个状态):
    - 多样性等级: 低(0) / 中(1) / 高(2)
    - 改进等级: 停滞(0) / 缓慢(1) / 快速(2)
    - 状态编码: div_level * 3 + imp_level

动作空间 (6个动作):
    0: 轮盘选择 + 均匀交叉 + 均匀变异
    1: 锦标赛选择 + 均匀交叉 + 均匀变异
    2: 锦标赛选择 + 两点交叉 + 高斯变异 (默认)
    3: 轮盘选择 + 两点交叉 + 高斯变异
    4: 排序选择 + 两点交叉 + 自适应高斯变异
    5: 排序选择 + 均匀交叉 + 自适应高斯变异
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RLController:
    """Q-learning 强化学习控制器"""
    
    # 状态离散化阈值
    DIV_LOW_THRESHOLD = 0.3
    DIV_HIGH_THRESHOLD = 0.7
    IMP_STAGNANT_THRESHOLD = 0.001
    IMP_FAST_THRESHOLD = 0.01
    
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
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.reward_fitness_weight = reward_fitness_weight
        self.reward_diversity_weight = reward_diversity_weight
        
        # Q 表和动作计数
        self.q_table = np.zeros((num_states, num_actions), dtype=float)
        self.counts = np.zeros_like(self.q_table, dtype=int)
        
        # 学习历史（用于分析）
        self.reward_history: list[float] = []
        self.action_history: list[int] = []

    def epsilon(self, g: int, g_max: int) -> float:
        """计算当前探索率（线性衰减）
        
        Args:
            g: 当前代数
            g_max: 最大代数
        
        Returns:
            当前探索率
        """
        progress = g / max(1, g_max)
        eps = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        return float(max(self.epsilon_end, eps))

    @staticmethod
    def encode_state(div_norm: float, improvement: float) -> int:
        """将连续状态编码为离散状态 ID
        
        Args:
            div_norm: 归一化的多样性值 [0, 1]
            improvement: 适应度改进率
        
        Returns:
            状态 ID (0-8)
        """
        # 多样性等级
        if div_norm < RLController.DIV_LOW_THRESHOLD:
            div_level = 0  # 低多样性
        elif div_norm <= RLController.DIV_HIGH_THRESHOLD:
            div_level = 1  # 中等多样性
        else:
            div_level = 2  # 高多样性

        # 改进等级
        if improvement <= RLController.IMP_STAGNANT_THRESHOLD:
            imp_level = 0  # 停滞
        elif improvement <= RLController.IMP_FAST_THRESHOLD:
            imp_level = 1  # 缓慢改进
        else:
            imp_level = 2  # 快速改进

        return div_level * 3 + imp_level

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
    
    def select_action(self, state: int, g: int, g_max: int) -> int:
        """选择动作（epsilon-greedy 策略）
        
        Args:
            state: 当前状态
            g: 当前代数
            g_max: 最大代数
        
        Returns:
            选择的动作 ID
        """
        eps = self.epsilon(g, g_max)
        
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
        
        # 记录奖励历史
        self.reward_history.append(reward)
    
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
        }
    
    def reset(self) -> None:
        """重置控制器状态（保留参数）"""
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.counts = np.zeros_like(self.q_table, dtype=int)
        self.reward_history = []
        self.action_history = []
    
    def save(self, path: str) -> None:
        """保存 Q 表和统计信息
        
        Args:
            path: 保存路径（.npz 格式）
        """
        np.savez(
            path,
            q_table=self.q_table,
            counts=self.counts,
            params={
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "reward_fitness_weight": self.reward_fitness_weight,
                "reward_diversity_weight": self.reward_diversity_weight,
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
        
        return controller
