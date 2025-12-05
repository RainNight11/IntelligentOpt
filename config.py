"""
配置模块

从 config.yaml 加载配置，提供类型安全的配置访问和辅助工具。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ga_core import GAConfig


# ============================================================================
# 一、配置数据类
# ============================================================================

@dataclass
class GABaseParams:
    """GA 基础参数"""
    population_size: int = 200
    dim: int = 30
    num_generations: int = 500
    p_c: float = 0.9
    p_m: float = 0.05
    p_m_min: float = 0.005
    p_m_max: float = 0.3
    explore_p_m_scale: float = 1.6
    exploit_p_m_scale: float = 0.6
    explore_sigma_scale: float = 1.4
    exploit_sigma_scale: float = 0.7
    elitism_count: int = 2
    early_stop_patience: int = 50
    early_stop_tolerance: float = 1e-8
    gaussian_sigma: float = 0.1
    tournament_k: int = 3


@dataclass
class MultiPopParams:
    """多种群参数"""
    num_subpops: int = 4
    migration_period: int = 10
    improvement_window: int = 10
    migration_count: int = 3
    migration_good_threshold: float = 0.01
    migration_bad_threshold: float = 0.001
    migration_noise_sigma: float = 0.02
    stagnation_div_threshold: float = 0.15
    stagnation_imp_threshold: float = 0.001
    stagnation_replace_frac: float = 0.15


@dataclass
class RLParams:
    """RL 参数"""
    num_states: int = 27
    num_actions: int = 6
    alpha: float = 0.3
    gamma: float = 0.9
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    reward_fitness_weight: float = 0.8
    reward_diversity_weight: float = 0.2
    reward_best_weight: float = 0.3
    reward_clip: float = 1.0
    reward_smoothing: float = 0.15
    use_adaptive_thresholds: bool = True
    adaptive_window: int = 60
    adaptive_min_samples: int = 15
    stagnation_eps_boost: float = 0.2
    div_adaptive_quantiles: Tuple[float, float] = (0.3, 0.7)
    imp_adaptive_quantiles: Tuple[float, float] = (0.25, 0.7)
    mp_effect_quantiles: Tuple[float, float] = (0.3, 0.7)
    # Alpha（MP-融合权重）控制
    alpha_init: float = 0.8
    alpha_min: float = 0.05
    alpha_max: float = 1.0
    alpha_lr: float = 0.05
    alpha_schedule_pow: float = 1.5
    per_subpop_controller: bool = True
    sync_q_interval: int = 60
    sync_q_tau: float = 0.25


@dataclass
class OutputParams:
    """输出参数"""
    figures_dir: str = "figures"
    results_dir: str = "results"
    save_every_seed: bool = True
    save_format: str = "csv"


@dataclass
class ExperimentConfig:
    """实验配置"""
    mode: str = "full"
    description: str = ""
    
    ga: GABaseParams = field(default_factory=GABaseParams)
    multipop: MultiPopParams = field(default_factory=MultiPopParams)
    rl: RLParams = field(default_factory=RLParams)
    output: OutputParams = field(default_factory=OutputParams)
    
    seeds: List[int] = field(default_factory=lambda: list(range(30)))
    algorithms: Tuple[str, ...] = ("GA_baseline", "GA_MP", "GA_RL", "GA_MP_RL")
    benchmarks: Tuple[str, ...] = ("rastrigin", "rosenbrock", "ackley")
    
    # 向后兼容属性
    @property
    def output_dir(self) -> str:
        return self.output.figures_dir
    
    @property
    def results_dir(self) -> str:
        return self.output.results_dir
    
    @property
    def save_every_seed(self) -> bool:
        return self.output.save_every_seed
    
    @property
    def ga_params(self) -> "GAParams":
        return GAParams(
            population_size=self.ga.population_size,
            num_generations=self.ga.num_generations,
            p_c=self.ga.p_c,
            p_m=self.ga.p_m,
            p_m_min=self.ga.p_m_min,
            p_m_max=self.ga.p_m_max,
            explore_p_m_scale=self.ga.explore_p_m_scale,
            exploit_p_m_scale=self.ga.exploit_p_m_scale,
            explore_sigma_scale=self.ga.explore_sigma_scale,
            exploit_sigma_scale=self.ga.exploit_sigma_scale,
            dim=self.ga.dim,
            elitism_count=self.ga.elitism_count,
            early_stop_patience=self.ga.early_stop_patience,
            early_stop_tolerance=self.ga.early_stop_tolerance,
            gaussian_sigma=self.ga.gaussian_sigma,
            tournament_k=self.ga.tournament_k,
            num_subpops=self.multipop.num_subpops,
            migration_period=self.multipop.migration_period,
            improvement_window=self.multipop.improvement_window,
            migration_count=self.multipop.migration_count,
            migration_good_threshold=self.multipop.migration_good_threshold,
            migration_bad_threshold=self.multipop.migration_bad_threshold,
            migration_noise_sigma=self.multipop.migration_noise_sigma,
            stagnation_div_threshold=self.multipop.stagnation_div_threshold,
            stagnation_imp_threshold=self.multipop.stagnation_imp_threshold,
            stagnation_replace_frac=self.multipop.stagnation_replace_frac,
        )


@dataclass
class GAParams:
    """向后兼容的扁平化 GA 参数"""
    population_size: int = 200
    num_generations: int = 500
    p_c: float = 0.9
    p_m: float = 0.05
    p_m_min: float = 0.005
    p_m_max: float = 0.3
    explore_p_m_scale: float = 1.6
    exploit_p_m_scale: float = 0.6
    explore_sigma_scale: float = 1.4
    exploit_sigma_scale: float = 0.7
    dim: int = 30
    elitism_count: int = 2
    early_stop_patience: int = 50
    early_stop_tolerance: float = 1e-8
    gaussian_sigma: float = 0.1
    tournament_k: int = 3
    num_subpops: int = 4
    migration_period: int = 10
    improvement_window: int = 10
    migration_count: int = 3
    migration_good_threshold: float = 0.01
    migration_bad_threshold: float = 0.001
    migration_noise_sigma: float = 0.02
    stagnation_div_threshold: float = 0.15
    stagnation_imp_threshold: float = 0.001
    stagnation_replace_frac: float = 0.15


# ============================================================================
# 二、配置加载器
# ============================================================================

class ConfigLoader:
    """YAML 配置加载器"""
    
    ALGORITHM_SWITCHES = {
    "GA_baseline": {"use_multipop": False, "use_rescue_migration": False, "use_rl": False},
    "GA_MP": {"use_multipop": True, "use_rescue_migration": True, "use_rl": False},
    "GA_RL": {"use_multipop": False, "use_rescue_migration": False, "use_rl": True},
    "GA_MP_RL": {"use_multipop": True, "use_rescue_migration": True, "use_rl": True},
}

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._load_yaml()
    
    def _load_yaml(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
    
    def get_mode_config(self, mode: str) -> ExperimentConfig:
        modes = self._raw_config.get("modes", {})
        if mode not in modes:
            raise ValueError(f"未知模式: {mode}。可用: {list(modes.keys())}")
        
        mode_cfg = modes[mode]
        ga_base = self._merge_dict(self._raw_config.get("ga", {}), mode_cfg.get("ga", {}))
        multipop_cfg = self._merge_dict(self._raw_config.get("multipop", {}), mode_cfg.get("multipop", {}))
        rl_cfg = self._raw_config.get("rl", {})
        output_cfg = self._raw_config.get("output", {})
        
        seeds_val = mode_cfg.get("seeds", 30)
        seeds = list(range(seeds_val)) if isinstance(seeds_val, int) else list(seeds_val)
        mode_output_dir = mode_cfg.get("output_dir", f"results/{mode}")
        
        return ExperimentConfig(
            mode=mode,
            description=mode_cfg.get("description", ""),
            ga=GABaseParams(**ga_base),
            multipop=MultiPopParams(**multipop_cfg),
            rl=RLParams(**rl_cfg),
            output=OutputParams(
                figures_dir=os.path.join(mode_output_dir, "figures"),
                results_dir=mode_output_dir,
                save_every_seed=mode_cfg.get("save_every_seed", output_cfg.get("save_every_seed", True)),
                save_format=output_cfg.get("save_format", "csv"),
            ),
            seeds=seeds,
            algorithms=tuple(mode_cfg.get("algorithms", ["GA_baseline", "GA_MP", "GA_RL", "GA_MP_RL"])),
            benchmarks=tuple(mode_cfg.get("benchmarks", ["rastrigin", "rosenbrock", "ackley"])),
        )
    
    def get_default_mode(self) -> str:
        return self._raw_config.get("default_mode", "full")
    
    def list_modes(self) -> List[str]:
        return list(self._raw_config.get("modes", {}).keys())
    
    def get_algorithm_switches(self, algo_name: str) -> Dict[str, bool]:
        algos = self._raw_config.get("algorithms", {})
        if algo_name in algos:
            algo_cfg = algos[algo_name]
            return {
                "use_multipop": algo_cfg.get("use_multipop", False),
                "use_rescue_migration": algo_cfg.get("use_rescue_migration", False),
                "use_rl": algo_cfg.get("use_rl", False),
            }
        return self.ALGORITHM_SWITCHES.get(algo_name, self.ALGORITHM_SWITCHES["GA_baseline"])
    
    @staticmethod
    def _merge_dict(base: Dict, override: Dict) -> Dict:
        result = base.copy()
        result.update(override)
        return result


# ============================================================================
# 三、全局便捷函数
# ============================================================================

_loader: Optional[ConfigLoader] = None


def get_loader(config_path: str = "config.yaml") -> ConfigLoader:
    """获取配置加载器单例"""
    global _loader
    if _loader is None or str(_loader.config_path) != str(config_path):
        _loader = ConfigLoader(config_path)
    return _loader


def load_config(mode: Optional[str] = None, config_path: str = "config.yaml") -> ExperimentConfig:
    """加载指定模式的配置"""
    loader = get_loader(config_path)
    if mode is None:
        mode = loader.get_default_mode()
    return loader.get_mode_config(mode)


def build_ga_config(
    ga_params: GAParams,
    algo_name: str,
    rl_params: Optional[Dict[str, Any]] = None,
) -> GAConfig:
    """构建 GAConfig"""
    loader = get_loader()
    switches = loader.get_algorithm_switches(algo_name)
    params_dict = ga_params.__dict__.copy()
    if rl_params is not None:
        params_dict["rl_params"] = rl_params
    return GAConfig(**params_dict, **switches)


# ============================================================================
# 四、配置打印与验证
# ============================================================================

def print_config(config: ExperimentConfig, title: Optional[str] = None) -> None:
    """打印配置摘要"""
    title = title or f"[{config.mode}] {config.description}"
    
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    
    print(f"\n【GA 基础参数】")
    print(f"  种群大小:     {config.ga.population_size}")
    print(f"  问题维度:     {config.ga.dim}")
    print(f"  进化代数:     {config.ga.num_generations}")
    print(f"  交叉概率:     {config.ga.p_c}")
    print(f"  变异概率:     {config.ga.p_m}")
    print(f"  高斯变异σ:    {config.ga.gaussian_sigma}")
    print(f"  锦标赛k:      {config.ga.tournament_k}")
    
    print(f"\n【精英保留与早停】")
    print(f"  精英数量:     {config.ga.elitism_count}")
    print(f"  早停耐心:     {config.ga.early_stop_patience}")
    print(f"  早停容忍:     {config.ga.early_stop_tolerance}")
    
    print(f"\n【多种群参数】")
    print(f"  子种群数量:   {config.multipop.num_subpops}")
    print(f"  迁移周期:     {config.multipop.migration_period}")
    print(f"  迁移个体数:   {config.multipop.migration_count}")
    print(f"  改进窗口:     {config.multipop.improvement_window}")
    
    print(f"\n【RL 参数】")
    print(f"  状态数:       {config.rl.num_states}")
    print(f"  动作数:       {config.rl.num_actions}")
    print(f"  学习率:       {config.rl.alpha}")
    print(f"  折扣因子:     {config.rl.gamma}")
    print(f"  探索率:       {config.rl.epsilon_start} → {config.rl.epsilon_end}")
    print(f"  奖励权重:     fit={config.rl.reward_fitness_weight}, div={config.rl.reward_diversity_weight}, best={config.rl.reward_best_weight}")
    print(f"  奖励裁剪/平滑: clip={config.rl.reward_clip}, smooth={config.rl.reward_smoothing}")
    adapt = "启用" if config.rl.use_adaptive_thresholds else "禁用"
    print(f"  自适应状态:   {adapt}, window={config.rl.adaptive_window}, boost={config.rl.stagnation_eps_boost}")
    print(f"  MP影响分箱:   mp_effect_q={config.rl.mp_effect_quantiles}")
    print(f"  MP融合α:     init={config.rl.alpha_init}, range=[{config.rl.alpha_min}, {config.rl.alpha_max}], lr={config.rl.alpha_lr}")
    print(f"  α调度:       pow={config.rl.alpha_schedule_pow}")
    agent_mode = "多子种群独立" if config.rl.per_subpop_controller else "全局单控制器"
    print(f"  RL 控制器:    {agent_mode}, sync_interval={config.rl.sync_q_interval}, tau={config.rl.sync_q_tau}")
    
    print(f"\n【实验设置】")
    print(f"  随机种子数:   {len(config.seeds)}")
    print(f"  算法:         {', '.join(config.algorithms)}")
    print(f"  基准函数:     {', '.join(config.benchmarks)}")
    print(f"  输出目录:     {config.output.results_dir}")
    print(f"  保存格式:     {config.output.save_format}")
    
    print(f"{'='*70}\n")


def print_modes(config_path: str = "config.yaml") -> None:
    """打印所有可用模式"""
    try:
        loader = get_loader(config_path)
        print("\n可用实验模式:")
        print("-" * 60)
        for mode in loader.list_modes():
            cfg = loader.get_mode_config(mode)
            print(f"  {mode:15s} : {cfg.description}")
        print("-" * 60)
    except FileNotFoundError:
        print("配置文件 config.yaml 不存在")


def print_algorithms() -> None:
    """打印所有可用算法"""
    print("\n可用算法:")
    print("-" * 50)
    for name, switches in ConfigLoader.ALGORITHM_SWITCHES.items():
        features = []
        if switches["use_multipop"]:
            features.append("多种群")
        if switches["use_rescue_migration"]:
            features.append("救援迁移")
        if switches["use_rl"]:
            features.append("RL控制")
        print(f"  {name:15s} : {', '.join(features) if features else '基础版'}")
    print("-" * 50)


def validate_config(config: ExperimentConfig) -> List[str]:
    """验证配置，返回错误列表"""
    errors = []
    if config.ga.population_size <= 0:
        errors.append("种群大小必须为正整数")
    if config.ga.num_generations <= 0:
        errors.append("进化代数必须为正整数")
    if not 0 <= config.ga.p_c <= 1:
        errors.append("交叉概率必须在 [0, 1] 范围内")
    if not 0 <= config.ga.p_m <= 1:
        errors.append("变异概率必须在 [0, 1] 范围内")
    if config.multipop.num_subpops > 0:
        if config.ga.population_size % config.multipop.num_subpops != 0:
            errors.append(f"种群大小({config.ga.population_size})必须能被子种群数量({config.multipop.num_subpops})整除")
    if not 0 <= config.rl.alpha <= 1:
        errors.append("学习率必须在 [0, 1] 范围内")
    if not 0 <= config.rl.gamma <= 1:
        errors.append("折扣因子必须在 [0, 1] 范围内")
    for algo in config.algorithms:
        if algo not in ConfigLoader.ALGORITHM_SWITCHES:
            errors.append(f"未知算法: {algo}")
    return errors


def check_config(config: ExperimentConfig) -> bool:
    """检查配置是否有效"""
    errors = validate_config(config)
    if errors:
        print("配置验证失败:")
        for err in errors:
            print(f"  - {err}")
        return False
    print("✓ 配置验证通过")
    return True


# ============================================================================
# 五、实用工具
# ============================================================================

def estimate_runtime(config: ExperimentConfig, time_per_gen_ms: float = 10) -> str:
    """估算实验运行时间"""
    total_gens = len(config.seeds) * len(config.algorithms) * len(config.benchmarks) * config.ga.num_generations
    total_seconds = total_gens * time_per_gen_ms / 1000
    
    if total_seconds < 60:
        return f"约 {total_seconds:.1f} 秒"
    elif total_seconds < 3600:
        return f"约 {total_seconds/60:.1f} 分钟"
    else:
        return f"约 {total_seconds/3600:.1f} 小时"


def compare_configs(config1: ExperimentConfig, config2: ExperimentConfig, 
                    name1: str = "Config1", name2: str = "Config2") -> None:
    """对比两个配置的差异"""
    print(f"\n配置对比: {name1} vs {name2}")
    print("=" * 70)
    
    ga_params = [
        ("种群大小", "ga.population_size"),
        ("进化代数", "ga.num_generations"),
        ("问题维度", "ga.dim"),
        ("交叉概率", "ga.p_c"),
        ("变异概率", "ga.p_m"),
    ]
    
    print(f"\n{'参数':<15} {name1:<20} {name2:<20} {'差异':<10}")
    print("-" * 65)
    
    for name, attr_path in ga_params:
        parts = attr_path.split(".")
        val1, val2 = config1, config2
        for part in parts:
            val1 = getattr(val1, part)
            val2 = getattr(val2, part)
        diff = "✗" if val1 != val2 else ""
        print(f"{name:<15} {str(val1):<20} {str(val2):<20} {diff:<10}")
    
    print(f"\n{'种子数':<15} {len(config1.seeds):<20} {len(config2.seeds):<20} {'✗' if len(config1.seeds) != len(config2.seeds) else ''}")
    print(f"{'算法数':<15} {len(config1.algorithms):<20} {len(config2.algorithms):<20} {'✗' if len(config1.algorithms) != len(config2.algorithms) else ''}")
    print("=" * 70)


def get_experiment_summary(config: ExperimentConfig) -> Dict:
    """获取实验摘要"""
    return {
        "mode": config.mode,
        "description": config.description,
        "total_runs": len(config.seeds) * len(config.algorithms) * len(config.benchmarks),
        "seeds": len(config.seeds),
        "algorithms": list(config.algorithms),
        "benchmarks": list(config.benchmarks),
        "generations": config.ga.num_generations,
        "population_size": config.ga.population_size,
        "estimated_time": estimate_runtime(config),
    }


# ============================================================================
# 六、向后兼容
# ============================================================================

class _LazyConfig:
    def __init__(self, mode: str):
        self._mode = mode
        self._config: Optional[ExperimentConfig] = None
    
    def __getattr__(self, name: str):
        if self._config is None:
            try:
                self._config = load_config(self._mode)
            except FileNotFoundError:
                self._config = ExperimentConfig(mode=self._mode)
        return getattr(self._config, name)


FULL_CONFIG = _LazyConfig("full")
QUICK_CONFIG = _LazyConfig("quick")
DEBUG_CONFIG = _LazyConfig("debug")
ABLATION_CONFIG = _LazyConfig("ablation")

ALGORITHM_VARIANTS = ConfigLoader.ALGORITHM_SWITCHES
ALGORITHM_SWITCHES = ALGORITHM_VARIANTS


# ============================================================================
# 七、命令行工具
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="配置工具")
    parser.add_argument("--list-modes", "-l", action="store_true", help="列出所有模式")
    parser.add_argument("--list-algos", "-a", action="store_true", help="列出所有算法")
    parser.add_argument("--show", "-s", type=str, metavar="MODE", help="显示指定模式的配置")
    parser.add_argument("--validate", "-v", type=str, metavar="MODE", help="验证指定模式的配置")
    parser.add_argument("--compare", "-c", nargs=2, metavar=("MODE1", "MODE2"), help="对比两个模式")
    
    args = parser.parse_args()
    
    if args.list_modes:
        print_modes()
    elif args.list_algos:
        print_algorithms()
    elif args.show:
        cfg = load_config(args.show)
        print_config(cfg)
        print(f"预计运行时间: {estimate_runtime(cfg)}")
    elif args.validate:
        cfg = load_config(args.validate)
        check_config(cfg)
    elif args.compare:
        cfg1 = load_config(args.compare[0])
        cfg2 = load_config(args.compare[1])
        compare_configs(cfg1, cfg2, args.compare[0], args.compare[1])
    else:
        parser.print_help()
