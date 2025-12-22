"""
对比实验运行器

将 MPRL-GA（本文方法）与其他经典优化算法进行对比：
- Standard GA: GA_baseline（由 GA 框架实现）
- ACO: 连续蚁群优化 (ACOR)
- ES: Evolution Strategy
- PSO: 粒子群优化
- SA: 模拟退火
- DE: 差分进化

配置全部由 config.yaml 管理：
- comparison.our_methods: 本文方法（通常只包含 MPRL-GA）
- comparison.modes.*: 不同对比实验模式 (quick/medium/full)

使用方法:
    python compare_experiments.py --mode full      # 完整对比实验
    python compare_experiments.py --mode quick     # 快速测试
    python compare_experiments.py --mode medium    # 中等规模
    python compare_experiments.py --list           # 列出所有算法
    python compare_experiments.py --list-modes     # 列出所有模式
    python compare_experiments.py --config my.yaml # 指定配置文件
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import get_benchmark
from baseline_algorithms import (
    get_baseline_algorithm,
    list_baseline_algorithms,
    compute_diversity,
)
from ga_core import GARunner, GAConfig
from config import (
    ComparisonConfig,
    load_comparison_config,
    list_comparison_modes,
    get_baseline_algorithms,
)


# =============================================================================
# 本文 GA 系列算法封装
# =============================================================================


def run_our_algorithm(
    algo_name: str,
    benchmark_name: str,
    population_size: int,
    num_generations: int,
    dim: int,
    seed: int,
) -> Dict:
    """运行本文 GA 系列算法

    支持:
        - GA_baseline
        - GA_MP
        - GA_RL
        - MPRL-GA (== GA_MP_RL)
    """
    benchmark = get_benchmark(benchmark_name, dim=dim)

    switches = {
        "GA_baseline": {
            "use_multipop": False,
            "use_rescue_migration": False,
            "use_rl": False,
        },
        "GA_MP": {"use_multipop": True, "use_rescue_migration": True, "use_rl": False},
        "GA_RL": {
            "use_multipop": False,
            "use_rescue_migration": False,
            "use_rl": True,
        },
        "MPRL-GA": {
            "use_multipop": True,
            "use_rescue_migration": True,
            "use_rl": True,
        },
        "GA_MP_RL": {
            "use_multipop": True,
            "use_rescue_migration": True,
            "use_rl": True,
        },
    }

    if algo_name not in switches:
        raise ValueError(f"未知本文算法: {algo_name}")

    sw = switches[algo_name]

    config = GAConfig(
        population_size=population_size,
        num_generations=num_generations,
        dim=dim,
        p_c=0.9,
        p_m=0.05,
        elitism_count=2,
        early_stop_patience=0,
        use_multipop=sw["use_multipop"],
        use_rescue_migration=sw["use_rescue_migration"],
        use_rl=sw["use_rl"],
        num_subpops=4,
    )

    runner = GARunner(config)
    result = runner.run(benchmark, seed=seed)

    return {
        "best_curve": result["best_curve"],
        "diversity_curve": result["diversity_curve"],
        "final_best": result["final_best"],
    }


def run_baseline_algorithm(
    algo_name: str,
    benchmark_name: str,
    population_size: int,
    num_generations: int,
    dim: int,
    seed: int,
) -> Dict:
    """运行对比算法（PSO/DE/ES/SA/ACO 等）"""
    benchmark = get_benchmark(benchmark_name, dim=dim)
    algo = get_baseline_algorithm(algo_name)
    return algo.run(benchmark, population_size, num_generations, seed)


def run_single_experiment(
    algo_name: str,
    benchmark_name: str,
    cfg: ComparisonConfig,
    seeds: List[int],
    progress_prefix: str = "",
) -> Dict:
    """运行单个算法在单个基准上的多次独立实验"""
    best_curves = []
    div_curves = []
    final_bests = []

    is_our = algo_name in cfg.our_methods or algo_name in [
        "GA_baseline",
        "GA_MP",
        "GA_RL",
        "MPRL-GA",
        "GA_MP_RL",
    ]

    for idx, seed in enumerate(seeds):
        if progress_prefix:
            print(f"\r{progress_prefix} seed {idx+1}/{len(seeds)}", end="", flush=True)

        if is_our:
            res = run_our_algorithm(
                algo_name,
                benchmark_name,
                cfg.population_size,
                cfg.num_generations,
                cfg.dim,
                seed,
            )
        else:
            res = run_baseline_algorithm(
                algo_name,
                benchmark_name,
                cfg.population_size,
                cfg.num_generations,
                cfg.dim,
                seed,
            )

        best_curves.append(res["best_curve"])
        div_curves.append(res["diversity_curve"])
        final_bests.append(res["final_best"])

    if progress_prefix:
        print()

    max_len = max(len(c) for c in best_curves)
    padded_best = []
    padded_div = []
    for bc, dc in zip(best_curves, div_curves):
        if len(bc) < max_len:
            bc = np.concatenate([bc, np.full(max_len - len(bc), bc[-1])])
            dc = np.concatenate([dc, np.full(max_len - len(dc), dc[-1])])
        padded_best.append(bc)
        padded_div.append(dc)

    return {
        "best_mean": np.mean(padded_best, axis=0),
        "best_std": np.std(padded_best, axis=0),
        "div_mean": np.mean(padded_div, axis=0),
        "final_mean": float(np.mean(final_bests)),
        "final_std": float(np.std(final_bests)),
        "final_bests": final_bests,
    }


# =============================================================================
# 结果保存与绘图
# =============================================================================


def save_comparison_results(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: str,
) -> None:
    """保存对比实验结果到多个 CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # 主结果表
    main_results_path = os.path.join(output_dir, "comparison_results.csv")
    with open(main_results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        all_algos = sorted({a for bench in all_results.values() for a in bench.keys()})
        header = ["Benchmark"] + [f"{algo}_Mean" for algo in all_algos] + [
            f"{algo}_Std" for algo in all_algos
        ]
        writer.writerow(header)

        for bench_name, bench_results in all_results.items():
            row = [bench_name]
            for algo in all_algos:
                if algo in bench_results:
                    row.append(f"{bench_results[algo]['final_mean']:.6e}")
                else:
                    row.append("N/A")
            for algo in all_algos:
                if algo in bench_results:
                    row.append(f"{bench_results[algo]['final_std']:.6e}")
                else:
                    row.append("N/A")
            writer.writerow(row)

    # 排名表
    ranking_path = os.path.join(output_dir, "ranking_results.csv")
    with open(ranking_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        all_algos = sorted({a for bench in all_results.values() for a in bench.keys()})
        writer.writerow(["Benchmark"] + all_algos + ["Best Algorithm"])

        algo_ranks = {algo: [] for algo in all_algos}
        for bench_name, bench_results in all_results.items():
            means = {algo: bench_results[algo]["final_mean"] for algo in all_algos}
            sorted_algos = sorted(means.items(), key=lambda x: x[1])
            ranks = {}
            for rank, (algo, _) in enumerate(sorted_algos, start=1):
                ranks[algo] = rank
                algo_ranks[algo].append(rank)
            row = [bench_name] + [ranks.get(a, "N/A") for a in all_algos] + [
                sorted_algos[0][0]
            ]
            writer.writerow(row)

        writer.writerow([])
        avg_ranks = [
            f"{np.mean(algo_ranks[a]):.2f}" if algo_ranks[a] else "N/A"
            for a in all_algos
        ]
        writer.writerow(["Avg Rank"] + avg_ranks + [""])

    # 简洁汇总表（Mean±Std）
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        all_algos = sorted({a for bench in all_results.values() for a in bench.keys()})
        header = ["Benchmark"] + all_algos
        writer.writerow(header)

        for bench_name, bench_results in all_results.items():
            row = [bench_name]
            for algo in all_algos:
                if algo in bench_results:
                    mean = bench_results[algo]["final_mean"]
                    std = bench_results[algo]["final_std"]
                    row.append(f"{mean:.4e}±{std:.2e}")
                else:
                    row.append("N/A")
            writer.writerow(row)

        # 平均排名
        writer.writerow([])
        avg_row = ["Avg Rank"]
        for algo in all_algos:
            ranks = []
            for bench_results in all_results.values():
                means = {a: r["final_mean"] for a, r in bench_results.items()}
                sorted_means = sorted(means.items(), key=lambda x: x[1])
                for rank, (a, _) in enumerate(sorted_means, start=1):
                    if a == algo:
                        ranks.append(rank)
                        break
            avg_row.append(f"{np.mean(ranks):.2f}" if ranks else "N/A")
        writer.writerow(avg_row)


def plot_comparison_convergence(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: str,
) -> None:
    """绘制对比收敛曲线（多子图）"""
    os.makedirs(output_dir, exist_ok=True)

    n_benchmarks = len(all_results)
    n_cols = 3
    n_rows = (n_benchmarks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_benchmarks == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = {
        "GA_baseline": "#FF6B6B",
        "ACO": "#FFA500",
        "ES": "#1E90FF",
        "PSO": "#FFD700",
        "SA": "#8A2BE2",
        "DE": "#32CD32",
        "MPRL-GA": "#2E8B57",
    }

    for idx, (bench_name, bench_results) in enumerate(all_results.items()):
        ax = axes[idx]

        for algo_name, res in bench_results.items():
            color = colors.get(algo_name, "#808080")
            lw = 2.5 if algo_name == "MPRL-GA" else 1.2
            ls = "-" if algo_name == "MPRL-GA" else "--"
            ax.plot(res["best_mean"], label=algo_name, color=color, linewidth=lw, linestyle=ls)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.set_title(bench_name)
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

        finals = np.array([r["best_mean"][-1] for r in bench_results.values()])
        if np.all(finals > 0):
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=1e-10)

    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_convergence.png"), dpi=200)
    plt.close()


# =============================================================================
# 主流程
# =============================================================================


def run_comparison_experiments(cfg: ComparisonConfig) -> Dict:
    """运行完整对比实验"""
    os.makedirs(cfg.output_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, Dict]] = {}
    all_algorithms = list(cfg.our_methods) + list(cfg.baseline_methods)

    total_tasks = len(cfg.benchmarks) * len(all_algorithms)
    task_idx = 0

    for bench_name in cfg.benchmarks:
        print("\n" + "=" * 60)
        print(f"Benchmark: {bench_name}")
        print("=" * 60)

        all_results[bench_name] = {}

        for algo_name in all_algorithms:
            task_idx += 1
            print(f"[{task_idx}/{total_tasks}] Running {algo_name}...")

            res = run_single_experiment(
                algo_name,
                bench_name,
                cfg,
                cfg.seeds,
                progress_prefix=f"  {algo_name}",
            )

            all_results[bench_name][algo_name] = res
            print(f"  -> Final: {res['final_mean']:.4e} ± {res['final_std']:.2e}")

    print("\n" + "=" * 60)
    print("保存结果...")
    print("=" * 60)

    save_comparison_results(all_results, cfg.output_dir)
    plot_comparison_convergence(all_results, cfg.output_dir)

    return all_results


def print_comparison_config(cfg: ComparisonConfig) -> None:
    """打印对比实验配置摘要"""
    print("\n" + "=" * 60)
    print(f" 对比实验配置 [{cfg.mode}]")
    print("=" * 60)
    print(f"描述: {cfg.description}")
    print(f"种子数: {len(cfg.seeds)}")
    print(f"基准函数: {list(cfg.benchmarks)}")
    print(f"种群大小: {cfg.population_size}")
    print(f"代数: {cfg.num_generations}")
    print(f"维度: {cfg.dim}")
    print(f"本文方法: {list(cfg.our_methods)}")
    print(f"对比方法: {list(cfg.baseline_methods)}")
    print(f"输出目录: {cfg.output_dir}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比实验运行器 - MPRL-GA vs 经典智能优化算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python compare_experiments.py --mode full      # 完整对比实验
  python compare_experiments.py --mode quick     # 快速测试
  python compare_experiments.py --mode medium    # 中等规模
  python compare_experiments.py --list           # 列出所有算法
  python compare_experiments.py --list-modes     # 列出所有模式
        """,
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="quick",
        help="对比实验模式（在 config.yaml 的 comparison.modes 中定义）",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="列出所有可用算法",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="列出所有对比实验模式",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )

    args = parser.parse_args()

    if args.list_modes:
        print("\n可用的对比实验模式:")
        for mode in list_comparison_modes(args.config):
            try:
                cfg = load_comparison_config(mode, args.config)
                print(f"  - {mode}: {cfg.description}")
            except Exception:
                print(f"  - {mode}")
        return

    if args.list:
        print("\n本文方法:")
        print("  - MPRL-GA: 多种群 RL 增强遗传算法（本文提出）")
        print("\n对比算法（config.yaml 分组）:")
        baseline_groups = get_baseline_algorithms(args.config)
        for cat, algos in baseline_groups.items():
            print(f"  [{cat}]")
            for name in algos:
                print(f"    - {name}")
        print("\n全部已实现算法:")
        for name in list_baseline_algorithms():
            print(f"  - {name}")
        return

    try:
        cfg = load_comparison_config(args.mode, args.config)
    except ValueError as e:
        print(f"错误: {e}")
        print("\n使用 --list-modes 查看可用模式")
        return
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
        return

    print_comparison_config(cfg)
    run_comparison_experiments(cfg)
    print("\n对比实验完成！")


if __name__ == "__main__":
    main()


