"""
实验运行器

支持 GA baseline、多种群、RL增强、混合算法的实验运行和结果保存。
结果以 CSV 格式保存，方便查看和分析。

使用方法:
    python experiments.py --mode full     # 完整实验
    python experiments.py --mode quick    # 快速测试
    python experiments.py --mode ablation # 消融实验
    python experiments.py --list-modes    # 列出所有模式
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import BENCHMARKS
from config import (
    ExperimentConfig,
    ConfigLoader,
    build_ga_config,
    load_config,
    print_config,
    print_modes,
    check_config,
    estimate_runtime,
)
from ga_core import GARunner


def render_progress(prefix: str, idx: int, total: int, bar_width: int = 30):
    """渲染进度条"""
    done = int(bar_width * idx / max(1, total))
    bar = "#" * done + "-" * (bar_width - done)
    print(f"{prefix} [{bar}] {idx}/{total}", end="\r", flush=True)


def run_one_experiment(
    benchmark_name: str,
    algo_name: str,
    cfg: ExperimentConfig,
    seeds: List[int],
    progress_prefix: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """运行单个实验（指定基准函数和算法）
    
    Args:
        benchmark_name: 基准函数名称
        algo_name: 算法名称
        cfg: 实验配置
        seeds: 随机种子列表
        progress_prefix: 进度条前缀
        save_path: 保存路径（不含扩展名，会自动添加 .csv）
    
    Returns:
        实验结果字典
    """
    benchmark = BENCHMARKS[benchmark_name]

    best_curves = []
    div_curves = []
    final_bests = []
    rl_counts_total: Optional[np.ndarray] = None
    rl_kwargs = vars(cfg.rl).copy()

    total_runs = len(seeds)
    for idx, seed in enumerate(seeds, 1):
        if progress_prefix:
            render_progress(f"{progress_prefix} (seed={seed})", idx, total_runs)
        ga_cfg = build_ga_config(cfg.ga_params, algo_name, rl_params=rl_kwargs)
        runner = GARunner(ga_cfg)
        res = runner.run(benchmark, seed=seed, rl_controller=None)
        best_curves.append(res["best_curve"])
        div_curves.append(res["diversity_curve"])
        final_bests.append(res["final_best"])
        if res["rl_counts"] is not None:
            if rl_counts_total is None:
                rl_counts_total = np.zeros_like(res["rl_counts"])
            rl_counts_total += res["rl_counts"]
        
        # 中途保存
        if save_path and cfg.save_every_seed:
            _save_results_csv(
                save_path,
                best_curves,
                div_curves,
                final_bests,
                rl_counts_total if (ga_cfg.use_rl and rl_counts_total is not None) else None,
                seeds[:idx],
            )
    
    if progress_prefix:
        print()

    # 处理不同长度的曲线（早停导致）
    max_len = max(len(c) for c in best_curves)
    padded_best = []
    padded_div = []
    for bc, dc in zip(best_curves, div_curves):
        if len(bc) < max_len:
            bc = np.concatenate([bc, np.full(max_len - len(bc), bc[-1])])
            dc = np.concatenate([dc, np.full(max_len - len(dc), dc[-1])])
        padded_best.append(bc)
        padded_div.append(dc)
    
    best_mean = np.mean(padded_best, axis=0)
    best_std = np.std(padded_best, axis=0)
    div_mean = np.mean(padded_div, axis=0)
    final_mean = float(np.mean(final_bests))
    final_std = float(np.std(final_bests))

    out = {
        "best_mean": best_mean,
        "best_std": best_std,
        "div_mean": div_mean,
        "final_mean": final_mean,
        "final_std": final_std,
        "final_bests": final_bests,  # 每个seed的最终值（用于统计检验）
        "rl_counts": rl_counts_total if ga_cfg.use_rl else None,
    }
    
    if save_path:
        _save_results_csv(
            save_path,
            best_curves,
            div_curves,
            final_bests,
            rl_counts_total if (ga_cfg.use_rl and rl_counts_total is not None) else None,
            seeds,
        )
    
    return out


def _save_results_csv(
    base_path: str,
    best_curves: List[np.ndarray],
    div_curves: List[np.ndarray],
    final_bests: List[float],
    rl_counts: Optional[np.ndarray],
    seeds: List[int],
) -> None:
    """将实验结果保存为 CSV 格式
    
    生成以下文件：
    - {base_path}_curves.csv: 收敛曲线数据
    - {base_path}_summary.csv: 汇总统计
    - {base_path}_rl_counts.csv: RL动作计数（如果适用）
    
    Args:
        base_path: 基础路径（不含扩展名）
        best_curves: 各 seed 的最优适应度曲线
        div_curves: 各 seed 的多样性曲线
        final_bests: 各 seed 的最终最优值
        rl_counts: RL 动作统计矩阵
        seeds: 随机种子列表
    """
    if not best_curves:
        return
    
    # 移除可能的 .npz 扩展名
    if base_path.endswith('.npz'):
        base_path = base_path[:-4]
    
    # 处理不同长度的曲线（早停导致）：填充到最大长度
    max_len = max(len(c) for c in best_curves)
    padded_best = []
    padded_div = []
    for bc, dc in zip(best_curves, div_curves):
        if len(bc) < max_len:
            # 用最后一个值填充
            bc = np.concatenate([bc, np.full(max_len - len(bc), bc[-1])])
            dc = np.concatenate([dc, np.full(max_len - len(dc), dc[-1])])
        padded_best.append(bc)
        padded_div.append(dc)
    
    arr_best = np.stack(padded_best)
    arr_div = np.stack(padded_div)
    num_seeds, num_gens = arr_best.shape
    
    # ========== 1. 保存收敛曲线 ==========
    curves_path = f"{base_path}_curves.csv"
    with open(curves_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 表头：Generation, Best_Mean, Best_Std, Div_Mean, Seed_0, Seed_1, ...
        header = ['Generation', 'Best_Mean', 'Best_Std', 'Div_Mean']
        header += [f'Best_Seed_{s}' for s in seeds]
        header += [f'Div_Seed_{s}' for s in seeds]
        writer.writerow(header)
        
        # 计算统计量
        best_mean = arr_best.mean(axis=0)
        best_std = arr_best.std(axis=0)
        div_mean = arr_div.mean(axis=0)
        
        # 写入每一代的数据
        for g in range(num_gens):
            row = [g, best_mean[g], best_std[g], div_mean[g]]
            row += [arr_best[i, g] for i in range(num_seeds)]
            row += [arr_div[i, g] for i in range(num_seeds)]
            writer.writerow(row)
    
    # ========== 2. 保存汇总统计 ==========
    summary_path = f"{base_path}_summary.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入元信息
        writer.writerow(['项目', '值'])
        writer.writerow(['随机种子数', num_seeds])
        writer.writerow(['总代数', num_gens])
        writer.writerow([''])
        
        # 最终结果统计
        writer.writerow(['最终适应度统计', ''])
        writer.writerow(['均值', np.mean(final_bests)])
        writer.writerow(['标准差', np.std(final_bests)])
        writer.writerow(['最小值', np.min(final_bests)])
        writer.writerow(['最大值', np.max(final_bests)])
        writer.writerow(['中位数', np.median(final_bests)])
        writer.writerow([''])
        
        # 各 seed 的最终结果
        writer.writerow(['各Seed最终结果', ''])
        writer.writerow(['Seed', 'Final_Best'])
        for s, fb in zip(seeds, final_bests):
            writer.writerow([s, fb])
    
    # ========== 3. 保存 RL 动作计数（如果有）==========
    if rl_counts is not None and rl_counts.sum() > 0:
        rl_path = f"{base_path}_rl_counts.csv"
        with open(rl_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 表头说明
            writer.writerow(['# RL 动作选择统计'])
            writer.writerow([f'# 行: 状态 (0-{rl_counts.shape[0]-1}), 列: 动作 (0-{rl_counts.shape[1]-1})'])
            writer.writerow(['# 状态编码: div_level * 3 + imp_level（阈值为自适应估计）'])
            writer.writerow(['#   div_level: 0=低多样性, 1=中, 2=高'])
            writer.writerow(['#   imp_level: 0=停滞, 1=缓慢, 2=快速'])
            if rl_counts.shape[1] == 6:
                writer.writerow(['# 动作编码:'])
                writer.writerow(['#   0: 轮盘选择 + 均匀交叉 + 均匀变异'])
                writer.writerow(['#   1: 锦标赛选择 + 均匀交叉 + 均匀变异'])
                writer.writerow(['#   2: 锦标赛选择 + 两点交叉 + 高斯变异'])
                writer.writerow(['#   3: 轮盘选择 + 两点交叉 + 高斯变异'])
                writer.writerow(['#   4: 排序选择 + 两点交叉 + 自适应变异'])
                writer.writerow(['#   5: 排序选择 + 均匀交叉 + 自适应变异'])
            writer.writerow([''])
            
            # 表头
            header = ['State/Action'] + [f'Action_{i}' for i in range(rl_counts.shape[1])] + ['Total']
            writer.writerow(header)
            
            # 数据行
            default_state_names = [
                'S0(低div,停滞)', 'S1(低div,慢)', 'S2(低div,快)',
                'S3(中div,停滞)', 'S4(中div,慢)', 'S5(中div,快)',
                'S6(高div,停滞)', 'S7(高div,慢)', 'S8(高div,快)',
            ]
            for i, row in enumerate(rl_counts):
                if rl_counts.shape[0] == len(default_state_names):
                    name = default_state_names[i]
                else:
                    name = f'State_{i}'
                writer.writerow([name] + list(row) + [row.sum()])
            
            # 各动作总计
            writer.writerow(['Total'] + list(rl_counts.sum(axis=0)) + [rl_counts.sum()])


def plot_convergence(
    benchmark_name: str,
    results: Dict[str, Dict],
    output_dir: str,
):
    """绘制收敛曲线图"""
    gens = np.arange(len(next(iter(results.values()))["best_mean"]))
    
    plt.figure(figsize=(7, 5))
    for name, res in results.items():
        plt.plot(gens, res["best_mean"], label=name)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness (mean)")
    plt.title(f"Convergence on {benchmark_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{benchmark_name}_convergence.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, res in results.items():
        plt.plot(gens, res["div_mean"], label=name)
    plt.xlabel("Generation")
    plt.ylabel("Diversity (normalized)")
    plt.title(f"Diversity on {benchmark_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{benchmark_name}_diversity.png"), dpi=200)
    plt.close()


def plot_rl_heatmap(benchmark_name: str, algo_name: str, counts: np.ndarray, output_dir: str):
    """绘制 RL 动作选择热力图"""
    plt.figure(figsize=(6, 4))
    plt.imshow(counts, cmap="viridis")
    plt.colorbar(label="Selection count")
    plt.xlabel(f"Action id (0-{counts.shape[1]-1})")
    plt.ylabel(f"State id (0-{counts.shape[0]-1})")
    plt.title(f"RL policy counts: {algo_name} on {benchmark_name}")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{benchmark_name}_{algo_name}_rl_counts.png"), dpi=200)
    plt.close()


def save_overall_summary(
    summary: Dict[str, Dict[str, Dict]],
    output_dir: str,
) -> None:
    """保存所有实验的汇总对比表
    
    Args:
        summary: 嵌套字典 {benchmark: {algorithm: results}}
        output_dir: 输出目录
    """
    summary_path = os.path.join(output_dir, "overall_summary.csv")
    
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 获取所有算法名
        all_algos = set()
        for bench_results in summary.values():
            all_algos.update(bench_results.keys())
        all_algos = sorted(all_algos)
        
        # 表头
        header = ['Benchmark', 'Algorithm', 'Final_Mean', 'Final_Std', 'Final_Mean±Std']
        writer.writerow(header)
        
        # 数据
        for bench_name, bench_results in summary.items():
            for algo_name in all_algos:
                if algo_name in bench_results:
                    res = bench_results[algo_name]
                    mean = res['final_mean']
                    std = res['final_std']
                    writer.writerow([
                        bench_name,
                        algo_name,
                        f"{mean:.6e}",
                        f"{std:.6e}",
                        f"{mean:.4e} ± {std:.4e}",
                    ])
            writer.writerow([])  # 空行分隔不同基准函数


# ============================================================================
# 论文输出功能
# ============================================================================

def perform_wilcoxon_test(
    all_final_bests: Dict[str, Dict[str, List[float]]],
    baseline: str = "GA_baseline",
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Dict]]:
    """执行 Wilcoxon 秩和检验
    
    Args:
        all_final_bests: {benchmark: {algorithm: [final_best_values]}}
        baseline: 基线算法名称
        alpha: 显著性水平
    
    Returns:
        {benchmark: {algorithm: {"p_value": float, "significant": bool, "better": bool}}}
    """
    from scipy import stats
    
    results = {}
    for bench_name, bench_data in all_final_bests.items():
        results[bench_name] = {}
        if baseline not in bench_data:
            continue
        baseline_values = bench_data[baseline]
        
        for algo_name, algo_values in bench_data.items():
            if algo_name == baseline:
                results[bench_name][algo_name] = {
                    "p_value": 1.0,
                    "significant": False,
                    "better": False,
                    "symbol": "",
                }
                continue
            
            try:
                # Wilcoxon 秩和检验（双侧）
                stat, p_value = stats.wilcoxon(baseline_values, algo_values)
                significant = p_value < alpha
                # 判断是否更好（均值更小）
                better = np.mean(algo_values) < np.mean(baseline_values)
                
                # 显著性符号
                if significant and better:
                    symbol = "+"  # 显著更好
                elif significant and not better:
                    symbol = "-"  # 显著更差
                else:
                    symbol = "≈"  # 无显著差异
                    
            except Exception:
                p_value = 1.0
                significant = False
                better = False
                symbol = "?"
            
            results[bench_name][algo_name] = {
                "p_value": p_value,
                "significant": significant,
                "better": better,
                "symbol": symbol,
            }
    
    return results


def save_statistical_results(
    all_final_bests: Dict[str, Dict[str, List[float]]],
    summary: Dict[str, Dict[str, Dict]],
    output_dir: str,
    baseline: str = "GA_baseline",
) -> None:
    """保存含统计检验的详细结果表
    
    生成包含均值、标准差、最优值、最差值、中位数、排名、显著性的完整表格
    """
    try:
        stat_results = perform_wilcoxon_test(all_final_bests, baseline)
    except ImportError:
        print("警告: scipy 未安装，跳过统计检验")
        stat_results = {}
    
    output_path = os.path.join(output_dir, "paper_results.csv")
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 表头
        header = [
            'Benchmark', 'Algorithm', 'Mean', 'Std', 'Min', 'Max', 'Median',
            'Rank', 'vs_Baseline', 'p_value'
        ]
        writer.writerow(header)
        
        for bench_name in all_final_bests.keys():
            bench_data = all_final_bests[bench_name]
            bench_summary = summary.get(bench_name, {})
            
            # 计算排名
            algo_means = [(algo, np.mean(vals)) for algo, vals in bench_data.items()]
            algo_means.sort(key=lambda x: x[1])
            ranks = {algo: i + 1 for i, (algo, _) in enumerate(algo_means)}
            
            for algo_name, values in bench_data.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                median_val = np.median(values)
                rank = ranks[algo_name]
                
                # 统计检验结果
                stat = stat_results.get(bench_name, {}).get(algo_name, {})
                symbol = stat.get("symbol", "")
                p_val = stat.get("p_value", "")
                
                writer.writerow([
                    bench_name,
                    algo_name,
                    f"{mean_val:.6e}",
                    f"{std_val:.6e}",
                    f"{min_val:.6e}",
                    f"{max_val:.6e}",
                    f"{median_val:.6e}",
                    rank,
                    symbol,
                    f"{p_val:.4f}" if isinstance(p_val, float) else p_val,
                ])
            
            writer.writerow([])  # 空行分隔
    
    # 保存排名汇总
    _save_ranking_summary(all_final_bests, output_dir)
    
    print(f"论文结果表已保存到: {output_path}")


def _save_ranking_summary(
    all_final_bests: Dict[str, Dict[str, List[float]]],
    output_dir: str,
) -> None:
    """保存算法排名汇总表"""
    ranking_path = os.path.join(output_dir, "ranking_summary.csv")
    
    # 收集所有算法
    all_algos = set()
    for bench_data in all_final_bests.values():
        all_algos.update(bench_data.keys())
    all_algos = sorted(all_algos)
    
    # 计算每个基准函数上的排名
    algo_ranks = {algo: [] for algo in all_algos}
    
    with open(ranking_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 表头
        header = ['Benchmark'] + list(all_algos)
        writer.writerow(header)
        
        for bench_name, bench_data in all_final_bests.items():
            # 计算排名
            algo_means = [(algo, np.mean(bench_data.get(algo, [float('inf')]))) 
                          for algo in all_algos]
            algo_means.sort(key=lambda x: x[1])
            ranks = {algo: i + 1 for i, (algo, _) in enumerate(algo_means)}
            
            row = [bench_name] + [ranks.get(algo, '-') for algo in all_algos]
            writer.writerow(row)
            
            for algo in all_algos:
                algo_ranks[algo].append(ranks.get(algo, len(all_algos)))
        
        # 平均排名
        writer.writerow([])
        avg_ranks = [f"{np.mean(algo_ranks[algo]):.2f}" for algo in all_algos]
        writer.writerow(['Average Rank'] + avg_ranks)
        
        # 最佳次数
        best_counts = [sum(1 for r in algo_ranks[algo] if r == 1) for algo in all_algos]
        writer.writerow(['Best Count'] + best_counts)
    
    print(f"排名汇总表已保存到: {ranking_path}")


def plot_boxplots(
    all_final_bests: Dict[str, Dict[str, List[float]]],
    output_dir: str,
) -> None:
    """为每个基准函数绘制箱线图"""
    os.makedirs(output_dir, exist_ok=True)
    
    for bench_name, bench_data in all_final_bests.items():
        algos = list(bench_data.keys())
        data = [bench_data[algo] for algo in algos]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, tick_labels=algos, patch_artist=True)
        
        # 设置颜色
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors[:len(algos)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Final Best Fitness')
        ax.set_title(f'Algorithm Comparison on {bench_name}')
        ax.grid(True, alpha=0.3)
        
        # 添加均值点
        means = [np.mean(d) for d in data]
        ax.scatter(range(1, len(algos) + 1), means, marker='D', color='red', 
                   s=50, zorder=3, label='Mean')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{bench_name}_boxplot.png"), dpi=200)
        plt.close()
    
    print(f"箱线图已保存到: {output_dir}")


def plot_convergence_comparison(
    all_curves: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
) -> None:
    """绘制所有基准函数的收敛曲线对比图（子图形式）"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_benchmarks = len(all_curves)
    if n_benchmarks == 0:
        return
    
    # 计算子图布局
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_benchmarks == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD', '#F0E68C']
    
    for idx, (bench_name, bench_curves) in enumerate(all_curves.items()):
        ax = axes[idx]
        for i, (algo_name, curve) in enumerate(bench_curves.items()):
            ax.plot(curve, label=algo_name, color=colors[i % len(colors)], linewidth=1.5)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Best Fitness')
        ax.set_title(bench_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 对数坐标更好展示收敛
    
    # 隐藏多余的子图
    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_convergence.png"), dpi=200)
    plt.close()
    
    print(f"收敛曲线对比图已保存到: {output_dir}/all_convergence.png")


def generate_paper_outputs(
    summary: Dict[str, Dict[str, Dict]],
    all_final_bests: Dict[str, Dict[str, List[float]]],
    all_curves: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
) -> None:
    """生成论文所需的所有输出"""
    print("\n" + "=" * 60)
    print(" 生成论文输出...")
    print("=" * 60)
    
    # 1. 详细统计结果表（含Wilcoxon检验）
    save_statistical_results(all_final_bests, summary, output_dir)
    
    # 2. 箱线图
    plot_boxplots(all_final_bests, output_dir)
    
    # 3. 收敛曲线对比图
    plot_convergence_comparison(all_curves, output_dir)
    
    print("=" * 60)
    print(" 论文输出生成完成！")
    print("=" * 60 + "\n")


def run_experiments(cfg: ExperimentConfig) -> Dict:
    """运行完整实验
    
    Args:
        cfg: 实验配置
    
    Returns:
        汇总结果字典
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    
    summary = {}
    all_final_bests = {}  # {benchmark: {algorithm: [final_best_values]}}
    all_curves = {}       # {benchmark: {algorithm: mean_curve}}
    
    for bench_name in cfg.benchmarks:
        bench_results = {}
        all_final_bests[bench_name] = {}
        all_curves[bench_name] = {}
        print(f"\n=== Benchmark: {bench_name} ===")
        
        for algo_name in cfg.algorithms:
            print(f"Running {algo_name} ...")
            save_path = os.path.join(cfg.results_dir, f"{bench_name}_{algo_name}")
            
            res = run_one_experiment(
                bench_name,
                algo_name,
                cfg,
                cfg.seeds,
                progress_prefix=f"{bench_name}/{algo_name}",
                save_path=save_path,
            )
            bench_results[algo_name] = res
            
            # 收集论文输出数据
            all_final_bests[bench_name][algo_name] = res.get('final_bests', [res['final_mean']])
            all_curves[bench_name][algo_name] = res['best_mean']
            
            print(f"{algo_name}: final best = {res['final_mean']:.4e} ± {res['final_std']:.4e}")
        
        summary[bench_name] = bench_results
        plot_convergence(bench_name, bench_results, cfg.output_dir)
        
        for algo_name, res in bench_results.items():
            if res["rl_counts"] is not None:
                plot_rl_heatmap(bench_name, algo_name, res["rl_counts"], cfg.output_dir)
    
    # 保存总体汇总表
    save_overall_summary(summary, cfg.results_dir)
    print(f"\n汇总表已保存到: {os.path.join(cfg.results_dir, 'overall_summary.csv')}")
    
    # 生成论文输出（统计检验、箱线图、收敛曲线对比）
    generate_paper_outputs(summary, all_final_bests, all_curves, cfg.results_dir)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="RL-GA 实验运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python experiments.py --mode full       # 完整实验
  python experiments.py --mode quick      # 快速测试
  python experiments.py --mode ablation   # 消融实验
  python experiments.py --mode debug      # 调试模式
  python experiments.py --list-modes      # 列出所有模式
  python experiments.py --show-config quick  # 显示配置详情
        """
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default=None,
        help="实验模式 (full/quick/ablation/debug/sensitivity/scalability)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="快速测试（等同于 --mode quick）"
    )
    parser.add_argument(
        "--list-modes", "-l",
        action="store_true",
        help="列出所有可用实验模式"
    )
    parser.add_argument(
        "--show-config", "-s",
        type=str,
        metavar="MODE",
        help="显示指定模式的配置详情"
    )
    parser.add_argument(
        "--validate", "-v",
        type=str,
        metavar="MODE",
        help="验证指定模式的配置"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）"
    )
    
    args = parser.parse_args()

    # 列出模式
    if args.list_modes:
        print_modes(args.config)
        return
    
    # 显示配置
    if args.show_config:
        cfg = load_config(args.show_config, config_path=args.config)
        print_config(cfg)
        print(f"预计运行时间: {estimate_runtime(cfg)}")
        return
    
    # 验证配置
    if args.validate:
        cfg = load_config(args.validate, config_path=args.config)
        check_config(cfg)
        return
    
    # 确定模式
    if args.quick:
        mode = "quick"
    elif args.mode:
        mode = args.mode
    else:
        # 从配置文件读取默认模式
        try:
            loader = ConfigLoader(args.config)
            mode = loader.get_default_mode()
        except FileNotFoundError:
            mode = "full"
    
    # 加载配置并运行
    try:
        cfg = load_config(mode, config_path=args.config)
    except ValueError as e:
        print(f"错误: {e}")
        print("\n使用 --list-modes 查看可用模式")
        return
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
        return
    
    # 显示配置摘要
    print_config(cfg)
    print(f"预计运行时间: {estimate_runtime(cfg)}\n")
    
    # 验证配置
    if not check_config(cfg):
        return
    
    # 运行实验
    print("\n开始实验...")
    run_experiments(cfg)


if __name__ == "__main__":
    main()
