import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACT4_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(ACT4_DIR, "resultsAct4")

SIZE_COLORS = {100: "#1f77b4", 500: "#9467bd", 1000: "#d62728", 1500: "#2ca02c"}
WORKER_TICKS = [1, 2, 4, 8, 16, 22]


def load():
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary_time_act4.csv"))
    ss = pd.read_csv(os.path.join(RESULTS_DIR, "strong_scaling_act4.csv"))
    osp = pd.read_csv(os.path.join(RESULTS_DIR, "omp_speedup_act4.csv"))
    return summary, ss, osp


def plot_time_vs_workers(summary):
    df = summary[summary["method"] == "MPI"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for size in sorted(df["size"].unique()):
        d = df[df["size"] == size].sort_values("workers")
        ax.errorbar(d["workers"], d["time_e2e_median"], yerr=d["time_e2e_iqr"],
                    marker="o", color=SIZE_COLORS.get(size), label=f"n = {size}",
                    capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of MPI workers (processes)")
    ax.set_ylabel("Median end-to-end time (s) with IQR error bars")
    ax.set_title("Distributed matrix multiplication: time vs workers")
    ax.set_xticks(WORKER_TICKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "plot_time_vs_workers_act4.png"), dpi=150)
    plt.close(fig)
    print("Wrote plot_time_vs_workers_act4.png")


def plot_speedup(ss, osp):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([1, 22], [1, 22], color="gray", linestyle="--", linewidth=1, label="Ideal S(p) = p")
    for size in sorted(ss["size"].unique()):
        if size == 100:
            continue
        d = ss[ss["size"] == size].sort_values("workers")
        ax.plot(d["workers"], d["speedup_e2e"], marker="o",
                color=SIZE_COLORS.get(size), label=f"MPI e2e, n = {size}")
        ax.plot(d["workers"], d["speedup_compute"], marker="x", linestyle=":",
                color=SIZE_COLORS.get(size), alpha=0.7,
                label=f"MPI compute only, n = {size}")
    for size in sorted(osp["size"].unique()):
        if size == 100:
            continue
        d = osp[osp["size"] == size].sort_values("threads")
        ax.plot(d["threads"], d["speedup"], marker="s", linestyle="-.",
                color=SIZE_COLORS.get(size), alpha=0.7,
                label=f"OpenMP, n = {size}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Workers / threads (p)")
    ax.set_ylabel("Speedup S(p) = T_seq / T_p")
    ax.set_title("Strong scaling: MPI vs OpenMP vs ideal")
    ax.set_xticks(WORKER_TICKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "plot_speedup_vs_workers_act4.png"), dpi=150)
    plt.close(fig)
    print("Wrote plot_speedup_vs_workers_act4.png")


def plot_efficiency(ss, osp):
    fig, ax = plt.subplots(figsize=(8, 5))
    for size in sorted(ss["size"].unique()):
        if size == 100:
            continue
        d = ss[ss["size"] == size].sort_values("workers")
        ax.plot(d["workers"], d["efficiency_e2e"], marker="o",
                color=SIZE_COLORS.get(size), label=f"MPI e2e, n = {size}")
    for size in sorted(osp["size"].unique()):
        if size == 100:
            continue
        d = osp[osp["size"] == size].sort_values("threads")
        ax.plot(d["threads"], d["efficiency"], marker="s", linestyle="-.",
                color=SIZE_COLORS.get(size), alpha=0.7, label=f"OpenMP, n = {size}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Workers / threads (p)")
    ax.set_ylabel("Efficiency E(p) = S(p) / p")
    ax.set_title("Parallel efficiency")
    ax.set_xticks(WORKER_TICKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "plot_efficiency_act4.png"), dpi=150)
    plt.close(fig)
    print("Wrote plot_efficiency_act4.png")


def plot_phase_breakdown(ss):
    sizes = [s for s in sorted(ss["size"].unique()) if s >= 1000]
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 4.5), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        d = ss[ss["size"] == size].sort_values("workers")
        x = range(len(d))
        ax.bar(x, d["t_transfer_median"], color="#ff7f0e", label="Transfer (scatter + bcast)")
        ax.bar(x, d["t_compute_median"], bottom=d["t_transfer_median"],
               color="#1f77b4", label="Local compute")
        ax.bar(x, d["t_gather_median"],
               bottom=d["t_transfer_median"] + d["t_compute_median"],
               color="#2ca02c", label="Gather")
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(w) for w in d["workers"]])
        ax.set_xlabel("MPI workers (processes)")
        ax.set_ylabel("Median time (s)")
        ax.set_title(f"Phase breakdown, n = {size}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "plot_phase_breakdown_act4.png"), dpi=150)
    plt.close(fig)
    print("Wrote plot_phase_breakdown_act4.png")


def plot_method_comparison(summary):
    sizes = [s for s in sorted(summary["size"].unique()) if s >= 1000]
    methods = [("Sequential", 1), ("OpenMP", 16), ("MPI", 16)]
    labels = ["Sequential", "OpenMP p=16", "MPI p=16"]
    colors = ["#999999", "#ff7f0e", "#1f77b4"]
    import numpy as np
    x = np.arange(len(sizes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, ((method, conf), label, color) in enumerate(zip(methods, labels, colors)):
        vals = []
        for size in sizes:
            if method == "Sequential":
                d = summary[(summary["size"] == size) & (summary["method"] == method)]
            elif method == "OpenMP":
                d = summary[(summary["size"] == size) & (summary["method"] == method) &
                            (summary["threads"] == conf)]
            else:
                d = summary[(summary["size"] == size) & (summary["method"] == method) &
                            (summary["workers"] == conf)]
            vals.append(float(d["time_e2e_median"].iloc[0]) if len(d) else float("nan"))
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f"n = {s}" for s in sizes])
    ax.set_ylabel("Median end-to-end time (s)")
    ax.set_title("Sequential vs OpenMP vs MPI at the same inputs")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "plot_method_comparison_act4.png"), dpi=150)
    plt.close(fig)
    print("Wrote plot_method_comparison_act4.png")


def main():
    summary, ss, osp = load()
    plot_time_vs_workers(summary)
    plot_speedup(ss, osp)
    plot_efficiency(ss, osp)
    plot_phase_breakdown(ss)
    plot_method_comparison(summary)


if __name__ == "__main__":
    main()
