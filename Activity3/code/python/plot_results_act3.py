import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACT3_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(ACT3_DIR, "resultsAct3")

SIZE_COLORS = {
    10: "#999999", 100: "#1f77b4", 250: "#ff7f0e",
    500: "#2ca02c", 1000: "#d62728",
}
PLOT_SIZES = [250, 500, 1000]
BUILD_ORDER = ["O3_no_vec", "O3", "O3_native"]
BUILD_LABELS = {"O3_no_vec": "-O3\n-fno-tree-vectorize",
                "O3": "-O3",
                "O3_native": "-O3\n-march=native"}

def plot_time_vs_threads(summary):
    df = summary[(summary["build"] == "O3") & (summary["method"] == "OpenMP")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for size in sorted(df["size"].unique()):
        d = df[df["size"] == size].sort_values("threads")
        ax.errorbar(d["threads"], d["time_median"], yerr=d["time_iqr"],
                    marker="o", color=SIZE_COLORS.get(size), label=f"n = {size}",
                    capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of OpenMP threads (p)")
    ax.set_ylabel("Median time (s) with IQR error bars")
    ax.set_title("OpenMP matrix multiplication: time vs thread count")
    ax.set_xticks([1, 2, 4, 8, 16, 22, 28])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axvline(22, color="gray", linestyle=":", linewidth=1, label="HW contexts (22)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_time_vs_threads_act3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

def plot_speedup(sp):
    df = sp[(sp["size"].isin(PLOT_SIZES))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ideal = sorted(df["threads"].unique())
    ax.plot(ideal, ideal, "k--", label="Ideal S(p) = p", linewidth=1)
    for size in PLOT_SIZES:
        d = df[df["size"] == size].sort_values("threads")
        ax.plot(d["threads"], d["speedup"], marker="o",
                color=SIZE_COLORS[size], label=f"n = {size}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of threads (p)")
    ax.set_ylabel("Speedup S(p) = T_seq / T_p")
    ax.set_title("Speedup vs number of threads (median times)")
    ax.set_xticks([1, 2, 4, 8, 16, 22, 28])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axvline(22, color="gray", linestyle=":", linewidth=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_speedup_vs_threads_act3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

def plot_efficiency(sp):
    df = sp[(sp["size"].isin(PLOT_SIZES))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal E(p) = 1")
    for size in PLOT_SIZES:
        d = df[df["size"] == size].sort_values("threads")
        ax.plot(d["threads"], d["efficiency"], marker="o",
                color=SIZE_COLORS[size], label=f"n = {size}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of threads (p)")
    ax.set_ylabel("Efficiency E(p) = S(p) / p")
    ax.set_title("Parallel efficiency vs number of threads")
    ax.set_xticks([1, 2, 4, 8, 16, 22, 28])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.axvline(22, color="gray", linestyle=":", linewidth=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_efficiency_vs_threads_act3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

def plot_seq_vs_omp(summary):
    methods = [("Sequential", None, "Sequential"), ("OpenMP", 1, "OpenMP, p=1"),
               ("OpenMP", 16, "OpenMP, p=16")]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    xs = range(len(PLOT_SIZES))
    for idx, (method, threads, label) in enumerate(methods):
        vals = []
        for size in PLOT_SIZES:
            d = summary[(summary["size"] == size) & (summary["method"] == method) &
                        (summary["build"] == "O3")]
            if threads is not None:
                d = d[d["threads"] == threads]
            vals.append(float(d["time_median"].iloc[0]))
        pos = [x + (idx - 1) * width for x in xs]
        ax.bar(pos, vals, width=width, label=label)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"n = {s}" for s in PLOT_SIZES])
    ax.set_ylabel("Median time (s)")
    ax.set_title("Sequential vs OpenMP (p=1 overhead, p=16)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_seq_vs_omp_act3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

def plot_simd(simd):
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.2
    xs = range(len(PLOT_SIZES))
    for idx, build in enumerate(BUILD_ORDER):
        vals = []
        for size in PLOT_SIZES:
            d = simd[(simd["size"] == size) & (simd["build"] == build)]
            vals.append(float(d["time_median"].iloc[0]) if len(d) else float("nan"))
        pos = [x + (idx - 1.5) * width for x in xs]
        ax.bar(pos, vals, width=width, label=f"ikj, {BUILD_LABELS[build]}")
    vals_ref = []
    for size in PLOT_SIZES:
        d = simd[(simd["size"] == size) & (simd["build"] == "O3_ijk_ref")]
        vals_ref.append(float(d["time_median"].iloc[0]) if len(d) else float("nan"))
    pos = [x + 1.5 * width for x in xs]
    ax.bar(pos, vals_ref, width=width, label="i-j-k, -O3 (baseline)", color="#d62728")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"n = {s}" for s in PLOT_SIZES])
    ax.set_ylabel("Median time (s)")
    ax.set_title("SIMD study: sequential i-k-j under different compiler flags")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "plot_simd_act3.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

def main():
    summary = pd.read_csv(os.path.join(RESULTS_DIR, "summary_time_act3.csv"))
    sp = pd.read_csv(os.path.join(RESULTS_DIR, "speedup_efficiency_act3.csv"))
    simd = pd.read_csv(os.path.join(RESULTS_DIR, "simd_summary_act3.csv"))

    plot_time_vs_threads(summary)
    plot_speedup(sp)
    plot_efficiency(sp)
    plot_seq_vs_omp(summary)
    plot_simd(simd)

if __name__ == "__main__":
    main()
