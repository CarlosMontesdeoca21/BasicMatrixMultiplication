import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACT4_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(ACT4_DIR, "data")
RESULTS_DIR = os.path.join(ACT4_DIR, "resultsAct4")

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_raw():
    path = os.path.join(DATA_DIR, "results_act4.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Raw results not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {os.path.basename(path)}: {len(df)} rows")
    return df


def summarize(df):
    summary = (
        df.groupby(["size", "method", "workers", "threads", "build"])
          .agg(
              time_e2e_median=("time_e2e", "median"),
              time_e2e_q1=("time_e2e", lambda x: x.quantile(0.25)),
              time_e2e_q3=("time_e2e", lambda x: x.quantile(0.75)),
              time_e2e_min=("time_e2e", "min"),
              time_e2e_max=("time_e2e", "max"),
              time_transfer_median=("time_transfer", "median"),
              time_compute_median=("time_compute", "median"),
              time_gather_median=("time_gather", "median"),
              memory_median_kb=("memory_kb", "median"),
              max_diff=("max_diff", "max"),
          )
          .reset_index()
    )
    summary["time_e2e_iqr"] = summary["time_e2e_q3"] - summary["time_e2e_q1"]
    return summary


def strong_scaling(df, summary):
    seq = summary[(summary["method"] == "Sequential")]
    mpi = df[df["method"] == "MPI"]
    rows = []
    for size in sorted(df["size"].unique()):
        t_seq = seq[seq["size"] == size]["time_e2e_median"]
        if t_seq.empty:
            continue
        t_seq = float(t_seq.iloc[0])
        p1 = mpi[(mpi["size"] == size) & (mpi["workers"] == 1)]
        t_mpi_p1 = float(p1["time_e2e"].median()) if len(p1) else float("nan")
        for w in sorted(mpi[mpi["size"] == size]["workers"].unique()):
            d = mpi[(mpi["size"] == size) & (mpi["workers"] == w)]
            e2e = d["time_e2e"].median()
            comp = d["time_compute"].median()
            tr = d["time_transfer"].median()
            ga = d["time_gather"].median()
            mem = d["memory_kb"].median()
            sp_e2e = t_seq / e2e
            rows.append({
                "size": size, "workers": w, "t_seq_seconds": t_seq,
                "t_e2e_median": e2e, "t_compute_median": comp,
                "t_transfer_median": tr, "t_gather_median": ga,
                "speedup_e2e": sp_e2e, "speedup_compute": t_seq / comp,
                "speedup_vs_mpi_p1": t_mpi_p1 / e2e,
                "efficiency_e2e": sp_e2e / w, "memory_median_kb": mem,
            })
    return pd.DataFrame(rows)


def omp_speedup(df, summary):
    seq = summary[(summary["method"] == "Sequential")]
    omp = df[df["method"] == "OpenMP"]
    rows = []
    for size in sorted(df["size"].unique()):
        t_seq = seq[seq["size"] == size]["time_e2e_median"]
        if t_seq.empty:
            continue
        t_seq = float(t_seq.iloc[0])
        for t in sorted(omp[omp["size"] == size]["threads"].unique()):
            d = omp[(omp["size"] == size) & (omp["threads"] == t)]
            e2e = d["time_e2e"].median()
            sp = t_seq / e2e
            rows.append({
                "size": size, "threads": t, "t_seq_seconds": t_seq,
                "t_omp_median": e2e, "speedup": sp, "efficiency": sp / t,
            })
    return pd.DataFrame(rows)


def main():
    df = load_raw()
    print(f"Total raw measurements: {len(df)}")

    summary = summarize(df)
    path = os.path.join(RESULTS_DIR, "summary_time_act4.csv")
    summary.to_csv(path, index=False)
    print(f"Wrote {path}")

    ss = strong_scaling(df, summary)
    path = os.path.join(RESULTS_DIR, "strong_scaling_act4.csv")
    ss.to_csv(path, index=False)
    print(f"Wrote {path}")

    osp = omp_speedup(df, summary)
    path = os.path.join(RESULTS_DIR, "omp_speedup_act4.csv")
    osp.to_csv(path, index=False)
    print(f"Wrote {path}")

    print("\n=== Strong scaling (MPI, vs sequential baseline of the same run) ===")
    print(ss.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== OpenMP reference ===")
    print(osp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fails = summary[summary["max_diff"] > 1e-6]
    if len(fails):
        print("\n[warn] configurations with max_diff > 1e-6:")
        print(fails[["size", "method", "workers", "threads", "max_diff"]].to_string(index=False))
    else:
        print("\nAll configurations validated (max_diff <= 1e-6).")


if __name__ == "__main__":
    main()
