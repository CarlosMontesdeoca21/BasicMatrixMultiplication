import os
import glob
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACT3_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(ACT3_DIR, "data")
RESULTS_DIR = os.path.join(ACT3_DIR, "resultsAct3")

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_raw():
    files = [os.path.join(DATA_DIR, "results_omp_act3.csv")]
    files += sorted(glob.glob(os.path.join(DATA_DIR, "results_simd_*.csv")))
    frames = []
    for f in files:
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
            print(f"Loaded {os.path.basename(f)}")
        else:
            print(f"[warn] Missing {f}, skipped")
    if not frames:
        raise SystemExit("No raw result files found in Activity3/data")
    return pd.concat(frames, ignore_index=True)

def summarize(df):
    summary = (
        df.groupby(["size", "method", "threads", "build"])
          .agg(
              time_median=("time_seconds", "median"),
              time_q1=("time_seconds", lambda x: x.quantile(0.25)),
              time_q3=("time_seconds", lambda x: x.quantile(0.75)),
              time_min=("time_seconds", "min"),
              time_max=("time_seconds", "max"),
              memory_median_kb=("memory", "median"),
          )
          .reset_index()
    )
    summary["time_iqr"] = summary["time_q3"] - summary["time_q1"]
    return summary

def speedup_table(df):
    main = df[(df["build"] == "O3") & (df["method"].isin(["Sequential", "OpenMP"]))]
    rows = []
    for size in sorted(main["size"].unique()):
        t_seq = main[(main["size"] == size) & (main["method"] == "Sequential")]["time_seconds"].median()
        rows.append({
            "size": size, "threads": 1, "t_seq_seconds": t_seq,
            "t_parallel_seconds": t_seq, "speedup": 1.0, "efficiency": 1.0, "note": "baseline",
        })
        omp = main[(main["size"] == size) & (main["method"] == "OpenMP")]
        for t in sorted(omp["threads"].unique()):
            t_p = omp[omp["threads"] == t]["time_seconds"].median()
            sp = t_seq / t_p if t_p > 0 else float("nan")
            note = "one thread (framework overhead)" if t == 1 else (
                   "oversubscription" if t > 22 else "")
            rows.append({
                "size": size, "threads": t, "t_seq_seconds": t_seq,
                "t_parallel_seconds": t_p, "speedup": sp,
                "efficiency": sp / t, "note": note,
            })
    return pd.DataFrame(rows)

def simd_table(df):
    ikj = df[df["method"] == "Sequential_ikj"][["size", "build", "time_seconds"]]
    ijk = df[(df["method"] == "Sequential") & (df["build"] == "O3")][["size", "build", "time_seconds"]]
    ijk = ijk.assign(build="O3_ijk_ref")
    all_builds = pd.concat([ikj, ijk], ignore_index=True)
    table = (
        all_builds.groupby(["size", "build"])["time_seconds"]
                  .median().reset_index(name="time_median")
    )
    return table

def main():
    df = load_raw()
    print(f"\nTotal raw measurements: {len(df)}")

    summary = summarize(df)
    summary_path = os.path.join(RESULTS_DIR, "summary_time_act3.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    sp = speedup_table(df)
    sp_path = os.path.join(RESULTS_DIR, "speedup_efficiency_act3.csv")
    sp.to_csv(sp_path, index=False)
    print(f"Wrote {sp_path}")

    simd = simd_table(df)
    simd_path = os.path.join(RESULTS_DIR, "simd_summary_act3.csv")
    simd.to_csv(simd_path, index=False)
    print(f"Wrote {simd_path}")

    print("\n=== Speedup / efficiency (build O3) ===")
    print(sp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Sequential_ikj median times by build (SIMD study) ===")
    print(simd.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

if __name__ == "__main__":
    main()
