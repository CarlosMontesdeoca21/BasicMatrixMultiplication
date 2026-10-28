# Activity 3 — Parallel Matrix Multiplication (C + OpenMP)

Research study on how the number of threads affects the performance of a dense
`n x n` matrix multiplication in shared memory, plus an optional SIMD
vectorization study of a sequential variant.

**Research questions**

1. How does the thread count `p` affect the execution time of an OpenMP
   parallelization of the classic i-j-k kernel, compared with its sequential
   baseline?
2. Where does the speedup saturate, and what efficiency is achieved at each
   thread count (including oversubscription beyond the 22 hardware contexts)?
3. (Optional SIMD extension) What is the effect of compiler auto-vectorization
   and `-march=native` on a sequential i-k-j kernel that exposes a
   vectorizable rank-1 update?

## Environment

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 155H — 16 physical cores (6 P + 8 E + 2 LPE), 22 hardware threads |
| RAM | 32 GB |
| OS | Windows 11 |
| Compiler | gcc 16.2.0 (MSYS2) |
| Python | 3.12.10 (pandas, matplotlib, numpy) |
| Time budgets | 5 min / 4 GB per multiplication (declared, not reached) |

Thread counts tested: `p ∈ {1, 2, 4, 8, 16, 22, 28}`. `p = 22` matches the
number of hardware contexts; `p = 28` deliberately oversubscribes the machine
and is labelled as such in the results.

**Affinity note (limitation):** `OMP_PROC_BIND` / `OMP_PLACES` are ignored by
libgomp on Windows (`libgomp: Affinity not supported on this configuration`),
so no explicit thread affinity could be enforced; the OS free-threaded
scheduler placed the workers.

## Directory structure

```
Activity3/
├── .gitignore                       # data/ (generated), *.exe, *.o
├── README.md
├── code/
│   ├── c/benchmark_omp.c            # Sequential i-j-k, OpenMP, i-k-j (SIMD study)
│   └── python/
│       ├── generate_inputsAct3.py   # input matrices (seed 42, sizes 10..1000)
│       ├── validate_act3.py         # correctness checks (NumPy reference)
│       ├── merge_results_act3.py    # median/IQR, speedup, efficiency, SIMD table
│       └── plot_results_act3.py     # figures
├── data/                            # generated inputs + raw measurements (not tracked)
└── resultsAct3/                     # consolidated CSVs, figures, evidence (tracked)
```

## Implementations

All kernels use `double` values, row-major `double**` storage (array of row
pointers) and identical input matrices per configuration. The kernel and
storage design are the same validated ones used in Activity 1 (C baseline);
this activity only adds threading on top.

| Method | Description |
|---|---|
| `Sequential` | Classic i-j-k triple loop, O(n^3) (baseline, identical to Activity 1) |
| `OpenMP` | Same i-j-k kernel with `#pragma omp parallel for` over rows (outer loop), `num_threads(p)`, `schedule(static)`. Each row of C is written by exactly one thread, so there are no data races and no synchronization is needed |
| `Sequential_ikj` | i-k-j loop order (SIMD study only): same arithmetic work and storage, but the innermost loop is a vectorizable rank-1 row update |

**Measurement protocol**

- Timer: `QueryPerformanceCounter` (monotonic, ~100 ns resolution).
  `omp_get_wtime()` was rejected because libgomp on Windows only resolves
  ~1 ms, which quantizes the smaller sizes.
- 2 warm-up runs + 5 measured repetitions per configuration; the reported
  statistic is the **median**, with the interquartile range (IQR) as
  variability.
- Only the multiplication kernel is timed; input loading and allocation are
  outside the timed region.
- Memory: process working set (`GetProcessMemoryInfo`, KB). It includes the
  input matrices and the runtime, so it is used only as a coarse indicator.
- Correctness: after every OpenMP configuration the checksum of C is compared
  against the sequential reference (relative tolerance 1e-6). Additionally,
  `validate_act3.py` checks the sequential result for n = 10 against NumPy's
  `A @ B` (atol = 1e-6). All validations pass; sequential vs OpenMP elementwise
  difference is exactly 0 (same operation order per element).

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 + PowerShell | commands below assume PowerShell at the repository root |
| gcc with OpenMP | tested with gcc 16.2.0 (MSYS2). Any gcc with `-fopenmp` works; on Windows it links `libgomp` and needs `-lpsapi` for the memory metric |
| Python 3.10+ | if `python` opens the Microsoft Store, use `py` instead |
| Python packages | `pip install numpy pandas matplotlib` |

All commands are run **from the repository root** (the folder that contains
`Activity1/`, `Activity2/` and `Activity3/`):

```powershell
cd "...\BasicMatrixMultiplication"
```

The Python scripts and the benchmark binary resolve `Activity3/data` by
themselves (relative to the script/exe location), so they also work from any
other directory. Only the `gcc` commands and the `--output` paths below are
relative to the repository root. If your terminal is inside `Activity3/`,
remove the leading `Activity3/` from those paths.

## How to reproduce

1. Generate the input matrices (deterministic, seed 42; the matrices for
   n = 10, 100, 250 are identical by construction to those of Activity 1):

```powershell
python Activity3/code/python/generate_inputsAct3.py
```

2. Build the main binary (`-O3`):

```powershell
gcc -O3 -fopenmp -Wall -DBUILD_LABEL=O3 Activity3/code/c/benchmark_omp.c -o Activity3/code/c/benchmark_omp.exe -lpsapi -lm
```

3. Run the main sweep (sequential + OpenMP thread sweep + i-k-j at -O3).
   Takes roughly 2–4 minutes on the reference machine. Results are written to
   `Activity3/data/results_omp_act3.csv`:

```powershell
.\Activity3\code\c\benchmark_omp.exe --method all
```

4. Correctness checks (the first command dumps the sequential result of
   n = 10 to `Activity3/data/result_C_10.csv`, the second one compares it
   against NumPy):

```powershell
.\Activity3\code\c\benchmark_omp.exe --validate 10
python Activity3/code/python/validate_act3.py
```

5. SIMD study — build and run the two additional variants (single-threaded,
   sizes 250, 500, 1000):

```powershell
gcc -O3 -march=native -fopenmp -Wall -DBUILD_LABEL=O3_native Activity3/code/c/benchmark_omp.c -o Activity3/code/c/benchmark_omp_native.exe -lpsapi -lm
gcc -O3 -fno-tree-vectorize -fopenmp -Wall -DBUILD_LABEL=O3_no_vec Activity3/code/c/benchmark_omp.c -o Activity3/code/c/benchmark_omp_no_vec.exe -lpsapi -lm
.\Activity3\code\c\benchmark_omp_native.exe --method ikj --sizes 250,500,1000 --output Activity3\data\results_simd_native.csv
.\Activity3\code\c\benchmark_omp_no_vec.exe --method ikj --sizes 250,500,1000 --output Activity3\data\results_simd_no_vec.csv
```

6. (Optional) Regenerate the vectorization evidence with gcc's
   `-fopt-info-vec-optimized` (a curated copy is already included in the
   repository):

```powershell
cmd /c "gcc -O3 -fopenmp -DBUILD_LABEL=O3 -fopt-info-vec-optimized -c Activity3/code/c/benchmark_omp.c -o Activity3/code/c/benchmark_omp.o 2> Activity3/resultsAct3/vectorization_evidence.txt"
```

   The i-k-j inner loop is vectorized with 16-byte vectors, while the i-j-k
   inner loop (FP reduction over k) is not vectorizable without `-ffast-math`.

7. Consolidate and plot (writes the summary CSVs and the five figures of
   `resultsAct3/`):

```powershell
python Activity3/code/python/merge_results_act3.py
python Activity3/code/python/plot_results_act3.py
```

## Outputs

Raw measurements (generated locally in `data/`, not tracked):

- `results_omp_act3.csv`, `results_simd_native.csv`, `results_simd_no_vec.csv`
  — columns: `size,method,threads,build,repetition,time_seconds,memory`

Consolidated results (tracked, `resultsAct3/`):

- `summary_time_act3.csv` — median, Q1, Q3, IQR, min, max, median memory per
  (size, method, threads, build)
- `speedup_efficiency_act3.csv` — `S(p) = T_seq(n) / T_p(n)` and
  `E(p) = S(p) / p` from median times
- `simd_summary_act3.csv` — median times of the i-k-j kernel per compiler
  build (plus the i-j-k `-O3` reference)
- `plot_time_vs_threads_act3.png`, `plot_speedup_vs_threads_act3.png`,
  `plot_efficiency_vs_threads_act3.png`, `plot_seq_vs_omp_act3.png`,
  `plot_simd_act3.png`
- `vectorization_evidence.txt`

## Reused materials from previous activities

- The baseline kernel, storage layout, CSV loading, checksum validation and
  memory metric are the ones validated in **Activity 1** (`code/c/benchmark.c`);
  they are copied (not imported) into `benchmark_omp.c` so Activity 3 is
  self-contained.
- `generate_inputsAct3.py` follows the same generator as Activity 1
  (`random.seed(42)`, `random.random()` in [0, 1)), extended with sizes
  500 and 1000; Activity 3 does not read any file from Activity 1/2.
