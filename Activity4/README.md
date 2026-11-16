# Activity 4 — Distributed Matrix Multiplication (C + MPI)

Research study on how the number of MPI worker processes affects the
performance of a dense `n x n` matrix multiplication distributed across
processes, including the communication/computation breakdown and a comparison
against the validated sequential baseline and the Activity 3 OpenMP version.

**Research questions**

1. How does the worker count `p` affect the end-to-end execution time of an
   MPI distribution of the classic i-j-k kernel, compared with its sequential
   baseline?
2. Where does the strong-scaling speedup saturate, and what efficiency is
   achieved at each worker count?
3. How is the end-to-end time split between transfer (scatter + broadcast),
   local computation and gather as `p` grows?
4. How does the distributed version compare with the shared-memory OpenMP
   version at the same worker/thread counts on a single physical machine?
5. How do workers, processes and threads relate in this setup, and what
   memory does each worker consume?

## Environment

| Item | Value |
|---|---|
| CPU | Intel Core Ultra 7 155H — 16 physical cores (6 P + 8 E + 2 LPE), 22 hardware threads |
| RAM | 32 GB |
| OS | Windows 11 |
| MPI runtime | Microsoft MPI 10.1.12498.52 (`mpiexec`, winget id `Microsoft.msmpi`) |
| MPI SDK / compiler | `mingw-w64-ucrt-x86_64-msmpi` 10.1.1 (MSYS2 ucrt64, `mpicc` wrapping gcc 16.2.0) |
| Python | 3.12.10 (pandas, matplotlib, numpy) |
| Power plan | **High Performance** during all official measurements (see note below) |
| Time/memory budgets | 5 min / 4 GB per multiplication (declared; largest observed: 7.49 s, 129 MB) |

Worker/thread counts tested: `p ∈ {1, 2, 4, 8, 16, 22}`. `p = 22` matches the
number of hardware contexts. Matrix sizes: `n ∈ {100, 500, 1000, 1500}`
(generated inputs also include `n = 10` for self-tests and validation).

**Local multi-worker execution (permission):** all workers run on the single
physical machine of the author. This multi-worker local configuration was
explicitly authorized by the professor for this assignment; it is documented
as such in the report. No remote host and no physical network are involved.

**Power-plan note:** under the default *Balanced* plan, sustained
single-thread throughput dropped by about 1.8x after the initial
turbo-boost window (~2 s), which made consecutive repetitions incomparable
(first repetition fast, later ones throttled). All official measurements were
therefore taken with the Windows power plan set to **High Performance**
(`powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`), and 2 warm-up
runs absorb the remaining boost window. The plan was left in place during the
campaign; restore *Balanced* afterwards with
`powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e` if desired.

**Affinity note:** Microsoft MPI applies its own CPU-set affinity policy on
Windows 11 (unlike libgomp in Activity 3, which ignored
`OMP_PROC_BIND`/`OMP_PLACES`); the OpenMP reference runs in this activity
again leave thread placement to the OS scheduler.

## Directory structure

```
Activity4/
├── .gitignore                       # data/ (generated), *.exe, *.o
├── README.md
├── code/
│   ├── c/benchmark_mpi.c            # Sequential, OpenMP and MPI kernels + phase timing
│   └── python/
│       ├── generate_inputsAct4.py   # input matrices (seed 42, sizes 10..1500)
│       ├── run_sweep_act4.py        # orchestrates mpiexec for the whole campaign
│       ├── validate_act4.py         # correctness checks (NumPy reference)
│       ├── merge_results_act4.py    # median/IQR, strong scaling, phase medians
│       └── plot_results_act4.py     # figures
├── report/                          # LaTeX report (Overleaf)
│   └── Report_assigment4.tex
├── data/                            # generated inputs + raw measurements (not tracked)
└── resultsAct4/                     # consolidated CSVs, figures (tracked)
```

## Implementations

All kernels use `double` values and the same arithmetic (classic i-j-k triple
loop, O(n^3)) validated in Activity 1. The sequential and OpenMP variants keep
the original row-pointer (`double**`) storage; the MPI variant packs rows into
contiguous buffers, which is required for the data-distribution calls.

| Method | Description |
|---|---|
| `Sequential` | Classic i-j-k triple loop over row-pointer storage (baseline, identical to Activity 1/3) |
| `OpenMP` | Same i-j-k kernel with `#pragma omp parallel for` over rows, `num_threads(p)`, `schedule(static)` (reference from Activity 3, re-measured in this campaign) |
| `MPI` | 1-D row-block distribution: rank 0 loads A and B, `MPI_Scatterv` distributes contiguous row blocks of A, `MPI_Bcast` sends the full B to every rank, each rank multiplies its block into a contiguous C buffer, `MPI_Gatherv` returns the rows of C to rank 0. Exactly `p` MPI processes (one address space each) |

**Partitioning / data flow (per multiplication):**

1. **Setup (rank 0, outside the timed region):** load A and B from CSV,
   allocate row blocks.
2. **Transfer phase (timed):** `MPI_Scatterv` of A's row blocks
   (logical bytes: `n^2 * 8 * (p-1)/p`) + `MPI_Bcast` of the full B
   (`n^2 * 8 * (p-1)`).
3. **Compute phase (timed):** local i-j-k multiplication of the block.
4. **Gather phase (timed):** `MPI_Gatherv` of C's row blocks
   (`n^2 * 8 * (p-1)/p`).
5. End-to-end time = transfer + compute + gather (barrier before each
   repetition so the phases are comparable; phases are summed, never
   overlapped in the reported e2e figure).

**Measurement protocol**

- Timer: `QueryPerformanceCounter` (monotonic, ~100 ns resolution) around each
  phase; end-to-end is the sum of the three phases.
- 2 warm-up runs + 5 measured repetitions per configuration; the reported
  statistic is the **median**, with the interquartile range (IQR) as
  variability.
- Matrix loading, allocation and the single-process reference computation are
  outside the timed region.
- Memory: per-process working set (`GetProcessMemoryInfo`, KB), gathered with
  `MPI_Gather` so the reported value is the **maximum over all workers**.
- Correctness: a self-test (I x A, A x I, 0 x A at n = 8, 10) runs at the
  start of every campaign; after every configuration the checksum of C is
  compared against the sequential reference computed in the same process
  (relative tolerance 1e-6, max elementwise diff observed: exactly 0).
  `validate_act4.py` additionally checks the n = 10 result against NumPy's
  `A @ B` (atol = 1e-6, max abs diff 4.9e-7). All validations pass.

## Prerequisites

| Requirement | Notes |
|---|---|
| Windows 10/11 + PowerShell | commands below assume PowerShell at the repository root |
| MSYS2 (ucrt64) + MS-MPI SDK | `pacman -S mingw-w64-ucrt-x86_64-msmpi` provides `C:\msys64\ucrt64\bin\mpicc.exe` and `mpi.h`/`libmsmpi` |
| Microsoft MPI runtime | `winget install -e --id Microsoft.msmpi` installs `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe` (v10.1.12498.52) |
| gcc with OpenMP | bundled with MSYS2 ucrt64 (16.2.0); `mpicc -fopenmp` links both runtimes |
| Python 3.10+ | if `python` opens the Microsoft Store, use `py` instead |
| Python packages | `pip install numpy pandas matplotlib` |

All commands are run **from the repository root** (the folder that contains
`Activity1/`, `Activity2/`, `Activity3/` and `Activity4/`):

```powershell
cd "...\BasicMatrixMultiplication"
```

The Python scripts and the benchmark binary resolve `Activity4/data` by
themselves (relative to the script/exe location). The sweep script invokes
`mpiexec` by its absolute path, so no PATH changes are required.

## How to reproduce

1. Generate the input matrices (deterministic, seed 42; the matrices for
   n = 10, 100 and 1000 are byte-identical to those of Activity 3):

```powershell
python Activity4/code/python/generate_inputsAct4.py
```

2. Build the main binary (`-O3`, OpenMP + MPI in one executable):

```powershell
C:\msys64\ucrt64\bin\mpicc.exe -O3 -fopenmp -Wall -DBUILD_LABEL=O3 Activity4/code/c/benchmark_mpi.c -o Activity4/code/c/benchmark_mpi.exe -lpsapi -lm
```

3. Run the full campaign (self-test, validation, then the sweep of
   sequential + OpenMP + MPI over all sizes and worker counts). Takes roughly
   7 minutes on the reference machine with the High Performance power plan.
   The script deletes any previous `results_act4.csv` first:

```powershell
python Activity4/code/python/run_sweep_act4.py
```

   Individual runs are also possible, e.g.:

```powershell
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 Activity4\code\c\benchmark_mpi.exe --method mpi --sizes 1000
```

4. Correctness checks (also executed automatically by the sweep script):

```powershell
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 1 Activity4\code\c\benchmark_mpi.exe --validate 10
python Activity4/code/python/validate_act4.py
```

5. Consolidate and plot (writes the summary CSVs and the five figures of
   `resultsAct4/`):

```powershell
python Activity4/code/python/merge_results_act4.py
python Activity4/code/python/plot_results_act4.py
```

## Outputs

Raw measurements (generated locally in `data/`, not tracked):

- `results_act4.csv` — columns:
  `size,method,workers,threads,build,repetition,time_e2e,time_transfer,time_compute,time_gather,memory_kb,max_diff`

Consolidated results (tracked, `resultsAct4/`):

- `summary_time_act4.csv` — median, Q1, Q3, IQR, min, max and phase medians
  per (size, method, workers, threads)
- `strong_scaling_act4.csv` — per MPI configuration: `S(p)` vs the
  sequential baseline, `S(p)` vs the MPI single-process time (isolates the
  data-layout effect), efficiency, phase medians and median memory
- `omp_speedup_act4.csv` — OpenMP reference `S(p)` and `E(p)` from the same
  campaign
- `plot_time_vs_workers_act4.png`, `plot_speedup_vs_workers_act4.png`,
  `plot_efficiency_act4.png`, `plot_phase_breakdown_act4.png`,
  `plot_method_comparison_act4.png`

## Experimental decisions

- **Sizes `n ∈ {100, 500, 1000, 1500}`.** `n = 2000` was generated and
  measured during development but dropped from the official campaign: with
  B = 32 MB exceeding the 24 MB shared L3, one multiplication takes 70-102 s
  (cache-residency cliff), which multiplies the full sweep to hours without
  adding a new regime that `n = 1500` (working set already beyond L3) does
  not cover. The declared per-multiplication budget (5 min) was therefore
  respected with a wide margin (largest official time: 7.49 s).
- **Strong scaling only.** Weak scaling is not meaningful for cubic work
  with a fixed total problem size (the assignment allows skipping it with
  justification): enlarging `n` per worker grows the work as `n^3` and the
  communication as `n^2`, so per-worker times are never comparable.
- **Same-campaign OpenMP reference.** The Activity 3 OpenMP measurements were
  taken two weeks earlier under a different power-plan state, so the OpenMP
  reference is re-measured inside this campaign (same inputs, same day, same
  environment) for a fair comparison; Activity 3 results are only cited
  qualitatively.
- **Network limitation.** All transfers use Microsoft MPI's intra-host
  transport over shared memory; no physical network is involved. Only
  *logical* bytes moved by the collectives are reported (formulas above),
  because the physical cost of the shared-memory transport cannot be
  isolated on this machine.

## Reused materials from previous activities

- The baseline kernel, storage layout, CSV loading, checksum validation and
  memory metric are the ones validated in **Activity 1** (`code/c/benchmark.c`)
  and used in **Activity 3**; they are copied (not imported) into
  `benchmark_mpi.c` so Activity 4 is self-contained. The OpenMP kernel is the
  Activity 3 kernel, re-compiled into this binary for the same-campaign
  reference.
- `generate_inputsAct4.py` follows the same generator as Activity 1/3
  (`random.seed(42)`, `random.random()` in [0, 1)); the matrices for
  n = 10, 100 and 1000 are byte-identical (MD5-verified) to the Activity 3
  inputs. Activity 4 does not read any file from Activity 1/2/3.
