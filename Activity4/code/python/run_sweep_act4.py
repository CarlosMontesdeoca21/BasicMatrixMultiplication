import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACT4_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
EXE = os.path.join(ACT4_DIR, "code", "c", "benchmark_mpi.exe")
OUTPUT = os.path.join(ACT4_DIR, "data", "results_act4.csv")

MPIEXEC = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
MSMPI_BIN = r"C:\Program Files\Microsoft MPI\Bin"

SIZES = [100, 500, 1000, 1500]
OMP_THREADS = [1, 2, 4, 8, 16, 22]
MPI_WORKERS = [1, 2, 4, 8, 16, 22]


def run(label, args):
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PATH"] = MSMPI_BIN + os.pathsep + env.get("PATH", "")
    proc = subprocess.run([MPIEXEC] + args, env=env, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[FAIL] {label} exited with code {proc.returncode}")
        sys.exit(1)
    print(f"[ok] {label} ({dt:.1f} s)")
    return dt


def main():
    if not os.path.exists(EXE):
        print(f"Binary not found: {EXE}")
        return 1
    if os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    run("selftest", ["-n", "4", EXE, "--selftest"])
    run("validate n=10", ["-n", "1", EXE, "--validate", "10"])
    proc = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "validate_act4.py")])
    if proc.returncode != 0:
        print("[FAIL] validate_act4.py failed")
        return 1

    total = 0.0
    for n in SIZES:
        total += run(f"seq n={n}", ["-n", "1", EXE, "--method", "seq", "--sizes", str(n)])
        for t in OMP_THREADS:
            total += run(f"omp n={n} t={t}",
                         ["-n", "1", EXE, "--method", "omp", "--threads", str(t), "--sizes", str(n)])
        for p in MPI_WORKERS:
            total += run(f"mpi n={n} p={p}",
                         ["-n", str(p), EXE, "--method", "mpi", "--sizes", str(n)])

    print(f"\nSweep finished in {total:.1f} s. Raw data: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
