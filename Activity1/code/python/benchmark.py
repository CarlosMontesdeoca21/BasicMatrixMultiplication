import csv
import time
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def get_memory_usage():
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss // 1024
    return 0

def multiply_matrices(A, B, C, n):
    for i in range(n):
        for j in range(n):
            C[i][j] = 0.0
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]

def read_csv(filename, n):
    matrix = [[0.0] * n for _ in range(n)]
    try:
        with open(filename, 'r') as f:
            for row_idx, line in enumerate(f):
                if row_idx >= n:
                    break
                values = line.strip().split(',')
                for col_idx, val in enumerate(values):
                    if col_idx < n:
                        matrix[row_idx][col_idx] = float(val)
        return matrix
    except FileNotFoundError:
        return None

def compute_checksum(C, n):
    return sum(sum(row) for row in C)

def main():
    sizes = [10, 100, 250]
    warm_ups = 2
    repetitions = 5
    csv_path = "Activity1/data/results_python.csv"

    print("=== Python Benchmark (Warm-up + 5 Repetitions) ===")

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["size", "language", "numeric_type", "repetition", "time_seconds", "memory"])

        for n in sizes:
            path_A = f"Activity1/data/matrix_A_{n}.csv"
            path_B = f"Activity1/data/matrix_B_{n}.csv"

            A = read_csv(path_A, n)
            B = read_csv(path_B, n)

            if A is None or B is None:
                print(f"Python (n={n}): Input files not found.")
                continue

            C = [[0.0] * n for _ in range(n)]

            for _ in range(warm_ups):
                multiply_matrices(A, B, C, n)

            for rep in range(1, repetitions + 1):
                start_time = time.perf_counter()
                
                multiply_matrices(A, B, C, n)
                
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                memory_kb = get_memory_usage()

                writer.writerow([n, "Python", "double", rep, f"{elapsed_time:.6f}", memory_kb])

                checksum = compute_checksum(C, n)
                print(f"Python | n={n} | Rep {rep} | Time: {elapsed_time:.6f} s | Memory: {memory_kb} KB | Checksum: {checksum:.4f}")

    print(f"Python results successfully saved to {csv_path}!")

if __name__ == "__main__":
    main()