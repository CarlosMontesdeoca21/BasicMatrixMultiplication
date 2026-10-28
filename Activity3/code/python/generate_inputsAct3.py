import random
import csv
import os

def create_and_save_matrices(n, output_dir):
    random.seed(42)

    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]

    with open(os.path.join(output_dir, f"matrix_A_{n}.csv"), "w", newline="") as f:
        csv.writer(f).writerows(A)
    with open(os.path.join(output_dir, f"matrix_B_{n}.csv"), "w", newline="") as f:
        csv.writer(f).writerows(B)

    print(f"Generated matrices of size {n}x{n}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for size in [10, 100, 250, 500, 1000]:
        create_and_save_matrices(size, output_dir)
