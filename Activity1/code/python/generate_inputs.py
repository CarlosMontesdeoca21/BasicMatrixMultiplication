import random
import csv
import os

def create_and_save_matrices(n):
    random.seed(42) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = os.path.join(script_dir, "..", "..", "data")
    
    output_dir = os.path.normpath(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]
    
    with open(os.path.join(output_dir, f"A_{n}x{n}.csv"), "w", newline="") as f:
        csv.writer(f).writerows(A)
    with open(os.path.join(output_dir, f"B_{n}x{n}.csv"), "w", newline="") as f:
        csv.writer(f).writerows(B)

for size in [10, 100, 250]:
    create_and_save_matrices(size)