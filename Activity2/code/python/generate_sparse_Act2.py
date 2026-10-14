import random
import csv
import os

def generate_sparse_matrix(n, sparsity, seed=42):
    random.seed(seed)
    matrix = [[0.0] * n for _ in range(n)]
    density = 1.0 - sparsity / 100.0
    
    for i in range(n):
        for j in range(n):
            if random.random() < density:
                matrix[i][j] = random.random()
    
    return matrix

def dense_to_csr(matrix, n):
    values = []
    col_indices = []
    row_ptr = [0]
    
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0.0:
                values.append(matrix[i][j])
                col_indices.append(j)
        row_ptr.append(len(values))
    
    return values, col_indices, row_ptr

def count_nnz(matrix, n):
    count = 0
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != 0.0:
                count += 1
    return count

def save_dense_csv(matrix, path, n):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(n):
            writer.writerow(matrix[i])

def save_csr_csv(values, col_indices, row_ptr, base_path):
    with open(f"{base_path}_values.csv", 'w') as f:
        f.write(','.join(f"{v:.10f}" for v in values))
    
    with open(f"{base_path}_cols.csv", 'w') as f:
        f.write(','.join(str(c) for c in col_indices))
    
    with open(f"{base_path}_rows.csv", 'w') as f:
        f.write(','.join(str(r) for r in row_ptr))

if __name__ == '__main__':
    sizes = [100, 250]
    sparsity_levels = [50, 75, 90, 95, 99]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, '..', '..', 'data', 'sparse'))
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== SPARSE MATRIX INPUT GENERATOR ===")
    print(f"Output directory: {output_dir}\n")
    
    for n in sizes:
        for sparsity in sparsity_levels:
            print(f"Generating n={n}, sparsity={sparsity}%...")
            
            A = generate_sparse_matrix(n, sparsity, seed=42)
            B = generate_sparse_matrix(n, sparsity, seed=43)
            
            nnz_A = count_nnz(A, n)
            nnz_B = count_nnz(B, n)
            density_A = 100.0 * nnz_A / (n * n)
            density_B = 100.0 * nnz_B / (n * n)
            
            save_dense_csv(A, os.path.join(output_dir, f'dense_A_{n}_{sparsity}.csv'), n)
            save_dense_csv(B, os.path.join(output_dir, f'dense_B_{n}_{sparsity}.csv'), n)
            
            A_vals, A_cols, A_rows = dense_to_csr(A, n)
            B_vals, B_cols, B_rows = dense_to_csr(B, n)
            
            save_csr_csv(A_vals, A_cols, A_rows,
                        os.path.join(output_dir, f'csr_A_{n}_{sparsity}'))
            save_csr_csv(B_vals, B_cols, B_rows,
                        os.path.join(output_dir, f'csr_B_{n}_{sparsity}'))
            
            print(f"  nnz_A={nnz_A} ({density_A:.1f}% density), nnz_B={nnz_B} ({density_B:.1f}% density)")
            print(f"  Files: dense_A_{n}_{sparsity}.csv, csr_A_{n}_{sparsity}_*.csv")
            print()
    
    print("Generation complete!")
    print("\nNote: Sparsity = % of zeros, Density = % of non-zeros")
    print("Seed used: 42 for matrix A, 43 for matrix B (different to ensure independence)")
