import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data"))
    
    input_csv = os.path.join(data_dir, "results_sparse_java.csv")
    output_plot = os.path.join(data_dir, "plot_sparse_crossover.png")

    if not os.path.exists(input_csv):
        print(f"Error: File not found in {input_csv}")
        return

    print("=== Generating Sparse Matrix Plot (Part B) ===")
    
    df = pd.read_csv(input_csv)
    grouped = df.groupby(['size', 'sparsity', 'method'])['time_seconds'].median().reset_index()

    df_250 = grouped[grouped['size'] == 250]
    
    plt.figure(figsize=(10, 6))
    
    # Línea del método Denso
    dense_data = df_250[df_250['method'] == 'Dense']
    plt.plot(dense_data['sparsity'], dense_data['time_seconds'], 
             label='Dense Baseline', color='red', marker='o', linewidth=2)
             
    # Línea del método CSR
    csr_data = df_250[df_250['method'] == 'CSR']
    plt.plot(csr_data['sparsity'], csr_data['time_seconds'], 
             label='CSR Sparse', color='blue', marker='s', linewidth=2)

    plt.title("Performance: Dense vs CSR (Size n=250)", fontsize=14)
    plt.xlabel("Sparsity Level (% zeros)", fontsize=12)
    plt.ylabel("Median Time (seconds)", fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig(output_plot)
    print(f"Plot saved to: {output_plot}")

if __name__ == "__main__":
    main()