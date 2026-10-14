import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, "..", "..", "data"))
    
    input_csv = os.path.join(data_dir, "results_java_act2.csv")
    output_time_plot = os.path.join(data_dir, "plot_time_vs_size_act2.png")
    output_mem_plot = os.path.join(data_dir, "plot_memory_vs_size_act2.png")

    if not os.path.exists(input_csv):
        print(f"Error: File not found at {input_csv}. Make sure to run BenchmarkAct2.java first.")
        return

    print("=== Data Analysis and Graph Generation (Part A) ===")
    
    df = pd.read_csv(input_csv)

    grouped = df.groupby(['size', 'method']).agg(
        median_time=('time_seconds', 'median'),
        q1_time=('time_seconds', lambda x: x.quantile(0.25)),
        q3_time=('time_seconds', lambda x: x.quantile(0.75)),
        median_memory=('memory', 'median')
    ).reset_index()
    
    grouped['iqr_time'] = grouped['q3_time'] - grouped['q1_time']
    print(grouped[['size', 'method', 'median_time', 'iqr_time', 'median_memory']].to_string(index=False))

    methods = df['method'].unique()
    colors = {'Baseline': 'red', 'Opt1': 'blue', 'Opt2': 'green'}
    markers = {'Baseline': 'o', 'Opt1': 's', 'Opt2': '^'}

    plt.figure(figsize=(10, 6))
    for method in methods:
        method_data = grouped[grouped['method'] == method]
        plt.plot(method_data['size'], method_data['median_time'], 
                 label=method, color=colors.get(method, 'black'), 
                 marker=markers.get(method, 'x'), linewidth=2)

    plt.title("Execution Time vs Matrix Size (Java Optimizations)", fontsize=14)
    plt.xlabel("Matrix Size (n)", fontsize=12)
    plt.ylabel("Median Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_time_plot)
    print(f"\nTime plot saved to: {output_time_plot}")

    plt.figure(figsize=(10, 6))
    for method in methods:
        method_data = grouped[grouped['method'] == method]
        plt.plot(method_data['size'], method_data['median_memory'], 
                 label=method, color=colors.get(method, 'black'), 
                 marker=markers.get(method, 'x'), linewidth=2, linestyle='-.')

    plt.title("Memory Usage vs Matrix Size (Java Optimizations)", fontsize=14)
    plt.xlabel("Matrix Size (n)", fontsize=12)
    plt.ylabel("Median Memory (KB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_mem_plot)
    print(f"Memory plot saved to: {output_mem_plot}")

if __name__ == "__main__":
    main()