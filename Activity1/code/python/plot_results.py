import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_dir = os.path.join(script_dir, "..", "..", "data")
    
    input_csv = os.path.join(data_dir, "benchmark_results.csv")
    output_time_plot = os.path.join(data_dir, "plot_time_vs_size.png")
    output_mem_plot = os.path.join(data_dir, "plot_memory_vs_size.png")

    if not os.path.exists(input_csv):
        print(f"Error: File not found at {input_csv}")
        return

    print("=== Data Analysis and Graph Generation ===")
    
    df = pd.read_csv(input_csv)

    grouped = df.groupby(['size', 'language']).agg(
        median_time=('time_seconds', 'median'),
        q1_time=('time_seconds', lambda x: x.quantile(0.25)),
        q3_time=('time_seconds', lambda x: x.quantile(0.75)),
        median_memory=('memory', 'median')
    ).reset_index()

    grouped['iqr_time'] = grouped['q3_time'] - grouped['q1_time']

    print(grouped[['size', 'language', 'median_time', 'iqr_time', 'median_memory']].to_string(index=False))

    languages = df['language'].unique()
    colors = {'C': 'blue', 'Java': 'orange', 'Python': 'green'}
    markers = {'C': 'o', 'Java': 's', 'Python': '^'}

    # -- GRAPH 1: Time vs Size --
    plt.figure(figsize=(10, 6))
    for lang in languages:
        lang_data = grouped[grouped['language'] == lang]
        plt.plot(lang_data['size'], lang_data['median_time'], 
                 label=lang, color=colors.get(lang, 'black'), 
                 marker=markers.get(lang, 'x'), linewidth=2)

    plt.title("Execution Time vs Matrix Size", fontsize=14)
    plt.xlabel("Matrix Size (n)", fontsize=12)
    plt.ylabel("Median Time (seconds)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Language", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_time_plot)
    print(f"\nTime plot saved to: {output_time_plot}")

    # -- GRAPH 2: Memory vs. Size --
    plt.figure(figsize=(10, 6))
    for lang in languages:
        lang_data = grouped[grouped['language'] == lang]
        plt.plot(lang_data['size'], lang_data['median_memory'], 
                 label=lang, color=colors.get(lang, 'black'), 
                 marker=markers.get(lang, 'x'), linewidth=2, linestyle='-.')

    plt.title("Memory Usage vs Matrix Size", fontsize=14)
    plt.xlabel("Matrix Size (n)", fontsize=12)
    plt.ylabel("Median Memory (KB)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Language", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_mem_plot)
    print(f"Memory plot saved to: {output_mem_plot}")

if __name__ == "__main__":
    main()