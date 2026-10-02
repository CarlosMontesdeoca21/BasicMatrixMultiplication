import csv
import os
import statistics

def main():
    input_files = [
        "Activity1/data/results_c.csv",
        "Activity1/data/results_java.csv",
        "Activity1/data/results_python.csv"
    ]
    output_file = "Activity1/data/benchmark_results.csv"

    print("=== MASTER UNIFIER AND BENCHMARK ANALYZER ===")

    master_rows = []
    header = None

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"[NOTICE] File not found: {file_path}")
            continue
        
        print(f"Reading: {file_path}")
        with open(file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            
            if header is None:
                header = file_header
            
            for row in reader:
                if row:
                    master_rows.append(row)

    if not header:
        print("No valid data found to merge.")
        return

    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(master_rows)

    print(f"Master file successfully created in: {output_file}!")
    print(f"Total of consolidated records: {len(master_rows)}")
    print("\n--- STATISTICAL ANALYSIS OF TIMES  ---")

    grouped_data = {}
    for row in master_rows:
        size = int(row[0])
        language = row[1]
        time_val = float(row[4])

        key = (size, language)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(time_val)

    print(f"{'Size (n)':<12} | {'Language':<10} | {'Median (s)':<12} | {'MMin - MMax (s)':<18} | {'Std Dev (s)'}")
    print("-" * 75)

    for (size, language), times in sorted(grouped_data.items()):
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0

        print(f"{size:<12} | {language:<10} | {median_time:<12.6f} | {min_time:.6f} - {max_time:.6f} | {stdev_time:.6f}")

if __name__ == "__main__":
    main()