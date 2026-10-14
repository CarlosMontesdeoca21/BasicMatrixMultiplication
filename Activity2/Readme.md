# Activity 2: Matrix Multiplication Optimization and Sparse Formats

This directory contains the implementation and benchmark suite for Activity 2. The project is divided into two parts: cache optimization for dense matrices (Part A) and evaluation of sparse matrix multiplication using the CSR format (Part B).

## 1. Required Dependencies

To ensure the project runs correctly and results can be reproduced, the environment must include:

- **Java Development Kit (JDK):** Version 8 or later, for compiling and running the mathematical benchmarks.
- **Python:** Version 3.6 or later, used as the orchestrator for data generation and visualization.
- **Python libraries:**
  - `pandas` for processing and grouping CSV data.
  - `matplotlib` for generating performance plots.
  - **Install the libraries:**
    ```bash
    pip install pandas matplotlib
    ```

## 2. Methodology and Experimental Protocol

The benchmark follows a strict academic protocol to ensure fair and accurate measurements while accounting for the effects of the Java Virtual Machine (JVM) Just-In-Time (JIT) compiler:

- **Measurement Protocol:** For each configuration (matrix size, sparsity level, and method), run **2 unrecorded warm-up iterations** to initialize the cache and allow JIT optimizations, followed by **5 official recorded repetitions** measuring execution time (in seconds) and memory usage (in KB).
- **Strict Validation:** The optimized implementations (Opt1, Opt2, and CSR) calculate a checksum at the end. The script internally verifies that this result matches the baseline to ensure mathematical correctness.
- **Part A (Dense Optimization):** Compares performance for matrices of sizes \(n \in \{10, 100, 250\}\) using:
  - **Baseline:** Traditional triple nested loop, `i-j-k`.
  - **Opt1 (Loop Reordering):** Reordered loops, `i-k-j`, to maximize L1/L2 cache hits.
  - **Opt2 (Cache Blocking/Tiling):** Block matrix multiplication with a block size of 32×32 to improve spatial locality.
- **Part B (Sparse Matrices):** Compares the dense method against the compressed CSR (*Compressed Sparse Row*) format for matrices of sizes \(n \in \{100, 250\}\), using progressively higher sparsity levels (50%, 75%, 90%, 95%, and 99%).

## 3. Strict Execution Order

To evaluate the project correctly, **open a terminal in the root directory of the `Activity2` folder** and run the following commands in sequence. This ensures that the input data is generated locally before any scripts attempt to read it.

### Step 1: Generate Input Data

Generate the dense and sparse matrix CSV files in the `data/` directory.

```bash
# 1. Generate dense matrices for Part A
python code/python/generate_inputs.py

# 2. Generate matrices and CSR conversions for Part B (generates 80 files)
python code/python/generate_sparse_inputs.py
```

### Step 2: Run the Benchmarks (Java)

Compile and run the mathematical kernels. Results will be saved automatically to CSV files.

```bash
# 1. Compile the Java files
javac code/java/BenchmarkAct2.java
javac code/java/BenchmarkSparse.java

# 2. Run the Part A benchmark (Loop Reordering & Tiling)
java -cp code/java BenchmarkAct2

# 3. Run the Part B benchmark (Dense vs. CSR)
java -cp code/java BenchmarkSparse
```

### Step 3: Generate Analysis Plots

Process the results and generate the corresponding visualizations in the `data/` directory.

```bash
# 1. Generate time and memory plots (Part A)
python code/python/plot_results.py

# 2. Generate the CSR vs. dense crossover-point plot (Part B)
python code/python/plot_sparse.py
```
