# Basic Matrix Multiplication in Different Languages (Assignment 1)

## 1. Experimental Environment

### Hardware Specifications
* **CPU Model:** Intel Core Ultra i7-155H
* **Physical Cores:** 16
* **Logical Processors (Threads):** 22
* **RAM:** 32 GB
* **Operating System:** Windows 11

### Software & Languages
* **C Compiler:** 16.2.0
* **Java Version:** 21.0.10
* **Python Version:** 3.12.10

## 2. Compilation Flags and Runtime Settings
* **C:** Compiled with `gcc -O3 -Wall` (optimization -O3).
* **Java:** Standard JVM settings. Warm-up runs are excluded from the final measurements.
* **Python:** Standard interpreter.

## 3. Resource Budgets
* **Maximum Execution Time:** 5 minutes per matrix multiplication.
* **Memory Budget:** 4 GB maximum heap/resident memory per process.

## 4. Methodology and Test Sizes
* **Test Sequence (n x n):** 10, 100, 250, 500, 1000, 2000.
* **Correctness Validation:** The size \(n=10\) will be used to validate the results against a trusted reference before performance benchmarking.
* **Execution limits:** The sequence will be stopped for a specific implementation if it reaches the declared maximum execution time or memory budget.

The codebase is intentionally divided into two sets of files to separate the core logic from the testing infrastructure:
* **Base Implementations (`matrix_mult.*`):** These files contain the pure, reference implementations of the `O(n^3)` algorithm. They serve as evidence of the initial assignment phase and can be used for single-run, standalone matrix multiplications without any experimental overhead.
* **Benchmark Scripts (`benchmark.*`):** These files encapsulate the matrix multiplication logic alongside high-precision timers, warm-up iterations, and CSV data export. This encapsulation avoids import overheads and allows the compilers to fully optimize the benchmark loop, ensuring strict isolation for performance measurement.

## 5. Run Instructions

### 5.1. Input Data Generation
To replicate the benchmarking experiment for matrix multiplication in C, Java, and Python reproducibly, follow the execution order detailed below.

The first step is to execute the generator script `(python generate_inputs.py)` to create the test matrices for the entire sequence of sizes (`matrix_A_n.csv` and `matrix_B_n.csv`). These will be automatically stored in the project's data folder:

### 5.2 Individual Benchmarks Execution
Each language has its own script or program that performs warm-up rounds (2 rounds), executes 5 official consecutive repetitions, and exports the raw data to an individual CSV file.

* **C Language:** 
    1. Compile the source code enabling optimization flags (-O3 and -Wall):

        `gcc Activity1/code/c/benchmark.c -O3 -Wall -o Activity1/code/c/benchmark.exe`
    2. Run the generated binary to produce results_c.csv:

        `./Activity1/code/c/benchmark.exe`

* **Java Language:**
    1. Compile the Java class:

        `javac Activity1/code/java/Benchmark.java`
    
    2. Run the benchmark to produce results_java.csv

        `java -cp Activity1/code/java Benchmark`

* **Python Language:**
    1. Directly execute the Python script to produce results_python.csv:

        `python Activity1/code/python/benchmark.py`

### 5.3 Results Consolidation
Once the three individual files are generated `(results_c.csv, results_java.csv, and results_python.csv)`, run the unifier script (`python merge_results.py`) to consolidate all measurements into a single master file. This step also outputs a preliminary statistical analysis (median, extreme values, and variability) needed for the final report

* **The consolidated master file at:** `(Activity1/data/benchmark_results.csv)`
* **A summary table in the console with the time and variability statistics ready for the final report documentation.**