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

## 5. Run Instructions
*(To be completed when the code is ready)*