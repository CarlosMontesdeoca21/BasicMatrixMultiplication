import java.io.*;
import java.util.*;

public class BenchmarkSparse {

    private static final int WARM_UPS = 2;
    private static final int REPETITIONS = 5;

    public static void multiplyDense(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    public static void multiplyCSR(double[] valA, int[] colA, int[] rowPtrA,
                                   double[] valB, int[] colB, int[] rowPtrB,
                                   double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            Arrays.fill(C[i], 0.0);
        }

        for (int i = 0; i < n; i++) {

            for (int pA = rowPtrA[i]; pA < rowPtrA[i + 1]; pA++) {
                int k = colA[pA];
                double vA = valA[pA];
                
                for (int pB = rowPtrB[k]; pB < rowPtrB[k + 1]; pB++) {
                    int j = colB[pB];
                    C[i][j] += vA * valB[pB];
                }
            }
        }
    }

    public static double[][] readDenseCSV(String filePath, int n) throws IOException {
        double[][] matrix = new double[n][n];
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null && row < n) {
                String[] values = line.split(",");
                for (int col = 0; col < values.length && col < n; col++) {
                    matrix[row][col] = Double.parseDouble(values[col]);
                }
                row++;
            }
        }
        return matrix;
    }

    public static double[] readDoubleArray(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine();
            if (line == null || line.trim().isEmpty()) return new double[0];
            String[] parts = line.split(",");
            double[] arr = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                arr[i] = Double.parseDouble(parts[i]);
            }
            return arr;
        }
    }

    public static int[] readIntArray(String filePath) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine();
            if (line == null || line.trim().isEmpty()) return new int[0];
            String[] parts = line.split(",");
            int[] arr = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                arr[i] = Integer.parseInt(parts[i]);
            }
            return arr;
        }
    }

    public static double computeChecksum(double[][] C, int n) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += C[i][j];
            }
        }
        return sum;
    }
    public static void main(String[] args) {
        int[] sizes = {100, 250};
        int[] sparsities = {50, 75, 90, 95, 99};
        
        String outputCsv = "data/results_sparse_java.csv";
        File outputFile = new File(outputCsv);
        if (outputFile.getParentFile() != null && !outputFile.getParentFile().exists()) {
            outputFile.getParentFile().mkdirs();
        }


        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile))) {
            bw.write("size,sparsity,method,repetition,time_seconds");
            bw.newLine();

            for (int n : sizes) {
                for (int sparsity : sparsities) {
                    System.out.println("\nEvaluating n=" + n + " | Dispersion=" + sparsity + "%");
                    
                    String baseDenseA = "data/sparse/dense_A_" + n + "_" + sparsity + ".csv";
                    String baseDenseB = "data/sparse/dense_B_" + n + "_" + sparsity + ".csv";
                    
                    String baseCsrA = "data/sparse/csr_A_" + n + "_" + sparsity;
                    String baseCsrB = "data/sparse/csr_B_" + n + "_" + sparsity;

                    try {
                        double[][] denseA = readDenseCSV(baseDenseA, n);
                        double[][] denseB = readDenseCSV(baseDenseB, n);
                        double[][] denseC = new double[n][n];

                        double[] valA = readDoubleArray(baseCsrA + "_values.csv");
                        int[] colA = readIntArray(baseCsrA + "_cols.csv");
                        int[] rowA = readIntArray(baseCsrA + "_rows.csv");

                        double[] valB = readDoubleArray(baseCsrB + "_values.csv");
                        int[] colB = readIntArray(baseCsrB + "_cols.csv");
                        int[] rowB = readIntArray(baseCsrB + "_rows.csv");
                        double[][] csrC = new double[n][n];

                        double denseChecksum = 0.0;

                        for (int w = 0; w < WARM_UPS; w++) multiplyDense(denseA, denseB, denseC, n);
                        for (int rep = 1; rep <= REPETITIONS; rep++) {
                            long start = System.nanoTime();
                            multiplyDense(denseA, denseB, denseC, n);
                            long end = System.nanoTime();
                            double timeSec = (end - start) / 1e9;
                            
                            bw.write(String.format(Locale.US, "%d,%d,Dense,%d,%.6f", n, sparsity, rep, timeSec));
                            bw.newLine();
                            denseChecksum = computeChecksum(denseC, n);
                        }

                        for (int w = 0; w < WARM_UPS; w++) multiplyCSR(valA, colA, rowA, valB, colB, rowB, csrC, n);
                        for (int rep = 1; rep <= REPETITIONS; rep++) {
                            long start = System.nanoTime();
                            multiplyCSR(valA, colA, rowA, valB, colB, rowB, csrC, n);
                            long end = System.nanoTime();
                            double timeSec = (end - start) / 1e9;
                            
                            bw.write(String.format(Locale.US, "%d,%d,CSR,%d,%.6f", n, sparsity, rep, timeSec));
                            bw.newLine();
                            
                            double csrChecksum = computeChecksum(csrC, n);
                            if (rep == REPETITIONS && Math.abs(denseChecksum - csrChecksum) > 1e-5) {
                                System.out.println(" [ERROR] Checksums do not match! Dense: " + denseChecksum + " vs CSR: " + csrChecksum);
                            }
                        }
                        
                        System.out.println("  -> Tests successfully completed (Matching results).");

                    } catch (IOException e) {
                        System.out.println(" [NOTICE] No files were found for n=" + n + ", dispersion=" + sparsity);
                    }
                }
            }
            System.out.println("\nResults saved successfully in " + outputCsv);
        } catch (IOException e) {
            System.out.println("General error writing CSV: " + e.getMessage());
        }
    }
}