import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;

public class BenchmarkAct2 {

    private static final int BLOCK_SIZE = 32;

    private static final int WARM_UPS = 2;
    private static final int REPETITIONS = 5;

    public static long getMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        runtime.gc();
        long memoryUsedBytes = runtime.totalMemory() - runtime.freeMemory();
        return memoryUsedBytes / 1024;
    }

    public static void multiplyMatricesBaseline(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }

    public static void multiplyMatricesOpt1(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
            }
            for (int k = 0; k < n; k++) {
                double aik = A[i][k];
                for (int j = 0; j < n; j++) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
    }

    public static void multiplyMatricesOpt2(double[][] A, double[][] B, double[][] C, int n, int blockSize) {
        for (int ii = 0; ii < n; ii += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                int iEnd = Math.min(ii + blockSize, n);
                int jEnd = Math.min(jj + blockSize, n);
                for (int i = ii; i < iEnd; i++) {
                    Arrays.fill(C[i], jj, jEnd, 0.0);
                }
                for (int kk = 0; kk < n; kk += blockSize) {
                    int kEnd = Math.min(kk + blockSize, n);
                    for (int i = ii; i < iEnd; i++) {
                        for (int k = kk; k < kEnd; k++) {
                            double aik = A[i][k];
                            for (int j = jj; j < jEnd; j++) {
                                C[i][j] += aik * B[k][j];
                            }
                        }
                    }
                }
            }
        }
    }

    public static double[][] readCSV(String filePath, int n) throws IOException {
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
        int[] sizes = {10, 100, 250};

        String csvPath = "data/results_java_act2.csv";
        File outputFile = new File(csvPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        System.out.println("=== BENCHMARK ACT2 IN JAVA (Baseline vs Opt1 vs Opt2) ===");
        System.out.println("Protocol: " + WARM_UPS + " warm-ups + " + REPETITIONS + " repetitions per method and size");
        System.out.println("Opt2 block size: " + BLOCK_SIZE + "\n");

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile))) {
            bw.write("size,language,method,repetition,time_seconds,memory");
            bw.newLine();

            for (int n : sizes) {
                String pathA = "data/matrix_A_" + n + ".csv";
                String pathB = "data/matrix_B_" + n + ".csv";

                double[][] A, B;
                try {
                    A = readCSV(pathA, n);
                    B = readCSV(pathB, n);
                } catch (IOException e) {
                    System.out.println("Java Act2 (n=" + n + "): Input files not found.");
                    continue;
                }

                double[][] C = new double[n][n];
                double baselineChecksum = Double.NaN;
                boolean validationOk = true;

                String[] methodNames = {"Baseline", "Opt1", "Opt2"};

                for (int m = 0; m < methodNames.length; m++) {
                    String methodName = methodNames[m];

                    for (int w = 0; w < WARM_UPS; w++) {
                        runMethod(methodName, A, B, C, n);
                    }

                    for (int rep = 1; rep <= REPETITIONS; rep++) {
                        long startTime = System.nanoTime();

                        runMethod(methodName, A, B, C, n);

                        long endTime = System.nanoTime();
                        double elapsedTimeSec = (endTime - startTime) / 1_000_000_000.0;

                        long memoryKb = getMemoryUsage();

                        String rowData = String.format(Locale.US, "%d,Java,%s,%d,%.6f,%d",
                                n, methodName, rep, elapsedTimeSec, memoryKb);
                        bw.write(rowData);
                        bw.newLine();

                        double checksum = computeChecksum(C, n);
                        System.out.printf(Locale.US, "Java | n=%d | %-8s | Rep %d | Time: %.6f s | Memory: %d KB | Checksum: %.4f%n",
                                n, methodName, rep, elapsedTimeSec, memoryKb, checksum);

                        if (methodName.equals("Baseline")) {
                            baselineChecksum = checksum;
                        } else if (!Double.isNaN(baselineChecksum)
                                && Math.abs(checksum - baselineChecksum) > 1e-6 * Math.max(1.0, Math.abs(baselineChecksum))) {
                            validationOk = false;
                            System.out.printf(Locale.US, "  [VALIDATION WARNING] Checksum mismatch for %s at n=%d!%n", methodName, n);
                        }
                    }
                }
                System.out.printf(Locale.US, "Validation for n=%d: %s%n%n", n,
                        validationOk ? "OK (all methods match baseline)" : "FAILED");
            }
            System.out.println("Java Act2 results successfully saved to " + csvPath + "!");

        } catch (IOException e) {
            System.out.println("Error writing the file results_java_act2.csv: " + e.getMessage());
        }
    }

    private static void runMethod(String methodName, double[][] A, double[][] B, double[][] C, int n) {
        switch (methodName) {
            case "Baseline":
                multiplyMatricesBaseline(A, B, C, n);
                break;
            case "Opt1":
                multiplyMatricesOpt1(A, B, C, n);
                break;
            case "Opt2":
                multiplyMatricesOpt2(A, B, C, n, BLOCK_SIZE);
                break;
            default:
                throw new IllegalArgumentException("Unknown method: " + methodName);
        }
    }
}
