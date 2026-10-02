import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

public class Benchmark {

    public static long getMemoryUsage() {
        return 0;
    }

    public static void multiplyMatrices(double[][] A, double[][] B, double[][] C, int n) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
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
        int warmUps = 2;
        int repetitions = 5;

        String csvPath = "Activity1/data/results_java.csv";

        System.out.println("=== BENCHMARK IN JAVA (Warm-up + 5 Repetitions) ===");

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(csvPath))) {
            bw.write("size,language,numeric_type,repetition,time_seconds,memory");
            bw.newLine();

            for (int n : sizes) {
                String pathA = "Activity1/data/matrix_A_" + n + ".csv";
                String pathB = "Activity1/data/matrix_B_" + n + ".csv";

                double[][] A, B;
                try {
                    A = readCSV(pathA, n);
                    B = readCSV(pathB, n);
                } catch (IOException e) {
                    System.out.println("Java (n=" + n + "): Input files not found.");
                    continue;
                }

                double[][] C = new double[n][n];

                for (int w = 0; w < warmUps; w++) {
                    multiplyMatrices(A, B, C, n);
                }

                for (int rep = 1; rep <= repetitions; rep++) {
                    long startTime = System.nanoTime();

                    multiplyMatrices(A, B, C, n);

                    long endTime = System.nanoTime();
                    double elapsedTimeSec = (endTime - startTime) / 1_000_000_000.0;
                    long memoryKb = getMemoryUsage();

                    String rowData = String.format(Locale.US, "%d,Java,double,%d,%.6f,%d", n, rep, elapsedTimeSec, memoryKb);
                    bw.write(rowData);
                    bw.newLine();

                    double checksum = computeChecksum(C, n);
                    System.out.printf(Locale.US, "Java | n=%d | Rep %d | Time: %.6f s | Memory: %d KB | Checksum: %.4f\n", 
                                      n, rep, elapsedTimeSec, memoryKb, checksum);
                }
            }
            System.out.println("Java results successfully saved to " + csvPath + "!");

        } catch (IOException e) {
            System.out.println("Error writing the file results_java.csv: " + e.getMessage());
        }
    }
}