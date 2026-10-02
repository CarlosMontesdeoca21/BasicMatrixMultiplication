import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;

public class MatrixMult {

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

    public static void writeCSV(String filePath, double[][] matrix, int n) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filePath))) {
            for (int i = 0; i < n; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < n; j++) {
                    sb.append(String.format(Locale.US, "%.6f", matrix[i][j]));
                    if (j < n - 1) sb.append(",");
                }
                bw.write(sb.toString());
                bw.newLine();
            }
        }
    }

    public static void main(String[] args) {
        int[] sizes = {10, 100, 250};

        System.out.println("=== CSV Generation in Java (Multiple N) ===");

        for (int n : sizes) {
            String pathA = "Activity1/data/matrix_A_" + n + ".csv";
            String pathB = "Activity1/data/matrix_B_" + n + ".csv";
            String pathOut = "Activity1/data/result_Java_" + n + ".csv";

            try {
                double[][] A = readCSV(pathA, n);
                double[][] B = readCSV(pathB, n);
                double[][] C = new double[n][n];

                multiplyMatrices(A, B, C, n);
                writeCSV(pathOut, C, n);

                System.out.println("Java (n=" + n + "): Successfully saved in result_Java_" + n + ".csv");

            } catch (IOException e) {
                System.out.println("Java (n=" + n + "):Input files were not found, or an error occurred.");
            }
        }
    }
}