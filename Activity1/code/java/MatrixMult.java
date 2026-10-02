import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class MatrixMult {

    public static double[][] multiplyMatrices(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    public static double[][] readCSV(String filePath, int n) throws IOException {
        double[][] matrix = new double[n][n];
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            int row = 0;
            while ((line = br.readLine()) != null && row < n) {
                String[] values = line.split(",");
                for (int col = 0; col < values.length; col++) {
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
                    sb.append(String.format(java.util.Locale.US, "%.6f", matrix[i][j]));
                    if (j < n - 1) sb.append(",");
                }
                bw.write(sb.toString());
                bw.newLine();
            }
        }
    }

    public static void main(String[] args) {
        int n = 10;
        String pathA = "Activity1/data/matrix_A_" + n + ".csv";
        String pathB = "Activity1/data/matrix_B_" + n + ".csv";
        String pathOut = "Activity1/data/result_Java_" + n + ".csv";

        try {
            double[][] A = readCSV(pathA, n);
            double[][] B = readCSV(pathB, n);
            
            double[][] C = multiplyMatrices(A, B, n);
            
            writeCSV(pathOut, C, n);
            System.out.println("Java: Multiplication for n=" + n + " completed and saved successfully.");
        } catch (IOException e) {
            System.err.println("Error processing CSV files in Java: " + e.getMessage());
        }
    }
}