#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void multiply_matrices(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

double** allocate_matrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int read_csv(const char* filename, double** matrix, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening the input file");
        return 0;
    }

    char line[1024];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < n) {
        int col = 0;
        char* token = strtok(line, ",");
        while (token && col < n) {
            matrix[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }
    fclose(file);
    return 1;
}

int write_csv(const char* filename, double** matrix, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error creating the output file");
        return 0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%f", matrix[i][j]);
            if (j < n - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);
    return 1;
}

int main() {
    int n = 10;
    char path_A[256], path_B[256], path_out[256];
    
    snprintf(path_A, sizeof(path_A), "Activity1/data/matrix_A_%d.csv", n);
    snprintf(path_B, sizeof(path_B), "Activity1/data/matrix_B_%d.csv", n);
    snprintf(path_out, sizeof(path_out), "Activity1/data/result_C_%d.csv", n);

    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    if (!read_csv(path_A, A, n) || !read_csv(path_B, B, n)) {
        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);
        return 1;
    }

    multiply_matrices(A, B, C, n);

    if (write_csv(path_out, C, n)) {
        printf("C: Multiplication for n=%d completed and saved successfully..\n", n);
    }

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}