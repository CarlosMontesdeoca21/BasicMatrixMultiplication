#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

long get_memory_usage() {
    return 0; 
}

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
    if (!file) return 0;

    char line[65536];
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

double compute_checksum(double** C, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += C[i][j];
        }
    }
    return sum;
}

int main() {
    int sizes[] = {10, 100, 250};
    int num_sizes = 3;
    int warm_ups = 2;
    int repetitions = 5;

    FILE* csv_file = fopen("Activity1/data/results_c.csv", "w");
    if (!csv_file) {
        perror("Error creating the results file");
        return 1;
    }

    fprintf(csv_file, "size,language,numeric_type,repetition,time_seconds,memory\n");

    printf("=== BENCHMARK IN C (Warm-up + 5 Repetitions) ===\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        char path_A[256], path_B[256];
        
        snprintf(path_A, sizeof(path_A), "Activity1/data/matrix_A_%d.csv", n);
        snprintf(path_B, sizeof(path_B), "Activity1/data/matrix_B_%d.csv", n);

        double** A = allocate_matrix(n);
        double** B = allocate_matrix(n);
        double** C = allocate_matrix(n);

        if (!read_csv(path_A, A, n) || !read_csv(path_B, B, n)) {
            printf("C (n=%d): Input files not found.\n", n);
            free_matrix(A, n); free_matrix(B, n); free_matrix(C, n);
            continue;
        }

        for (int w = 0; w < warm_ups; w++) {
            multiply_matrices(A, B, C, n);
        }

        for (int rep = 1; rep <= repetitions; rep++) {
            clock_t start = clock();
            
            multiply_matrices(A, B, C, n);
            
            clock_t end = clock();

            double elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
            long memory_kb = get_memory_usage();

            fprintf(csv_file, "%d,C,double,%d,%.6f,%ld\n", n, rep, elapsed_time, memory_kb);
            
            double checksum = compute_checksum(C, n);
            printf("C | n=%d | Rep %d | Time: %.6f s | Memory: %ld KB | Checksum: %.4f\n", 
                   n, rep, elapsed_time, memory_kb, checksum);
        }

        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);
    }

    fclose(csv_file);
    return 0;
}