#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <psapi.h>
#include <omp.h>

#ifndef BUILD_LABEL
#define BUILD_LABEL O3
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define MAX_LIST 32

size_t get_memory_usage(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
    return 0;
}

/*
 * Monotonic high-resolution wall-clock timer.
 * omp_get_wtime() in libgomp on Windows only resolves ~1 ms, which is too
 * coarse for the smaller sizes, so we use QueryPerformanceCounter directly.
 */
static double get_time(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER cnt;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

/* Sequential baseline: classic i-j-k triple loop (same as Activity 1). */
void multiply_seq(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/*
 * Sequential i-k-j variant (optional SIMD study only).
 * Same arithmetic work O(n^3) and same storage, but the innermost loop
 * updates a full row of C with a rank-1 update, which gcc can auto-vectorize
 * without needing -ffast-math (unlike the FP reduction in i-j-k).
 */
void multiply_seq_ikj(double** A, double** B, double**C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double aik = A[i][k];
            for (int j = 0; j < n; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
}

/*
 * Parallel version: identical i-j-k kernel, parallelized over the outer
 * loop (rows). Each row of C is written by exactly one thread, so there
 * are no races and no synchronization is needed.
 */
void multiply_omp(double** A, double** B, double** C, int n, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
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

int write_matrix_csv(const char* filename, double** matrix, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(file, "%.6f%s", matrix[i][j], (j < n - 1) ? "," : "");
        }
        fprintf(file, "\n");
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

double max_abs_diff(double** X, double** Y, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = fabs(X[i][j] - Y[i][j]);
            if (d > m) m = d;
        }
    }
    return m;
}

int parse_int_list(const char* s, int* out, int max) {
    int count = 0;
    const char* p = s;
    while (*p && count < max) {
        out[count++] = atoi(p);
        p = strchr(p, ',');
        if (!p) break;
        p++;
    }
    return count;
}

/*
 * Default data directory and output file are resolved relative to the
 * executable location (Activity3/code/c/ -> Activity3/data), so the program
 * works regardless of the current working directory.
 */
static char g_default_datadir[1024];
static char g_default_output[1024];

static void resolve_default_paths(void) {
    char exe_path[512];
    DWORD len = GetModuleFileNameA(NULL, exe_path, sizeof(exe_path));
    if (len == 0 || len >= sizeof(exe_path)) {
        snprintf(g_default_datadir, sizeof(g_default_datadir), "Activity3/data");
        snprintf(g_default_output, sizeof(g_default_output), "Activity3/data/results_omp_act3.csv");
        return;
    }
    char* last_slash = strrchr(exe_path, '\\');
    if (last_slash) *last_slash = '\0';               /* ...\Activity3\code\c */
    last_slash = strrchr(exe_path, '\\');
    if (last_slash) *last_slash = '\0';               /* ...\Activity3\code   */
    last_slash = strrchr(exe_path, '\\');
    if (last_slash) *last_slash = '\0';               /* ...\Activity3        */
    snprintf(g_default_datadir, sizeof(g_default_datadir), "%s\\data", exe_path);
    snprintf(g_default_output, sizeof(g_default_output), "%s\\data\\results_omp_act3.csv", exe_path);
}

int main(int argc, char** argv) {
    int sizes[MAX_LIST];
    int num_sizes = 0;
    int threads[MAX_LIST];
    int num_threads_cfg = 0;
    int warm_ups = 2;
    int repetitions = 5;
    int validate_n = -1;
    const char* method = "all";

    resolve_default_paths();
    const char* datadir = g_default_datadir;
    const char* output = g_default_output;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            num_sizes = parse_int_list(argv[++i], sizes, MAX_LIST);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads_cfg = parse_int_list(argv[++i], threads, MAX_LIST);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            warm_ups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            method = argv[++i];
        } else if (strcmp(argv[i], "--datadir") == 0 && i + 1 < argc) {
            datadir = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--validate") == 0 && i + 1 < argc) {
            validate_n = atoi(argv[++i]);
        } else {
            fprintf(stderr,
                "Usage: %s [--sizes 10,100,250,500,1000] [--threads 1,2,4,8,16,22,28]\n"
                "          [--method all|seq|omp|ikj] [--warmups 2] [--reps 5]\n"
                "          [--datadir DIR] [--output FILE] [--validate N]\n",
                argv[0]);
            return 1;
        }
    }

    if (num_sizes == 0) {
        int defaults[] = {10, 100, 250, 500, 1000};
        num_sizes = 5;
        memcpy(sizes, defaults, sizeof(defaults));
    }
    if (num_threads_cfg == 0) {
        int defaults[] = {1, 2, 4, 8, 16, 22, 28};
        num_threads_cfg = 7;
        memcpy(threads, defaults, sizeof(defaults));
    }

    omp_set_dynamic(0);

    printf("=== MATRIX MULTIPLICATION BENCHMARK: SEQUENTIAL vs OPENMP ===\n");
    printf("Build label: %s\n", STR(BUILD_LABEL));
    printf("Data directory: %s\n", datadir);
    printf("Output file: %s\n", output);
    printf("OpenMP processors: %d | max threads: %d | dynamic adjustment disabled\n",
           omp_get_num_procs(), omp_get_max_threads());
    printf("Warm-ups: %d | Measured repetitions: %d\n\n", warm_ups, repetitions);

    /* ---- Validation mode: dump the sequential result of size N to CSV ---- */
    if (validate_n > 0) {
        char path_A[2048], path_B[2048], path_out[2048];
        snprintf(path_A, sizeof(path_A), "%s/matrix_A_%d.csv", datadir, validate_n);
        snprintf(path_B, sizeof(path_B), "%s/matrix_B_%d.csv", datadir, validate_n);
        snprintf(path_out, sizeof(path_out), "%s/result_C_%d.csv", datadir, validate_n);

        double** A = allocate_matrix(validate_n);
        double** B = allocate_matrix(validate_n);
        double** C = allocate_matrix(validate_n);
        double** C2 = allocate_matrix(validate_n);

        if (!read_csv(path_A, A, validate_n) || !read_csv(path_B, B, validate_n)) {
            printf("Validate: input files not found for n=%d.\n", validate_n);
            return 1;
        }

        multiply_seq(A, B, C, validate_n);
        int vthreads = 4 < omp_get_max_threads() ? 4 : omp_get_max_threads();
        multiply_omp(A, B, C2, validate_n, vthreads);

        if (!write_matrix_csv(path_out, C, validate_n)) {
            printf("Validate: could not write %s\n", path_out);
            return 1;
        }

        double diff = max_abs_diff(C, C2, validate_n);
        printf("Validation n=%d | sequential checksum: %.6f\n", validate_n, compute_checksum(C, validate_n));
        printf("Max |sequential - openmp(%d threads)| elementwise diff: %.3e\n", vthreads, diff);
        printf("Result matrix written to %s\n", path_out);

        free_matrix(A, validate_n); free_matrix(B, validate_n);
        free_matrix(C, validate_n); free_matrix(C2, validate_n);
        return 0;
    }

    /* ------------------------------ Benchmark sweep ------------------------------ */
    FILE* csv_file = fopen(output, "w");
    if (!csv_file) {
        perror("Error creating the results file");
        return 1;
    }
    fprintf(csv_file, "size,method,threads,build,repetition,time_seconds,memory\n");

    int run_seq  = (strcmp(method, "all") == 0 || strcmp(method, "seq") == 0);
    int run_omp  = (strcmp(method, "all") == 0 || strcmp(method, "omp") == 0);
    int run_ikj  = (strcmp(method, "all") == 0 || strcmp(method, "ikj") == 0);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        char path_A[2048], path_B[2048];
        snprintf(path_A, sizeof(path_A), "%s/matrix_A_%d.csv", datadir, n);
        snprintf(path_B, sizeof(path_B), "%s/matrix_B_%d.csv", datadir, n);

        double** A = allocate_matrix(n);
        double** B = allocate_matrix(n);
        double** C = allocate_matrix(n);

        if (!read_csv(path_A, A, n) || !read_csv(path_B, B, n)) {
            printf("Input files not found for n=%d, skipping.\n", n);
            free_matrix(A, n); free_matrix(B, n); free_matrix(C, n);
            continue;
        }

        /* Reference checksum from one untimed sequential run. */
        multiply_seq(A, B, C, n);
        double ref_checksum = compute_checksum(C, n);

        if (run_seq) {
            for (int w = 0; w < warm_ups; w++) multiply_seq(A, B, C, n);
            for (int rep = 1; rep <= repetitions; rep++) {
                double t0 = get_time();
                multiply_seq(A, B, C, n);
                double dt = get_time() - t0;
                size_t mem = get_memory_usage();
                fprintf(csv_file, "%d,Sequential,1,%s,%d,%.6f,%zu\n", n, STR(BUILD_LABEL), rep, dt, mem);
                printf("Sequential | n=%d | Rep %d | Time: %.6f s | Memory: %zu KB | Checksum: %.4f\n",
                       n, rep, dt, mem, compute_checksum(C, n));
            }
        }

        if (run_omp) {
            for (int t = 0; t < num_threads_cfg; t++) {
                int nt = threads[t];
                for (int w = 0; w < warm_ups; w++) multiply_omp(A, B, C, n, nt);
                for (int rep = 1; rep <= repetitions; rep++) {
                    double t0 = get_time();
                    multiply_omp(A, B, C, n, nt);
                    double dt = get_time() - t0;
                    size_t mem = get_memory_usage();
                    fprintf(csv_file, "%d,OpenMP,%d,%s,%d,%.6f,%zu\n", n, nt, STR(BUILD_LABEL), rep, dt, mem);
                    printf("OpenMP(%2d threads) | n=%d | Rep %d | Time: %.6f s | Memory: %zu KB\n",
                           nt, n, rep, dt, mem);
                }
                double checksum = compute_checksum(C, n);
                int ok = fabs(checksum - ref_checksum) <= 1e-6 * fmax(1.0, fabs(ref_checksum));
                printf("Validation n=%d threads=%d: %s (checksum %.6f vs ref %.6f)\n\n",
                       n, nt, ok ? "OK" : "FAILED", checksum, ref_checksum);
            }
        }

        if (run_ikj) {
            for (int w = 0; w < warm_ups; w++) multiply_seq_ikj(A, B, C, n);
            for (int rep = 1; rep <= repetitions; rep++) {
                double t0 = get_time();
                multiply_seq_ikj(A, B, C, n);
                double dt = get_time() - t0;
                size_t mem = get_memory_usage();
                fprintf(csv_file, "%d,Sequential_ikj,1,%s,%d,%.6f,%zu\n", n, STR(BUILD_LABEL), rep, dt, mem);
                printf("Sequential_ikj | n=%d | Rep %d | Time: %.6f s | Memory: %zu KB\n",
                       n, rep, dt, mem);
            }
            double checksum = compute_checksum(C, n);
            int ok = fabs(checksum - ref_checksum) <= 1e-6 * fmax(1.0, fabs(ref_checksum));
            printf("Validation n=%d ikj: %s (checksum %.6f vs ref %.6f)\n\n",
                   n, ok ? "OK" : "FAILED", checksum, ref_checksum);
        }

        free_matrix(A, n);
        free_matrix(B, n);
        free_matrix(C, n);
    }

    fclose(csv_file);
    printf("Results written to %s\n", output);
    return 0;
}
