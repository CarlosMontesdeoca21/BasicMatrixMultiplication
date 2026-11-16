#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef BUILD_LABEL
#define BUILD_LABEL O3
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define MAX_LIST 32
#define MAX_REPS 64

size_t get_memory_usage(void) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / 1024;
    }
    return 0;
}

static double get_time(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER cnt;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

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

static void multiply_seq_block(const double* A, const double* B, double* C, int n, int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
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

int read_csv_block(const char* filename, double* M, int n) {
    FILE* file = fopen(filename, "r");
    if (!file) return 0;

    char line[65536];
    int row = 0;
    while (fgets(line, sizeof(line), file) && row < n) {
        int col = 0;
        char* token = strtok(line, ",");
        while (token && col < n) {
            M[row * n + col] = atof(token);
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

double compute_checksum_block(const double* C, int n) {
    double sum = 0.0;
    long long total = (long long)n * n;
    for (long long i = 0; i < total; i++) {
        sum += C[i];
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

double max_abs_diff_block(const double* X, const double* Y, long long total) {
    double m = 0.0;
    for (long long i = 0; i < total; i++) {
        double d = fabs(X[i] - Y[i]);
        if (d > m) m = d;
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

static char g_default_datadir[1024];
static char g_default_output[1024];

static void resolve_default_paths(void) {
    char exe_path[512];
    DWORD len = GetModuleFileNameA(NULL, exe_path, sizeof(exe_path));
    if (len == 0 || len >= sizeof(exe_path)) {
        snprintf(g_default_datadir, sizeof(g_default_datadir), "Activity4/data");
        snprintf(g_default_output, sizeof(g_default_output), "Activity4/data/results_act4.csv");
        return;
    }
    char* last_slash = strrchr(exe_path, '\\');
    if (last_slash) *last_slash = '\0';
    last_slash = strrchr(exe_path, '\\');
    if (last_slash) *last_slash = '\0';
    last_slash = strrchr(exe_path, '\\');
    if (last_slash) *last_slash = '\0';
    snprintf(g_default_datadir, sizeof(g_default_datadir), "%s\\data", exe_path);
    snprintf(g_default_output, sizeof(g_default_output), "%s\\data\\results_act4.csv", exe_path);
}

static void compute_row_blocks(int n, int p, int* counts, int* displs) {
    int base = n / p;
    int rem = n % p;
    int acc = 0;
    for (int r = 0; r < p; r++) {
        int rows = base + (r < rem ? 1 : 0);
        counts[r] = rows * n;
        displs[r] = acc;
        acc += rows * n;
    }
}

static int run_selftest(int rank, int size) {
    int fails = 0;
    int ns[] = {8, 10};
    int* counts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));

    for (int si = 0; si < 2; si++) {
        int n = ns[si];
        long long total = (long long)n * n;
        compute_row_blocks(n, size, counts, displs);
        int rows = counts[rank] / n;

        double* A = (rank == 0) ? (double*)malloc(total * sizeof(double)) : NULL;
        double* ref = (double*)malloc(total * sizeof(double));
        double* left = (double*)malloc(total * sizeof(double));
        double* right = (double*)malloc(total * sizeof(double));
        double* a_loc = (double*)malloc((size_t)counts[rank] * sizeof(double));
        double* c_loc = (double*)malloc((size_t)counts[rank] * sizeof(double));
        double* c_full = (rank == 0) ? (double*)malloc(total * sizeof(double)) : NULL;

        if (rank == 0) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    A[i * n + j] = ((i * 7 + j * 13) % 19) / 19.0;
                }
            }
        }

        const char* names[] = {"I x A == A", "A x I == A", "0 x A == 0"};
        for (int c = 0; c < 3; c++) {
            if (rank == 0) {
                if (c == 0) {
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            left[i * n + j] = (i == j) ? 1.0 : 0.0;
                    memcpy(right, A, total * sizeof(double));
                    memcpy(ref, A, total * sizeof(double));
                } else if (c == 1) {
                    memcpy(left, A, total * sizeof(double));
                    for (int i = 0; i < n; i++)
                        for (int j = 0; j < n; j++)
                            right[i * n + j] = (i == j) ? 1.0 : 0.0;
                    memcpy(ref, A, total * sizeof(double));
                } else {
                    memset(left, 0, total * sizeof(double));
                    memcpy(right, A, total * sizeof(double));
                    memset(ref, 0, total * sizeof(double));
                }
            }

            MPI_Scatterv(rank == 0 ? left : NULL, counts, displs, MPI_DOUBLE,
                         a_loc, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(right, (int)total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            multiply_seq_block(a_loc, right, c_loc, n, rows);
            MPI_Gatherv(c_loc, counts[rank], MPI_DOUBLE, c_full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double diff = max_abs_diff_block(c_full, ref, total);
                int ok = diff <= 1e-15;
                printf("Selftest n=%d p=%d | %-12s | max diff %.3e | %s\n",
                       n, size, names[c], diff, ok ? "PASS" : "FAIL");
                if (!ok) fails++;
            }
        }

        free(ref); free(left); free(right);
        free(a_loc); free(c_loc); free(c_full);
        free(A);
    }

    free(counts); free(displs);
    return fails;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    setvbuf(stdout, NULL, _IONBF, 0);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int sizes[MAX_LIST];
    int num_sizes = 0;
    int warm_ups = 2;
    int repetitions = 5;
    int validate_n = -1;
    int omp_threads = 16;
    int selftest = 0;
    const char* method = "all";

    resolve_default_paths();
    const char* datadir = g_default_datadir;
    const char* output = g_default_output;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            num_sizes = parse_int_list(argv[++i], sizes, MAX_LIST);
        } else if (strcmp(argv[i], "--warmups") == 0 && i + 1 < argc) {
            warm_ups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            repetitions = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            method = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            omp_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--datadir") == 0 && i + 1 < argc) {
            datadir = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--validate") == 0 && i + 1 < argc) {
            validate_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--selftest") == 0) {
            selftest = 1;
        } else {
            if (rank == 0) {
                fprintf(stderr,
                    "Usage: %s [--sizes 10,100,1000,2000] [--method all|seq|omp|mpi]\n"
                    "          [--threads 16] [--warmups 2] [--reps 5]\n"
                    "          [--datadir DIR] [--output FILE] [--validate N] [--selftest]\n",
                    argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (num_sizes == 0) {
        int defaults[] = {10, 100, 1000, 2000};
        num_sizes = 4;
        memcpy(sizes, defaults, sizeof(defaults));
    }

    omp_set_dynamic(0);

    if (rank == 0) {
        printf("=== DISTRIBUTED MATRIX MULTIPLICATION BENCHMARK (MPI) ===\n");
        printf("Build label: %s | MPI processes: %d | max OpenMP threads: %d\n",
               STR(BUILD_LABEL), world, omp_get_max_threads());
        printf("Data directory: %s\n", datadir);
        printf("Output file: %s\n", output);
        printf("Warm-ups: %d | Measured repetitions: %d\n\n", warm_ups, repetitions);
    }

    if (selftest) {
        int fails = run_selftest(rank, world);
        if (rank == 0) {
            printf("\nSelftest: %s (%d failure(s))\n", fails == 0 ? "ALL PASS" : "FAILED", fails);
        }
        MPI_Finalize();
        return fails == 0 ? 0 : 1;
    }

    int run_seq = (strcmp(method, "all") == 0 || strcmp(method, "seq") == 0);
    int run_omp = (strcmp(method, "all") == 0 || strcmp(method, "omp") == 0);
    int run_mpi = (strcmp(method, "all") == 0 || strcmp(method, "mpi") == 0);

    if (!run_seq && !run_omp && !run_mpi) {
        if (rank == 0) fprintf(stderr, "Invalid --method: %s\n", method);
        MPI_Finalize();
        return 1;
    }

    if ((run_seq || run_omp) && world > 1) {
        if (rank == 0) {
            fprintf(stderr, "Sequential/OpenMP methods must run with mpiexec -n 1\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (validate_n > 0) {
        if (world != 1) {
            if (rank == 0) fprintf(stderr, "--validate must run with mpiexec -n 1\n");
            MPI_Finalize();
            return 1;
        }
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
        multiply_omp(A, B, C2, validate_n, omp_threads);

        if (!write_matrix_csv(path_out, C, validate_n)) {
            printf("Validate: could not write %s\n", path_out);
            return 1;
        }

        double diff = max_abs_diff(C, C2, validate_n);
        printf("Validation n=%d | sequential checksum: %.6f\n", validate_n, compute_checksum(C, validate_n));
        printf("Max |sequential - openmp(%d threads)| elementwise diff: %.3e\n", omp_threads, diff);
        printf("Result matrix written to %s\n", path_out);

        free_matrix(A, validate_n); free_matrix(B, validate_n);
        free_matrix(C, validate_n); free_matrix(C2, validate_n);
        MPI_Finalize();
        return 0;
    }

    FILE* csv_file = NULL;
    if (rank == 0) {
        csv_file = fopen(output, "a");
        if (!csv_file) {
            perror("Error opening the results file");
            MPI_Finalize();
            return 1;
        }
        fseek(csv_file, 0, SEEK_END);
        if (ftell(csv_file) == 0) {
            fprintf(csv_file, "size,method,workers,threads,build,repetition,"
                              "time_e2e,time_transfer,time_compute,time_gather,"
                              "memory_kb,max_diff\n");
        }
    }

    int* counts = (int*)malloc(world * sizeof(int));
    int* displs = (int*)malloc(world * sizeof(int));

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        long long total = (long long)n * n;
        char path_A[2048], path_B[2048];
        snprintf(path_A, sizeof(path_A), "%s/matrix_A_%d.csv", datadir, n);
        snprintf(path_B, sizeof(path_B), "%s/matrix_B_%d.csv", datadir, n);

        if (run_seq || run_omp) {
            double** A = allocate_matrix(n);
            double** B = allocate_matrix(n);
            double** C = allocate_matrix(n);
            double** C_ref = allocate_matrix(n);

            if (!read_csv(path_A, A, n) || !read_csv(path_B, B, n)) {
                printf("Input files not found for n=%d, skipping.\n", n);
                free_matrix(A, n); free_matrix(B, n); free_matrix(C, n); free_matrix(C_ref, n);
                continue;
            }

            multiply_seq(A, B, C_ref, n);

            if (run_seq) {
                double t_e2e[MAX_REPS];
                for (int w = 0; w < warm_ups; w++) multiply_seq(A, B, C, n);
                for (int rep = 0; rep < repetitions; rep++) {
                    double t0 = get_time();
                    multiply_seq(A, B, C, n);
                    t_e2e[rep] = get_time() - t0;
                }
                double diff = max_abs_diff(C, C_ref, n);
                for (int rep = 1; rep <= repetitions; rep++) {
                    size_t mem = get_memory_usage();
                    fprintf(csv_file, "%d,Sequential,1,1,%s,%d,%.6f,0,%.6f,0,%zu,%.3e\n",
                            n, STR(BUILD_LABEL), rep, t_e2e[rep - 1], t_e2e[rep - 1], mem, diff);
                    printf("Sequential | n=%d | Rep %d | Time: %.6f s | Memory: %zu KB\n",
                           n, rep, t_e2e[rep - 1], mem);
                }
            }

            if (run_omp) {
                double t_e2e[MAX_REPS];
                for (int w = 0; w < warm_ups; w++) multiply_omp(A, B, C, n, omp_threads);
                for (int rep = 0; rep < repetitions; rep++) {
                    double t0 = get_time();
                    multiply_omp(A, B, C, n, omp_threads);
                    t_e2e[rep] = get_time() - t0;
                }
                double diff = max_abs_diff(C, C_ref, n);
                int ok = fabs(compute_checksum(C, n) - compute_checksum(C_ref, n)) <=
                         1e-6 * fmax(1.0, fabs(compute_checksum(C_ref, n)));
                for (int rep = 1; rep <= repetitions; rep++) {
                    size_t mem = get_memory_usage();
                    fprintf(csv_file, "%d,OpenMP,1,%d,%s,%d,%.6f,0,%.6f,0,%zu,%.3e\n",
                            n, omp_threads, STR(BUILD_LABEL), rep, t_e2e[rep - 1], t_e2e[rep - 1], mem, diff);
                    printf("OpenMP(%2d threads) | n=%d | Rep %d | Time: %.6f s | Memory: %zu KB\n",
                           omp_threads, n, rep, t_e2e[rep - 1], mem);
                }
                printf("Validation n=%d omp: %s (max diff %.3e)\n\n",
                       n, ok && diff <= 1e-6 ? "OK" : "FAILED", diff);
            }

            free_matrix(A, n); free_matrix(B, n);
            free_matrix(C, n); free_matrix(C_ref, n);
            if (!run_mpi) continue;
        }

        compute_row_blocks(n, world, counts, displs);
        int rows = counts[rank] / n;

        double* A_blk = (rank == 0) ? (double*)malloc(total * sizeof(double)) : NULL;
        double* B_blk = (rank == 0) ? (double*)malloc(total * sizeof(double)) : NULL;
        double* C_full = (rank == 0) ? (double*)malloc(total * sizeof(double)) : NULL;
        double* C_ref = (rank == 0) ? (double*)malloc(total * sizeof(double)) : NULL;
        double* a_loc = (double*)malloc((size_t)counts[rank] * sizeof(double));
        double* c_loc = (double*)malloc((size_t)counts[rank] * sizeof(double));
        double* b_buf = (double*)malloc(total * sizeof(double));
        long long* mem_all = (rank == 0) ? (long long*)malloc(world * sizeof(long long)) : NULL;

        int files_ok = 1;
        if (rank == 0) {
            files_ok = read_csv_block(path_A, A_blk, n) && read_csv_block(path_B, B_blk, n);
            if (!files_ok) printf("Input files not found for n=%d, skipping.\n", n);
        }
        MPI_Bcast(&files_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!files_ok) {
            free(A_blk); free(B_blk); free(C_full); free(C_ref);
            free(a_loc); free(c_loc); free(b_buf); free(mem_all);
            continue;
        }
        if (rank == 0) {
            memcpy(b_buf, B_blk, total * sizeof(double));
            multiply_seq_block(A_blk, B_blk, C_ref, n, n);
            printf("MPI reference computed for n=%d (sequential checksum %.6f)\n\n",
                   n, compute_checksum_block(C_ref, n));
        }

        double t_transfer[MAX_REPS], t_compute[MAX_REPS], t_gather[MAX_REPS], t_e2e[MAX_REPS];
        long long mem_max[MAX_REPS];

        for (int w = 0; w < warm_ups; w++) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Scatterv(A_blk, counts, displs, MPI_DOUBLE, a_loc, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(b_buf, (int)total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            multiply_seq_block(a_loc, b_buf, c_loc, n, rows);
            MPI_Gatherv(c_loc, counts[rank], MPI_DOUBLE, C_full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        for (int rep = 0; rep < repetitions; rep++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = get_time();
            MPI_Scatterv(A_blk, counts, displs, MPI_DOUBLE, a_loc, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(b_buf, (int)total, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            double t1 = get_time();
            multiply_seq_block(a_loc, b_buf, c_loc, n, rows);
            double t2 = get_time();
            MPI_Gatherv(c_loc, counts[rank], MPI_DOUBLE, C_full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            double t3 = get_time();

            size_t mem = get_memory_usage();
            MPI_Gather(&mem, 1, MPI_LONG_LONG, mem_all, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

            t_transfer[rep] = t1 - t0;
            t_compute[rep] = t2 - t1;
            t_gather[rep] = t3 - t2;
            t_e2e[rep] = t3 - t0;
            if (rank == 0) {
                long long mx = 0;
                for (int r = 0; r < world; r++) {
                    if (mem_all[r] > mx) mx = mem_all[r];
                }
                mem_max[rep] = mx;
            }
        }

        if (rank == 0) {
            double diff = max_abs_diff_block(C_full, C_ref, total);
            double ck_ref = compute_checksum_block(C_ref, n);
            double ck = compute_checksum_block(C_full, n);
            int ok = fabs(ck - ck_ref) <= 1e-6 * fmax(1.0, fabs(ck_ref));
            for (int rep = 1; rep <= repetitions; rep++) {
                fprintf(csv_file, "%d,MPI,%d,0,%s,%d,%.6f,%.6f,%.6f,%.6f,%lld,%.3e\n",
                        n, world, STR(BUILD_LABEL), rep,
                        t_e2e[rep - 1], t_transfer[rep - 1], t_compute[rep - 1], t_gather[rep - 1],
                        mem_max[rep - 1], diff);
                printf("MPI(%2d procs) | n=%d | Rep %d | e2e: %.6f s (transfer %.6f + compute %.6f + gather %.6f) | Mem(max): %lld KB\n",
                       world, n, rep, t_e2e[rep - 1], t_transfer[rep - 1],
                       t_compute[rep - 1], t_gather[rep - 1], mem_max[rep - 1]);
            }
            printf("Validation n=%d procs=%d: %s (checksum %.6f vs ref %.6f, max diff %.3e)\n\n",
                   n, world, ok && diff <= 1e-6 ? "OK" : "FAILED", ck, ck_ref, diff);
        }

        free(A_blk); free(B_blk); free(C_full); free(C_ref);
        free(a_loc); free(c_loc); free(b_buf); free(mem_all);
    }

    free(counts); free(displs);
    if (rank == 0) fclose(csv_file);
    if (rank == 0) printf("Results appended to %s\n", output);

    MPI_Finalize();
    return 0;
}
