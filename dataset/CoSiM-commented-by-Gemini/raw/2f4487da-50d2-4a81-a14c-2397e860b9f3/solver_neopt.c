/**
 * @file solver_neopt.c
 * @brief Naive non-optimized matrix solver implementation.
 * Unoptimized memory access pattern. Baseline for performance comparison.
 */

#include "utils.h"

/**
 * @brief Computes C = At * A + A * B * Bt.
 * Allocates memory dynamically and executes matrix operations.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double* B) {
	printf("NEOPT SOLVER\n");

    double *C = (double *)malloc(N * N * sizeof(double));
    double *D = (double *)malloc(N * N * sizeof(double));
    int i, j, k;

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i++) {
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for (j = i; j < N; j++) {
            C[i * N + j] = 0;
            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for (k = 0; k < N; k++) {
                C[i * N + j] += B[i * N + k] * B[j * N + k];
            }

            /* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
            if (i != j)
                C[j * N + i] = C[i * N + j];
        }
    }

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i++) {
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for (j = 0; j < N; j++) {
            D[i * N + j] = 0;
            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for (k = i; k < N; k++) {
                D[i * N + j] += A[i * N + k] * C[k * N + j];
            }
        }
    }

    
    /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
    for (i = 0; i < N; i++) {
        /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
        for (j = i; j < N; j++) {
            C[i * N + j] = 0;
            /* @pre Loop bounds initialized. @invariant Iterates over assigned memory blocks, preserving data locality where possible. */
            for (k = 0; k <= i; k++) {
                C[i * N + j] += A[k * N + i] * A[k * N + j];
            }

            /* @pre Conditional evaluation. @invariant Taken branch maintains control flow invariants. */
            if (i != j) {
                D[j * N + i] += C[i * N + j];
            }

            D[i * N + j] += C[i * N + j];
        }
    }

    free(C);
	return D;
}
