/**
 * @5d34cc99-3606-4890-b2d1-11013fcf314a/solver_neopt.c
 * @brief Naive implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using unoptimized 
 * triple-nested loops. Serves as a baseline for performance comparisons.
 * Domain: HPC Numerical Baselines.
 */

#define MIN(a,b) (((a)<(b))?(a):(b))

#include "utils.h"
#include <string.h>

/**
 * Functional Utility: Workspace initialization for intermediate matrix products.
 */
void allocate_matrix(int N, double **AB, double **ABBt,
                        double **AtA, double **C) {

    *AB = calloc (N * N, sizeof (double));
    *ABBt = calloc (N * N, sizeof (double));
    *AtA = calloc (N * N, sizeof (double));
    *C = calloc (N * N, sizeof (double));
}

/**
 * @brief Computes the matrix expression via standard iterative loops.
 * Algorithm: Naive $O(N^3)$ matrix multiplication stages.
 */
double *my_solver(int N, double *A, double* B) {

    double *AB, *ABBt, *AtA, *C;
    int i, j, k;

    allocate_matrix(N, &AB, &ABBt, &AtA, &C);

    /**
     * Block Logic: Compute AB = A * B where A is Upper Triangular.
     * Invariant: k starts at i to exploit the fact that A[i][k] = 0 for k < i.
     * Time Complexity: $O(N^3)$ with constant factor optimization for triangularity.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = i; k < N; k++) {
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    /**
     * Block Logic: Compute ABBt = AB * B^T.
     * Logic: Standard matrix multiplication where the second operand B is accessed 
     * in column-major order to simulate transposition (B[j][k] instead of B[k][j]).
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                ABBt[i * N + j] += AB[i * N + k] * B[j * N + k];
            }
        }
    }

    /**
     * Block Logic: Compute AtA = A^T * A.
     * Logic: Accumulates products of elements from A's columns.
     * Invariant: Exploits the upper triangular property where A[k][i] is non-zero only if k <= i.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k <= MIN(i, j); k++) {
                AtA[i * N + j] += A[k * N + i] * A[k * N + j];
            }
        }
    }

    /**
     * Block Logic: Accumulates partial matrix results into the final output matrix C.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
        }
    }

    free(AB);
    free(ABBt);
    free(AtA);

    return C;
}
