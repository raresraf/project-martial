/**
 * @5d34cc99-3606-4890-b2d1-11013fcf314a/solver_opt.c
 * @brief Manually optimized implementation of a matrix expression solver.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using register-level 
 * optimizations and local sum accumulators to minimize memory store operations.
 * Domain: HPC Performance Tuning.
 */

#define MIN(a,b) (((a)<(b))?(a):(b))

#include "utils.h"

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
 * @brief Computes the matrix expression with basic compiler-oriented optimizations.
 * Optimization: Uses `register` hints and local accumulation to promote scalar 
 * promotion and reduce pointer aliasing overhead.
 */
double* my_solver(int N, double *A, double* B) {

    double *AB, *ABBt, *AtA, *C;
    register int i, j, k;

    allocate_matrix(N, &AB, &ABBt, &AtA, &C);

    /**
     * Block Logic: Compute AB = A * B.
     * Optimization: Employs a local `sum` register to accumulate dot products, 
     * significantly reducing writes to the `AB` buffer in the inner k-loop.
     * Invariant: k starts at i, leveraging matrix A's upper triangular structure.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            register double sum = 0.0;
            for (k = i; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            AB[i * N + j] = sum;
        }
    }

    /**
     * Block Logic: Compute ABBt = AB * B^T.
     * Optimization: Register-based dot product accumulation. 
     * Access Pattern: B is accessed with j and k swapped to effect a virtual transposition.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            register double sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += AB[i * N + k] * B[j * N + k];
            }
            ABBt[i * N + j] = sum;
        }
    }

    /**
     * Block Logic: Compute AtA = A^T * A.
     * Optimization: Local register accumulation for Gramian matrix components.
     * Invariant: Limits k-iteration to MIN(i, j) based on upper triangular sparsity.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double register sum = 0;
            for (k = 0; k <= MIN(i, j); k++) {
                sum += A[k * N + i] * A[k * N + j];
            }
            AtA[i * N + j] = sum;
        }
    }

    /**
     * Block Logic: Final summation pass to consolidate computing stages.
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
