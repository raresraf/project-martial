/**
 * @5d34cc99-3606-4890-b2d1-11013fcf314a/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 *
 * This implementation solves the matrix expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Stage-based linear algebra calculation.
 * 1. AB = A * B using cblas_dtrmm.
 * 2. ABBt = AB * B^T using cblas_dgemm.
 * 3. AtA = A^T * A using cblas_dtrmm (leading transposition).
 * 4. Sum Result = ABBt + AtA.
 *
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

#include <string.h>
#include <cblas.h>
#include "utils.h"

/**
 * Functional Utility: Orchestrates workspace memory allocation for intermediate matrix products.
 */
void allocate_matrix(int N, double **AB, double **ABBt,
                        double **AtA, double **C) {

    *AB = calloc (N * N, sizeof (**AB));
    *ABBt = calloc (N * N, sizeof (double));
    *AtA = calloc (N * N, sizeof (**AtA));
    *C = calloc (N * N, sizeof (**C));
}

/**
 * @brief Computes the matrix expression via optimized BLAS routines.
 * Functional Utility: Leverages hardware-accelerated Level 3 BLAS to perform high-density
 * matrix arithmetic with minimal memory movement.
 */
double *my_solver(int N, double *A, double *B) {
    double *AB, *ABBt, *AtA, *C;
    register int i, j;

    allocate_matrix(N, &AB, &ABBt, &AtA, &C);

    /**
     * Block Logic: Compute AB = A * B.
     * Optimization: Uses cblas_dtrmm to exploit the upper triangular property of matrix A,
     * significantly reducing the number of floating-point operations compared to dense GEMM.
     */
    memcpy(AB, B, N * N * sizeof(*AB));
    cblas_dtrmm(CblasRowMajor,
                CblasLeft,
                CblasUpper,
                CblasNoTrans,
                CblasNonUnit,
                N, N, 1.0,
                A, N,
                AB, N);

    /**
     * Block Logic: Compute ABBt = AB * B^T.
     * Functional Utility: Computes the outer product contribution using general matrix 
     * multiplication with implicit transposition on the right-hand operand.
     */
    memcpy(ABBt, AB, N * N * sizeof(*AB));
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                N, N, N, 1.0,
                AB, N,
                B, N,
                0.0, ABBt, N);

    /**
     * Block Logic: Compute AtA = A^T * A.
     * Optimization: Efficiently computes the Gramian matrix for A in-place using dtrmm 
     * with leading transposition.
     */
    memcpy(AtA, A, N * N * sizeof(*A));
    cblas_dtrmm(CblasRowMajor,
                CblasLeft,
                CblasUpper,
                CblasTrans,
                CblasNonUnit,
                N, N, 1.0,
                A, N,
                AtA, N);

    /**
     * Block Logic: Final summation pass.
     * Invariant: Aggregates the results of the two primary compute stages into the output buffer C.
     */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[i * N + j] = AtA[i * N + j] + ABBt[i * N + j];
        }
    }

    free(AB);
    free(ABBt);
    free(AtA);

    return C;
}
