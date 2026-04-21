
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief High-performance matrix solver using Level 3 BLAS library routines.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages standardized BLAS operations for efficient 
 * cubic-order linear algebra. Specifically, it uses `cblas_dtrmm` for 
 * triangular multiplications and `cblas_dgemm` for general matrix multiplication 
 * with transposition and additive updates.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Implementation of the matrix expression via BLAS calls.
 * 
 * Algorithm: Optimized algebraic transformation sequence.
 * 1. Compute T1 = A * B using `cblas_dtrmm` (Left, Upper, NoTrans) on a copy of B.
 * 2. Compute P1 = A^T * A using in-place `cblas_dtrmm` (Left, Upper, Trans) on a copy of A.
 * 3. Compute Result = P1 + T1 * B^T using `cblas_dgemm` with transposition 
 *    and additive accumulation (beta=1.0).
 */
double *my_solver(int N, double *A, double *B) {
        double alpha, beta;
        int dimensionMatrix = N * N;
 
        // Logic: Allocation of workspace and result matrices.
        double *B2 = (double *)calloc(dimensionMatrix, sizeof(double));
        double *C = (double *)calloc(dimensionMatrix, sizeof(double));

        if (B2 == NULL || C == NULL) {
                return NULL;
        }

        alpha = 1.0;
        beta = 1.0;

        /**
         * Block Logic: Initial data duplication.
         * Invariant: C starts as a copy of A (for A^T * A), B2 as a copy of B (for A * B).
         */
        memcpy(C, A, dimensionMatrix * sizeof(double));
        memcpy(B2, B, dimensionMatrix * sizeof(double));
                
        /**
         * Block Logic: Compute (A * B).
         * Algorithm: DTRMM.
         * Optimization: Exploits upper triangularity of A to transform B2 in-place.
         */
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, N, N, alpha, A, N, B2, N);

        /**
         * Block Logic: Compute (A^T * A).
         * Algorithm: DTRMM with Transposition.
         * Logic: Performs the symmetric product of triangular A directly into workspace C.
         */
        cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                CblasNonUnit, N, N, alpha, A, N, C, N);

        /**
         * Block Logic: Final accumulation Result = C + (B2 * B^T).
         * Algorithm: DGEMM.
         * Invariant: Accumulates the product of the intermediate B2 and transposed B 
         * into C, which already holds the (A^T * A) term.
         */
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, alpha, B2, N, B, N, beta, C, N);


        free(B2);
	return C;
}
