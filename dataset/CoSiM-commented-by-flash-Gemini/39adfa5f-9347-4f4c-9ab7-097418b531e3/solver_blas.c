
#include "utils.h"
#include "cblas.h"
#include <string.h>

/**
 * @file solver_blas.c
 * @brief Optimized matrix solver using standard BLAS (Basic Linear Algebra Subprograms).
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = A * (B * B^T) + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs Level 3 BLAS routines for cubic-order operations. 
 * Specifically, it uses `cblas_dgemm` for general multiplication, `cblas_dtrmm` 
 * to exploit A's triangularity, and `cblas_daxpy` for optimized element-wise 
 * vector addition.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Computes the complex matrix expression via sequence of BLAS calls.
 * 
 * Algorithm: Optimized algebraic transformation.
 * 1. Compute T = B * B^T using `cblas_dgemm` with transposition.
 * 2. Compute P1 = A * T using `cblas_dtrmm` (Left, Upper, NoTrans).
 * 3. Compute P2 = A^T * A using `cblas_dtrmm` (Left, Upper, Trans) on a copy of A.
 * 4. Sum the components P1 and P2 using `cblas_daxpy`.
 */
double* my_solver(int N, double *A, double *B) {
    // Logic: Workspace allocation for intermediate and final results.
    double *partial_result1 = (double *)malloc(N * N * sizeof(double));
    double *partial_result2 = (double *)malloc(N * N * sizeof(double));

    /**
     * Block Logic: Compute (B * B^T).
     * Algorithm: DGEMM (General Matrix Multiply).
     * Logic: Multiply B by its transpose to generate a symmetric intermediate matrix.
     */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1.0, B, N, B, N, 0, partial_result1, N);

    /**
     * Block Logic: Compute A * (B * B^T).
     * Algorithm: DTRMM (Triangular Matrix Multiply).
     * Optimization: Exploys the fact that A is upper triangular to reduce 
     * total operations by approximately half compared to general multiplication.
     */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N, 1.0, A, N, partial_result1, N);

    /**
     * Block Logic: Compute (A^T * A).
     * Algorithm: DTRMM with transposition.
     * Invariant: partial_result2 initially holds a copy of A, then is 
     * transformed in-place by the triangular product with A^T.
     */
    memcpy(partial_result2, A, N * N * sizeof(double));
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, partial_result2, N);

    /**
     * Block Logic: Final summation (P1 + P2).
     * Algorithm: DAXPY (Scalar * Vector + Vector).
     * Optimization: Treats matrices as linear buffers for high-throughput 
     * vector addition.
     */
    cblas_daxpy(N * N, 1.0, partial_result1, 1, partial_result2, 1);

    free(partial_result1);
	return partial_result2;
}
