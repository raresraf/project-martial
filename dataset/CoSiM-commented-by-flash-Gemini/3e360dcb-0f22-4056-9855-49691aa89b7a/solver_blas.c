
#include "utils.h"
#include "cblas.h"
#include <string.h>

/**
 * @file solver_blas.c
 * @brief Matrix solver implementation using standardized BLAS library routines.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A^T * A) + A * (B * B^T)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages Level 3 BLAS operations to achieve 
 * near-peak hardware performance. It uses `cblas_dtrmm` for triangular-aware 
 * multiplication and `cblas_dgemm` for high-throughput general matrix products.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Computes the complex matrix expression via sequence of BLAS calls.
 * 
 * Algorithm: Stage-based linear algebra calculation.
 * 1. Compute P1 = A^T * A using in-place `cblas_dtrmm` on a copy of A.
 * 2. Compute T1 = B * B^T using `cblas_dgemm` with transposition.
 * 3. Compute P2 = A * T1 using `cblas_dtrmm` (Left, Upper, NoTrans).
 * 4. Sum P1 and P2 via element-wise addition into the result matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double *C;
	double *AtA, *BBt, *ABBt;
	int i, j;

	// Logic: Allocation of result and workspace buffers.
	C = calloc(N * N, sizeof(*C));
	AtA = calloc(N * N, sizeof(*AtA));
	BBt = calloc(N * N, sizeof(*BBt));
	ABBt = calloc(N * N, sizeof(*ABBt));

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: In-place DTRMM.
	 * Logic: Exploys A's upper triangularity to compute the Gramian matrix efficiently.
	 */
	memcpy(AtA, A, N * N * sizeof(*C));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, AtA, N);

	/**
	 * Block Logic: Compute (B * B^T).
	 * Algorithm: DGEMM.
	 * Logic: Multiplies B by its own transpose to create a symmetric intermediate.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 1.0, BBt, N);

	/**
	 * Block Logic: Compute A * (B * B^T).
	 * Algorithm: DTRMM.
	 * Invariant: ABBt initially holds the product (B * B^T), which is then 
	 * left-multiplied by the triangular matrix A.
	 */
	memcpy(ABBt, BBt, N * N * sizeof(*ABBt));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, ABBt, N);

	/**
	 * Block Logic: Component aggregation.
	 * Logic: Final pass to sum the two major terms of the expression.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += AtA[i * N + j] + ABBt[i * N + j];
		}
	}
	
	free(AtA);
	free(BBt);
	free(ABBt);
	return C;
}
