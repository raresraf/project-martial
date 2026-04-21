
#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief High-performance matrix solver using BLAS Level 3 operations.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages industry-standard BLAS implementations 
 * for cubic-order operations. Specifically, it uses `cblas_dtrmm` to exploit 
 * A's triangularity, `cblas_dcopy` for efficient memory duplication, 
 * and `cblas_dgemm` for general matrix multiplication with transposition.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

#define ONE 1.0

/**
 * my_solver - Computes the complex matrix expression via BLAS sequence.
 * 
 * Algorithm: Atomic algebraic stages.
 * 1. Compute P1 = A^T * A using in-place `cblas_dtrmm` on a copy of A.
 * 2. Compute T1 = A * B using `cblas_dtrmm` (Left, Upper, NoTrans) on a copy of B.
 * 3. Compute Result = P1 + T1 * B^T using `cblas_dgemm` with transposition 
 *    and additive update (beta=1.0).
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C, *AB, *AA;

	// Logic: Pre-allocation of workspace and result matrices.
	C = (double *)calloc(sizeof(double), N * N);
	AB = (double *)calloc(sizeof(double), N * N);
	AA = (double *)calloc(sizeof(double), N * N);

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: In-place DTRMM.
	 * Logic: Transforms the upper triangular matrix A into the symmetric 
	 * product A^T * A.
	 */
	cblas_dcopy(N * N, A, 1, AA, 1);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, ONE, A, N, AA, N);

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM.
	 * Optimization: Exploits A's triangularity to perform the multiplication 
	 * into a copy of B.
	 */
	cblas_dcopy(N * N, B, 1, AB, 1);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, ONE, A, N, AB, N);

	/**
	 * Block Logic: Final accumulation Result = AA + (AB * B^T).
	 * Algorithm: DGEMM with additive beta.
	 * Invariant: C starts with the value of AA, then accumulates the 
	 * second product term in-place.
	 */
	cblas_dcopy(N * N, AA, 1, C, 1);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, ONE, AB, N, B,
		N, ONE, C, N);

	free(AB);
	free(AA);
	return C;
}
