/**
 * @file solver_blas.c
 * @brief Matrix expression solver using high-performance BLAS routines.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Stage-based implementation using Level 3 BLAS.
 * 1. AB = A * B via cblas_dtrmm (triangular-general product).
 * 2. A^T * A via cblas_dtrmm (symmetric product).
 * 3. Result = AB * B^T + (A^T * A) via cblas_dgemm.
 *
 * Domain: HPC, Linear Algebra.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * my_solver - Implementation of the matrix expression via BLAS calls.
 */
double* my_solver(int N, double *A, double *B) {

	/**
	 * Pre-condition: Allocation of temporary and result buffers.
	 */
	double *AB = calloc(N * N, sizeof(double));
	if(AB == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	double *C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	double *A_TA = calloc(N * N, sizeof(double));
	if(A_TA == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	/**
	 * Invariant: Duplicates input data for in-place transformation.
	 */
	memcpy(AB, B, N * N * sizeof(double));
	memcpy(A_TA, A, N * N * sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Exploits A's upper triangularity with DTRMM.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AB, N
	);

	/**
	 * Block Logic: Compute A^T * A.
	 * Functional Utility: Calculates the Gramian component of the expression.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		A_TA, N
	);

	memcpy(C, A_TA, N * N * sizeof(double));

	/**
	 * Block Logic: Compute C = 1.0 * (AB * B^T) + 1.0 * C.
	 * Functional Utility: Fuses the second product and final addition.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1,
		AB, N,
		B, N,
		1, C, N);

	free(A_TA);
	free(AB);

	return C;
}
