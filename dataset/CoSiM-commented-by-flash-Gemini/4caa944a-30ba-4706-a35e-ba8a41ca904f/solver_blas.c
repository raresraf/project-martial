
#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * @file solver_blas.c
 * @brief High-performance matrix expression solver utilizing Level 3 BLAS routines.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages standardized BLAS operations for 
 * efficient execution. It specifically uses `cblas_dtrmm` for triangular-aware 
 * multiplications, `cblas_dcopy` for fast memory duplication, and `cblas_dgemm` 
 * for generalized matrix-matrix multiplication with transposition and 
 * additive updates.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Implementation of the matrix expression via BLAS sequence.
 * 
 * Algorithm: Atomic algebraic stages.
 * 1. Compute T1 = A * B using in-place `cblas_dtrmm` on a copy of B.
 * 2. Compute P1 = A^T * A using in-place `cblas_dtrmm` on a copy of A.
 * 3. Compute Result = P1 + T1 * B^T using `cblas_dgemm` with transposition 
 *    and an additive beta (1.0).
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C, *AtA, *AB;

	// Logic: Pre-allocation of result and intermediate workspace matrices.
	C = calloc(N * N, sizeof(double));
	DIE(C == NULL, "calloc C");

	AtA = calloc(N * N, sizeof(double));
	DIE(AtA == NULL, "calloc AtA");

	AB = calloc(N * N, sizeof(double));
	DIE(AB == NULL, "calloc AB");

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: Triangular Matrix Multiply (DTRMM).
	 * Optimization: Exploits upper triangularity of A to transform the copy 
	 * of B ('AB') in-place.
	 */
	cblas_dcopy(N * N, B, 1, AB, 1);
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0,
		A, N,
		AB, N
	);

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: In-place DTRMM.
	 * Logic: Computes the symmetric product of the triangular matrix A directly 
	 * into the 'AtA' workspace.
	 */
	cblas_dcopy(N * N, A, 1, AtA, 1);
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0,
		A, N,
		AtA, N
	);

	/**
	 * Block Logic: Final accumulation Result = AtA + (AB * B^T).
	 * Algorithm: DGEMM with additive accumulation.
	 * Invariant: 'C' is initialized with 'AtA'; the subsequent general 
	 * multiplication adds its result directly into 'C' using beta=1.0.
	 */
	cblas_dcopy(N * N, AtA, 1, C, 1);
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
        N, N, N,
		1.00,
		AB, N,
		B, N,
		1.00,
		C, N);

	free(AB);
    free(AtA);
	return C;
}
