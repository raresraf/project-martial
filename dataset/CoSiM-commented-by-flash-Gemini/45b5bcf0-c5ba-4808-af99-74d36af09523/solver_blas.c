
#include "utils.h"
#include <string.h>
#include <cblas.h>

/**
 * @file solver_blas.c
 * @brief High-performance matrix solver utilizing Level 3 BLAS operations.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages standard BLAS library routines for 
 * efficient execution of cubic-order operations. It specifically uses 
 * `cblas_dtrmm` to exploit the triangular property of A and `cblas_dgemm` 
 * for generalized matrix multiplication with transposition.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Implementation of the matrix expression via BLAS calls.
 * 
 * Algorithm: Atomic algebraic computation steps.
 * 1. Compute T1 = A * B using in-place `cblas_dtrmm` (Left, Upper, NoTrans) on a copy of B.
 * 2. Compute P1 = T1 * B^T using `cblas_dgemm` (NoTrans, Trans).
 * 3. Compute P2 = A^T * A using in-place `cblas_dtrmm` (Left, Upper, Trans) on a copy of A.
 * 4. Sum P1 and P2 using optimized pointer-based addition.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	// Logic: Workspace and result matrix allocation.
	double *first_mul = calloc (N * N, sizeof(double));
	if (!first_mul)
		return NULL;

	double *second_mul = calloc (N * N, sizeof(double));
	if (!second_mul)
		return NULL;

	double *third_mul = calloc (N * N, sizeof(double));
	if (!third_mul)
		return NULL;

	double *result = malloc (N * N * sizeof(double));
	if (!result)
		return NULL;

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM.
	 * Optimization: Uses triangular properties of A to multiply B directly in workspace.
	 */
	memcpy(first_mul, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, 1.0, A, N, first_mul, N);

	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM with transposition.
	 * Logic: Multiplies the intermediate product by transposed B to generate P1.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1.0, first_mul, N, B, N, 1.0, second_mul, N);

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: Triangular symmetric product via DTRMM.
	 * Invariant: third_mul transformed from a copy of A into the symmetric product.
	 */
	memcpy(third_mul, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1.0, A, N, third_mul, N);

	register int i, j;

	/**
	 * Block Logic: Final summation Result = P1 + P2.
	 * Optimization: Pointer-based linear accumulation.
	 * Logic: Sums the two component matrix products using sequential pointer 
	 * increments to maximize cache hit rates.
	 */
	for (i = 0; i < N; i++) {
		register double *res = &result[i * N];
		register double *pa = &second_mul[i * N];
		register double *pb = &third_mul[i * N];

		for (j = 0; j < N; j++) {
			*res = *pa + *pb;
			res++;
			pa++;
			pb++;
		}
	}

	free(first_mul);
	free(second_mul);
	free(third_mul);
	return result;
}
