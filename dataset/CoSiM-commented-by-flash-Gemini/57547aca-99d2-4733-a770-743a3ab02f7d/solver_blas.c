/**
 * @file solver_blas.c
 * @brief Matrix expression solver utilizing high-performance BLAS routines.
 *
 * This module computes the expression: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Stage-based linear algebra calculation.
 * 1. P1 = A * B using cblas_dgemm.
 * 2. P2 = P1 * B^T using cblas_dgemm with transposition.
 * 3. P3 = A^T * A using cblas_dgemm with leading transposition.
 * 4. Accumulate result P2 + P3.
 *
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

#include "utils.h"
#include <cblas.h>

/**
 * my_solver - Implementation of the matrix expression via optimized BLAS calls.
 */
double* my_solver(int N, double *A, double *B) {
	int i = 0;
	int j = 0;

	/**
	 * Pre-condition: Allocation of temporary buffers for intermediate products.
	 */
	double *result1 = (double *)malloc(N * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            N, N, N, 1, A, N, B, N, 0, result1, N);

	double *result2 = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, result1, N, B, N, 0, result2, N);
	
	/**
	 * Block Logic: Compute A^T * A.
	 * Functional Utility: Calculates the symmetric component of the expression.
	 */
	double *res = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, res, N);

	/**
	 * Block Logic: Final summation pass.
	 */
	for (i = 0; i < N; ++i) {
		register int in = i * N;
		for (j = 0; j < N; ++j) {
			res[in + j] += result2[in + j];
		}
	}

	free(result1);
	free(result2);	
	return res;
}
