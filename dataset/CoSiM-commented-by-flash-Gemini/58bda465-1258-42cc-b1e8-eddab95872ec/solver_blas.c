/**
 * @file solver_blas.c
 * @brief Matrix expression solver using standardized BLAS library calls.
 *
 * This implementation solves the expression: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Sequential Level 3 BLAS operations.
 * 1. AB = A * B using cblas_dtrmm (triangular).
 * 2. ABBt = AB * B^T using cblas_dgemm.
 * 3. AtA = A^T * A using cblas_dtrmm (symmetric).
 * 4. Sum Result = ABBt + AtA.
 *
 * Domain: HPC, Linear Algebra.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * my_solver - Computes the matrix expression using BLAS routines.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	/**
	 * Pre-condition: Allocation of result and scratchpad buffers.
	 */
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Uses cblas_dtrmm to exploit the upper triangular property of A.
	 */
	memcpy(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, 1.0, A, N, AB, N);

	/**
	 * Block Logic: Compute ABBt = AB * B^T.
	 * Functional Utility: General matrix multiplication with implicit transposition.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, AB, N, B, N, 1.0, ABBt, N);

	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Computes the symmetric product of a triangular matrix in-place.
	 */
	memcpy(AtA, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1.0, A, N, AtA, N);

	/**
	 * Block Logic: Final summation of the two product terms.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
