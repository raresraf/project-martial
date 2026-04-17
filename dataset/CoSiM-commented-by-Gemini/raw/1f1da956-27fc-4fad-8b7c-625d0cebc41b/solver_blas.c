/**
 * @raw/1f1da956-27fc-4fad-8b7c-625d0cebc41b/solver_blas.c
 * @brief Computes the matrix expression C = A * B * B^T + A^T * A using highly optimized BLAS routines.
 * * Algorithm: Matrix multiplication with BLAS.
 * Time Complexity: $O(N^3)$ dominated by the general matrix-matrix multiplication (GEMM).
 * Space Complexity: $O(N^2)$ for storing the result and intermediate matrices.
 */

#include <string.h>
#include "utils.h"
#include "cblas.h"

/**
 * Functional Utility: Element-wise matrix addition.
 */
void addition(double *C, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			C[i * N + j] = *A + *B;
			++A;
			++B;
		}
	}
}

/**
 * Functional Utility: Calculates the resulting matrix leveraging BLAS library operations.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(*C));
	if (!C) {
		exit(-1);
	}

	double *AB = malloc(N * N * sizeof(*AB));
	if (!AB) {
		exit(-1);
	}
	memcpy(AB, B, N * N * sizeof(*B));

	double *ABBt = calloc(N * N, sizeof(*ABBt));
	if (!ABBt) {
		exit(-1);
	}

	double *AtA = malloc(N * N * sizeof(*AtA));
	if (!AtA) {
		exit(-1);
	}
	memcpy(AtA, A, N * N * sizeof(*A));

	/**
	 * Functional Utility: Computes AB = A * B in-place using triangular matrix multiplication.
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
	 * Functional Utility: Computes ABBt = AB * B^T.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AB, N,
		B, N,
		0.0,
		ABBt, N);

	/**
	 * Functional Utility: Computes AtA = A^T * A in-place.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AtA, N
	);

	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
