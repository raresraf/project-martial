/**
 * @raw/925c9aec-7fc6-4d6b-aed5-ede8c01dc703/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A * B * B^T + A^T * A.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>

double* my_solver(int N, double *A, double *B) {
	double *ABBT, *ATA, *C;

	/**
	 * Functional Utility: Initializes dynamically allocated matrix buffers and caches.
	 */
	ABBT = calloc(N * N , sizeof(double));
	ATA = calloc(N * N , sizeof(double));
	C = calloc(N * N, sizeof(double));

	/**
	 * Functional Utility: Computes the base product ABBT = B * B^T natively.
	 */
	cblas_dgemm	(
		CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1,
		B, N,
		B, N,
		1, ABBT, N);

	/**
	 * Functional Utility: Re-evaluates ABBT = A * ABBT.
	 * Transforms ABBT into A * (B * B^T).
	 */
	cblas_dtrmm (
		CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		ABBT, N
	);

	/**
	 * Functional Utility: Duplicates upper triangular matrix A into ATA for accumulation.
	 */
	memcpy(ATA, A, N * N * sizeof(double));

	/**
	 * Functional Utility: Alters ATA = A^T * ATA.
	 * Since ATA initially equates A, solves A^T * A.
	 */
	cblas_dtrmm (
		CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		ATA, N
	);

	/**
	 * Block Logic: Synthesizes final matrix summation computing C = ABBT + ATA.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			C[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
		}
	}

	free(ABBT);
	free(ATA);

	return C;
}
