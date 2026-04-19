/**
 * @raw/842a6f4e-158e-4c42-b844-5d5699aef0e9/solver_blas.c
 * @brief High-performance dense matrix multiplication solver utilizing BLAS routines.
 * Algorithm: Leverages the Level 3 BLAS function `cblas_dtrmm` (Triangular Matrix-Matrix Multiplication)
 * for optimal hardware acceleration.
 * Time Complexity: $O(N^3)$ (amortized by highly optimized BLAS kernel)
 * Space Complexity: $O(N^2)$ for intermediate accumulation buffers.
 */

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	/**
	 * Functional Utility: Preallocates contiguous memory for results. 
	 * `result1` and `result2` will be manipulated directly in-place by BLAS.
	 */
	double *C = (double *)calloc(N * N, sizeof(double));
	if (C == NULL) return NULL;

	double *result1 = (double *)calloc(N * N, sizeof(double));
	if (result1 == NULL) return NULL;

	double *result2 = (double *)calloc(N * N, sizeof(double));
	if (result2 == NULL) return NULL;

	/**
	 * Block Logic: Initializes intermediate vectors as exact copies of B and A.
	 * Invariant: Sets up base operands to be overwritten by the `dtrmm` execution.
	 */
	for (int i = 0; i < N * N; i++) {
		result1[i] = B[i];
		result2[i] = A[i];
	}

	/**
	 * Functional Utility: Multiplies upper triangular matrix A with B.
	 * Result overwrites `result1` computing A * B.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
		CblasNonUnit, N, N, 1, A, N, result1, N);

	/**
	 * Functional Utility: Executes an in-place matrix-matrix multiplication 
	 * targeting the transposed upper triangular matrix.
	 */
	cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1, A, N, result1, N);

	/**
	 * Functional Utility: Computes A^T * A. Overwrites `result2`.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1, A, N, result2, N);

	/**
	 * Block Logic: Sums the intermediately computed matrices into the final target array C.
	 * Invariant: Performs an element-wise matrix addition step.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = result1[i * N + j] + result2[i * N + j];
		}
	}

	free(result1);
	free(result2);

	return C;
}
