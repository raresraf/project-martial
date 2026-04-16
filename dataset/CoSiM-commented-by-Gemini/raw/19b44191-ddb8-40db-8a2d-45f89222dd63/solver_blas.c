/**
 * @file solver_blas.c
 * @brief BLAS-based implementation of a matrix equation solver.
 * @details This file contains a function that computes the result of the
 * matrix equation: C = A^T * A + A * (B * B^T). It leverages the CBLAS
 * interface for high-performance linear algebra operations.
 *
 * @warning This implementation modifies the input matrix A in-place.
 */
#include "utils.h"
#include <cblas.h>
#include <stdlib.h>

/**
 * @brief Computes the matrix expression C = A^T * A + A * (B * B^T) using BLAS.
 *
 * @warning This function modifies the input matrix A, overwriting it with A^T * A.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major, upper triangular).
 *          This matrix will be modified in-place.
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix (C). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double *B) {
	double *C, *rez;
	int i, j;
	// Allocate and zero-initialize memory for the final result and an intermediate matrix.
	C = calloc(N * N, sizeof(double));
	rez = calloc(N * N, sizeof(double));

	
	/**
	 * Block Logic: Compute rez = B * B^T.
	 * cblas_dgemm computes rez = 1.0 * B * B^T + 0.0 * rez.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, rez, N);

	
	/**
	 * Block Logic: Compute rez = A * (B * B^T).
	 * cblas_dtrmm computes rez = 1.0 * A * rez, where rez is the result from the previous step.
	 * This assumes A is an upper triangular matrix.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, rez, N);

	
	/**
	 * Block Logic: Compute A = A^T * A (in-place).
	 * cblas_dtrmm computes A = 1.0 * A^T * A.
	 * This operation overwrites the original content of matrix A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

	/**
	 * Block Logic: Final summation.
	 * C = A + rez => (A^T * A) + (A * B * B^T)
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = A[i * N + j] + rez[i * N + j];
		}
	}

	// Free the intermediate matrix.
	free(rez);
	return C;
}
