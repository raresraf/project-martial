/**
 * @file solver_blas.c
 * @brief BLAS-optimized implementation of matrix operations.
 *
 * Utilizes the CBLAS interface for highly tuned matrix multiplications,
 * computing C = A * B * B^T + A^T * A.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>

#define ALPHA 1.0


/**
 * @brief Solves the matrix equation using BLAS routines for maximum performance.
 *
 * @param N Matrix dimension (N x N).
 * @param A Pointer to the first input matrix (upper triangular).
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix C, or NULL on allocation failure.
 */
double* my_solver(int N, double *A, double *B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;

	memcpy(aux, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, ALPHA, A, N, aux, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, ALPHA, aux, N, B, N, ALPHA, C, N);

	memcpy(aux, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, ALPHA, A, N, aux, N);

	int i, j;
	/**
	 * @brief Accumulate the final result C = (A * B * B^T) + (A^T * A).
	 * Pre-condition: C contains A * B * B^T, aux contains A^T * A.
	 * Invariant: Rows up to index i are summed.
	 */
	for (i = 0; i < N; i++) {
		/**
		 * @brief Element-wise addition.
		 * Pre-condition: Valid row index i.
		 * Invariant: Elements up to j are summed.
		 */
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);


	return C;
}
