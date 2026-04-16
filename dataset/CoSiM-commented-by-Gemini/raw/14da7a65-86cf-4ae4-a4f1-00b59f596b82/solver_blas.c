/**
 * @file solver_blas.c
 * @brief BLAS-based implementation of a matrix equation solver.
 * @details This file contains a function that computes the result of the
 * matrix equation: result = (A * B) * B^T + A^T * A, where A and B are
 * N x N matrices. It leverages the CBLAS interface for high-performance
 * Basic Linear Algebra Subprograms, assuming A is an upper triangular matrix.
 */
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include "cblas.h"

/**
 * @brief Computes the matrix expression (A * B) * B^T + A^T * A using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N, row-major, upper triangular).
 * @param B A pointer to the second input matrix (N x N, row-major).
 * @return A pointer to the resulting N x N matrix (named C in the implementation). The caller is responsible for freeing this memory.
 */
double* my_solver(int N, double *A, double *B) {
	double *C, *AB, *ABB_t;
	int i, j;

	// Allocate memory for the result matrix C (which will first hold A^T * A).
	C = (double *)calloc(N * N, sizeof(double));
	if (NULL == C)
		exit(EXIT_FAILURE);
	
	// Allocate memory for the intermediate matrix AB (A * B).
	AB = (double *)calloc(N * N, sizeof(double));
	if (NULL == AB)
		exit(EXIT_FAILURE);

	// Allocate memory for the intermediate matrix ABB_t ((A * B) * B^T).
	ABB_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);

	
	// Initialize C with the contents of A to compute A^T * A in place.
	memcpy(C, A, N * N * sizeof(double));

	/**
	 * Block Logic: Compute C = A^T * A.
	 * cblas_dtrmm calculates C = 1.0 * A^T * C, where C is initially a copy of A.
	 * This calculates the first term of the main equation.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N
	);


	
	// Initialize AB with the contents of B to compute A * B in place.
	memcpy(AB, B, N * N * sizeof(double));

	/**
	 * Block Logic: Compute AB = A * B.
	 * cblas_dtrmm calculates AB = 1.0 * A * AB, where AB is initially a copy of B.
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
	 * Block Logic: Compute ABB_t = (A * B) * B^T.
	 * cblas_dgemm calculates ABB_t = 1.0 * AB * B^T + 0.0 * ABB_t.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, AB, N,
		B, N, 
		0.0, ABB_t, N
	);

	
	/**
	 * Block Logic: Final summation.
	 * C = C + ABB_t => (A^T * A) + ((A * B) * B^T)
	 */
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] += ABB_t[i * N + j];
	
	// Free intermediate matrices. The result matrix C is returned.
	free(AB);
	free(ABB_t);
	return C;
}

