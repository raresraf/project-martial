/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using a BLAS library (CBLAS interface).
 *
 * This file provides an implementation of the `my_solver` function, which is
 * defined in `utils.h`. This specific version leverages a high-performance
 * BLAS library to perform the underlying matrix computations. The solver calculates
 * the result of the expression: (A * B) * B^T + A^T * A.
 */
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>


/**
 * @brief Solves a matrix equation using BLAS functions.
 *
 * This function computes the expression: C = (A * B) * B^T + A^T * A, where
 * A and B are N x N matrices. It uses intermediate matrices to store partial
 * results and relies on `cblas_dtrmm` (triangular matrix multiplication) and
 * `cblas_dgemm` (general matrix multiplication) for the computations.
 *
 * @param N The dimension of the square matrices.
 * @param A A pointer to the N x N input matrix A. A is assumed to be upper triangular.
 * @param B A pointer to the N x N input matrix B.
 * @return A pointer to a newly allocated N x N matrix containing the result.
 *         The caller is responsible for freeing this memory.
 *
 * @note The implementation performs the following steps:
 * 1. Allocates two temporary matrices, `prod11` and `prod2`.
 * 2. Computes `prod2 = A^T * A` using `cblas_dtrmm`.
 * 3. Computes `prod11 = A * B` using `cblas_dtrmm`.
 * 4. Computes the final result `prod2 = (A * B) * B^T + (A^T * A)` which is equivalent to
 *    `prod2 = prod11 * B^T + prod2` using `cblas_dgemm`.
 * 5. Frees one temporary matrix (`prod11`) and returns the other (`prod2`) which
 *    holds the final result.
 */
double* my_solver(int N, double *A, double *B) {

	double *prod11, *prod2;

	int i, j;

	
	prod11 = calloc(N * N, sizeof(double));
	if (prod11 == NULL)
		exit(EXIT_FAILURE);

	
	prod2 = calloc(N * N, sizeof(double));
	if (prod2 == NULL)
		exit(EXIT_FAILURE);

	
	memcpy(prod2, A, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N, 1.0, A, N, prod2, N
	);


	
	memcpy(prod11, B, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N, 1.0, A, N, prod11, N
	);


	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N, 1.0, prod11, N, B, N, 1.0, prod2, N
	);

	free(prod11);

	return prod2;
}
