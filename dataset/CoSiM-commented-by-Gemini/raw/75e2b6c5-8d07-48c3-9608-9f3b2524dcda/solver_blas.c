/**
 * @file solver_blas.c
 * @brief A matrix solver implementation using BLAS routines.
 * @note This version uses a clear, step-by-step approach, allocating new memory
 *       for each intermediate result, which makes it easy to follow but less
 *       memory-efficient than other versions.
 */

#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "cblas.h"


/**
 * @brief Performs a sequence of matrix operations using BLAS.
 *
 * @param N The dimension of the square matrices A and B.
 * @param A A pointer to the first input matrix (N x N).
 * @param B A pointer to the second input matrix (N x N).
 * @return A pointer to the resulting matrix C. The caller is responsible for freeing this memory.
 *
 * @note The function computes the expression: C = A * (B * B^T) + A^T * A.
 *       It breaks down the calculation into clear, sequential BLAS calls.
 */
double* my_solver(int N, double *A, double *B) {
	register int size = N * N * sizeof(double);

	
	// Step 1: Compute BB = B * B^T.
	double *BB = malloc(size);
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1, B, N,
		B, N, 0,
		BB, N
	);

	// Step 2: Compute AB = A * (B * B^T).
	// First, copy the result of B*B^T into a new matrix AB.
	double *AB = malloc(size);
	memcpy(AB, BB, size);

	// Then, multiply by A, treating A as an upper triangular matrix.
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		AB, N
	);

	
	// Step 3: Initialize the final result matrix C with the result from Step 2.
	double *C = malloc(size);
	memcpy(C, AB, size);

	// Step 4: Compute C = (A^T * A) + C.
	// The cblas_dgemm call computes A^T * A and adds it to the existing C matrix.
	cblas_dgemm(
		CblasRowMajor,
		CblasTrans,
		CblasNoTrans,
		N, N, N,
		1, A, N,
		A, N, 1,
		C, N
	);


	// Free the memory used for intermediate matrices.
	free(AB);
	free(BB);

	return C;
}
