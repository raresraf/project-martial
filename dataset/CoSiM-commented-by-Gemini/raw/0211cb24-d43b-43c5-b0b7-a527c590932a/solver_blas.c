/**
 * @file solver_blas.c
 * @brief BLAS-based implementation of a matrix solver.
 *
 * This file provides an implementation of the `my_solver` function that uses
 * the Basic Linear Algebra Subprograms (BLAS) library to perform the required
 * matrix operations. This version is expected to be highly optimized and
 * efficient due to its reliance on a hardware-specific BLAS implementation.
 */
#include "utils.h"
#include "cblas.h"


/**
 * @brief Solves a matrix equation using BLAS functions.
 * @param N The dimension of the matrices.
 * @param A A pointer to the first input matrix.
 * @param B A pointer to the second input matrix.
 * @return A pointer to the resulting matrix.
 *
 * This function computes the matrix expression `A * B * B' + A' * A`, where `A'`
 * and `B'` are the transposes of matrices `A` and `B`, respectively.
 * The computation is performed using a series of BLAS calls for matrix
 * multiplication and copying.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *res1 = calloc(N * N, sizeof(double));
	if (res1 == NULL)
		exit(EXIT_FAILURE);

	double *res2 = calloc(N * N, sizeof(double));
	if (res2 == NULL)
		exit(EXIT_FAILURE);

	
	// Copies the content of matrix B into res1.
	cblas_dcopy(N * N, B, 1, res1, 1);

	
	
	// Computes the triangular matrix multiplication: res1 = A * res1.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
		CblasNoTrans, CblasNonUnit, N, N, 1, A, N, res1, N);

	
	// Copies the content of matrix A into res2.
	cblas_dcopy(N * N, A, 1, res2, 1);

	
	
	// Computes the triangular matrix multiplication: res2 = A' * res2.
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
		CblasTrans, CblasNonUnit, N, N, 1, A, N, res2, N);

	
	
	// Computes the general matrix-matrix multiplication: res2 = 1*res1*B' + 1*res2.
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, res1, N, B, N, 1, res2, N);


	free(res1);
	return res2;
}
