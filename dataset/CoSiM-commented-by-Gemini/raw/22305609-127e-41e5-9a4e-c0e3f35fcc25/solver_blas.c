/**
 * @file solver_blas.c
 * @brief A BLAS-based implementation of a matrix solver.
 * @details This file provides a high-performance solution for the matrix equation
 * C = (A * B) * B' + A' * A. It leverages the Basic Linear Algebra
 * Subprograms (BLAS) library for optimized matrix operations.
 */
#include "utils.h"
#include <cblas.h>
#include <string.h>


/**
 * @brief Solves C = (A * B) * B' + A' * A using BLAS functions.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details The function uses highly optimized BLAS routines to perform the calculation.
 * It does not exploit the upper-triangular nature of A in the first multiplication,
 * using a general matrix-matrix multiplication instead.
 * 1. Computes the intermediate product `C = A * B` using `cblas_dgemm`.
 * 2. Copies the result to a temporary buffer `tmp2`.
 * 3. Computes the first term `C = tmp2 * B'`, which is `(A * B) * B'`, using `cblas_dgemm`.
 * 4. Computes the second term `tmp = A' * A` using `cblas_dgemm`.
 * 5. Adds the second term to the first term in a final loop: `C = C + tmp`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	
	register size_t i, j;

	double *C = calloc(sizeof(double), N * N);
	double *tmp = calloc(sizeof(double), N * N);
	double *tmp2 = calloc(sizeof(double), N * N);

	
	/**
	 * Block Logic: Step 1: Compute the intermediate product C = A * B.
	 * `cblas_dgemm` calculates `C = alpha * A * B + beta * C`.
	 * With alpha=1.0 and beta=0.0, this computes `C = A * B`.
	 * Note: This does not exploit the upper-triangular nature of A.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
	
	memcpy(tmp2, C, N * N * sizeof(double));
	
	/**
	 * Block Logic: Step 2: Compute the first term C = (A * B) * B'.
	 * `cblas_dgemm` calculates `C = alpha * tmp2 * B' + beta * C`.
	 * With alpha=1.0 and beta=0.0, this computes `C = (A * B) * B'`.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, tmp2, N, B, N, 0, C, N);
	
	/**
	 * Block Logic: Step 3: Compute the second term tmp = A' * A.
	 * `cblas_dgemm` calculates `tmp = alpha * A' * A + beta * tmp`.
	 * With alpha=1.0 and beta=0.0, this computes `tmp = A' * A`.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 0, tmp, N);

	/**
	 * Block Logic: Step 4: Final addition: C = C + tmp.
	 * This adds the second term (A' * A) to the first term ((A * B) * B').
	 * Time Complexity: O(N^2)
	 */
	for (i = 0; i < N; ++i) {
		register double *pc = C + N * i;
		register double *ptmp = tmp + N * i;
		for (j = 0; j < N; ++j) {
			*pc += *ptmp;
			pc++, ptmp++;
		}
	}
	free(tmp);
	free(tmp2);

	return C;
}
