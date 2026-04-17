/**
 * @file solver_blas.c
 * @brief A BLAS-based implementation of a matrix solver.
 * @details This file provides a high-performance solution for the matrix equation
 * C = (A * B) * B' + A' * A, where A is an upper triangular matrix. It leverages the
 * Basic Linear Algebra Subprograms (BLAS) library for optimized matrix operations,
 * including a fused multiply-add operation.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"


/**
 * @brief Solves C = (A * B) * B' + A' * A using BLAS functions.
 * @param N The dimension of the square matrices.
 * @param A A pointer to the input upper triangular matrix A (N x N).
 * @param B A pointer to the input matrix B (N x N).
 * @return A pointer to the resulting matrix C (N x N).
 *
 * @details This function uses highly optimized BLAS routines in an efficient manner:
 * 1. Computes the intermediate product `result_1 = A * B` using `cblas_dtrmm`.
 * 2. Computes the term `result_3 = A' * A` using `cblas_dtrmm`.
 * 3. In a single, fused operation, it calculates the final result by computing
 *    `(A * B) * B'` and adding it to the existing `A' * A` term. This is done
 *    by `cblas_dgemm` with `beta = 1.0`.
 * This approach minimizes memory operations and leverages the highly optimized
 * routines for the bulk of the computation.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER
");
	double* result_1 = (double *)calloc(N * N, sizeof(double));
	double* result_2 = (double *)calloc(N * N, sizeof(double)); // This buffer is allocated but not used.
	double* result_3 = (double *)calloc(N * N, sizeof(double));

	
	/**
	 * Block Logic: Step 1: Compute the intermediate product result_1 = A * B.
	 * First, B is copied into result_1.
	 * Then, `cblas_dtrmm` performs a triangular matrix multiplication:
	 * `result_1 = 1.0 * op(A) * result_1`, where op(A) is the upper triangular A.
	 */
	memcpy(result_1, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, result_1, N);	
	
	
	/**
	 * Block Logic: Step 2: Compute the second term result_3 = A' * A.
	 * First, A is copied into result_3.
	 * Then, `cblas_dtrmm` performs a triangular matrix multiplication:
	 * `result_3 = 1.0 * op(A) * result_3`, where op(A) is A' (transpose).
	 */
	memcpy(result_3, A, N*N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, result_3, N);

	
	/**
	 * Block Logic: Step 3: Compute the final result in a fused operation.
	 * `cblas_dgemm` calculates `result_3 = alpha * result_1 * B' + beta * result_3`.
	 * With alpha=1.0 and beta=1.0, this computes:
	 * `result_3 = (A * B) * B' + (A' * A)`.
	 * The result is stored in `result_3`, which is returned.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasTrans, N, N, N, 1.0, result_1, N, B, N, 1.0, result_3, N);
	
	free(result_1);
	// Note: result_2 is allocated but never used, a minor memory leak if not freed.
	free(result_2);
	return result_3;
}
