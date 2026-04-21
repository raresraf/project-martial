/**
 * @5df61160-eba2-4f4a-8fa3-bddbe166f9d1/solver_blas.c
 * @brief Matrix expression solver utilizing high-performance BLAS routines.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using Level 3 
 * BLAS GEMM operations for optimized throughput.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include "cblas.h"
#include "string.h"


/**
 * @brief Solver kernel leveraging the Basic Linear Algebra Subprograms (BLAS) library.
 * Logic: Decomposes the complex matrix expression into a sequence of GEMM calls.
 * Optimization: Uses hardware-specific BLAS implementations for peak floating-point performance.
 */
double* my_solver(int N, double *A, double *B) {
	
	double *TEMPORARY = calloc(N * N, sizeof(double));
	double *RESULT = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute TEMPORARY = A * B.
	 * Functional Utility: Standard dense matrix-matrix multiplication.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			N, N, N, 1, A, N, B, N, 0, TEMPORARY, N);

	/**
	 * Block Logic: Compute RESULT = TEMPORARY * B^T.
	 * Logic: Multiplies the intermediate product by the transpose of B.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			N, N, N, 1, TEMPORARY, N, B, N, 0, RESULT, N);

	/**
	 * Block Logic: Compute RESULT = RESULT + (A^T * A).
	 * Optimization: Computes the Gramian matrix (A^T * A) and adds it directly to the 
	 * existing RESULT buffer (beta=1 parameter in dgemm).
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			N, N, N, 1, A, N, A, N, 1, RESULT, N);

	free(TEMPORARY);
	return RESULT;
}
