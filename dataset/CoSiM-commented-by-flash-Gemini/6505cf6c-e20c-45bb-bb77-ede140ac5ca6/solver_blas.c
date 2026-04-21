/**
 * @6505cf6c-e20c-45bb-bb77-ede140ac5ca6/solver_blas.c
 * @brief Matrix expression solver utilizing high-performance Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using optimized 
 * BLAS kernels (DTRMM, DGEMM) to minimize floating-point operation count and 
 * maximize memory throughput.
 * Domain: HPC Numerical Optimization.
 */

#include <string.h>
#include "cblas.h"
#include "utils.h"


/**
 * @brief Computes the matrix expression via BLAS stage decomposition.
 * Logic: Strategically uses DTRMM for the triangular matrix A and DGEMM for 
 * the dense accumulation stages.
 */
double* my_solver(int N, double *A, double *B) {
	double *AB;
	double *ABB_t; 
	double *C;

	// Pre-condition: Dynamic memory allocation for intermediate result buffers.
	AB = calloc(N * N , sizeof(double));
	if (AB == NULL) 
		exit(EXIT_FAILURE);
	ABB_t = calloc(N * N , sizeof(double));
	if (ABB_t == NULL) 
		exit(EXIT_FAILURE);
	C = calloc(N * N , sizeof(double));
	if (C == NULL)
		exit(EXIT_FAILURE);	

	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Uses cblas_dtrmm to exploit the upper triangular property of matrix A.
	 */
	memcpy(AB, B, N * N * sizeof(double));

	cblas_dtrmm(CblasRowMajor, CblasLeft,
				CblasUpper, CblasNoTrans,
				CblasNonUnit, N, N,
				1.0, A, N,
				AB, N);

	/**
	 * Block Logic: Compute C = AB * B^T.
	 * Functional Utility: Performs general matrix multiplication with explicit transposition.
	 */
	memcpy(ABB_t, B, N * N * sizeof(double));

	cblas_dgemm(CblasRowMajor, CblasNoTrans,
                 CblasTrans, N, N,
                 N, 1.0, AB,
                 N, ABB_t, N,
                 1.0, C, N);

	/**
	 * Block Logic: Compute C = C + (A^T * A).
	 * Optimization: Computes the Gramian matrix and accumulates it into the result buffer.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans,
                 CblasNoTrans, N, N,
                 N, 1.0, A,
                 N, A, N,
                 1.0, C, N);
	

	free(AB);
	free(ABB_t);
	return C;
}
