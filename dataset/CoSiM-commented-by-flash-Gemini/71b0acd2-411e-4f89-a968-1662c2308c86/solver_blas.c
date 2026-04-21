/**
 * @71b0acd2-411e-4f89-a968-1662c2308c86/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using optimized 
 * TRMM and GEMM kernels. 
 * Warning: Implementation performs in-place mutation of matrix A.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include <string.h>
#include "cblas.h"


/**
 * @brief Allocates heap memory for result buffers.
 */
void alloc_matrix(int N, double **C) {
		*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);
}

/**
 * @brief High-performance solver kernel utilizing BLAS routine delegation.
 * Logic:
 * 1. Computes C = B * B^T using GEMM.
 * 2. Multiplies the result by A using TRMM (C = A * B * B^T).
 * 3. Computes the Gramian A^T * A in-place within matrix A using TRMM.
 * 4. Aggregates results.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C;
	int i;

	
	alloc_matrix(N, &C);

	/**
	 * Block Logic: Compute C = B * B^T.
	 * Functional Utility: Dense GEMM with explicit transposition of the right operand.
	 */
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N, 1,
				B, N,
				B, N,
				0.0, C, N);
	
	
	/**
	 * Block Logic: Compute C = A * C (effectively A * B * B^T).
	 * Optimization: Exploits A's upper triangularity.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		C, N);

	
	/**
	 * Block Logic: Compute A = A^T * A.
	 * Optimization: Computes the Gramian matrix in-place.
	 * Invariant: Destructive update to input matrix A.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		A, N);

	
	/**
	 * Block Logic: Final summation stage.
	 */
	for(i = 0; i < N * N; i++) {
			C[i] += A[i];
	}

	return C;
}
