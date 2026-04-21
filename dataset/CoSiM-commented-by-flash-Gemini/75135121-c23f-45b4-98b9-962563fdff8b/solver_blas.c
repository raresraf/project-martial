/**
 * @75135121-c23f-45b4-98b9-962563fdff8b/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 *
 * Functional Utility: Solves the matrix expression Result = (A * B) * B^T + (A^T * A)
 * using high-performance GEMM (General Matrix Multiplication) routines. 
 * Leveraging BLAS ensures hardware-specific optimizations and efficient memory access.
 *
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

#include "utils.h"
#include "cblas.h"


/**
 * @brief Computes the complex matrix product sequence via BLAS routines.
 * Logic: Decomposes the expression into distinct multiplication stages.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	// Pre-condition: Allocates intermediate buffers for stage-wise results.
	double* res_AxB = (double*)malloc(N * N * sizeof(double));
	double* res_ABBt = (double*)malloc(N * N * sizeof(double));
	double* res_AtA = (double*)malloc(N * N * sizeof(double));
	double* res = (double*)malloc(N * N * sizeof(double));

	/**
	 * Block Logic: Compute stage 1: (A * B).
	 * Optimization: Employs `cblas_dgemm` for standard dense matrix multiplication.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		N, N, N, 1, A, N, B, N, 0, res_AxB, N);

	/**
	 * Block Logic: Compute stage 2: (A * B) * B^T.
	 * Logic: Applies transposition to the right operand B directly within GEMM.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1, res_AxB, N, B, N, 0, res_ABBt, N);

	/**
	 * Block Logic: Compute stage 3: (A^T * A).
	 * Functional Utility: Computes the Gramian matrix of A.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		N, N, N, 1, A, N, A, N, 0, res_AtA, N);

	/**
	 * Block Logic: Final summation pass.
	 * Invariant: Aggregates the results of stage 2 and stage 3 into the final output matrix.
	 */
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[i * N + j] = 0;
			res[i * N + j] += res_ABBt[i * N + j] + res_AtA[i * N + j];
		}
	}

	/**
	 * Cleanup: Releases intermediate stage buffers.
	 */
	free(res_ABBt);
	free(res_AtA);
	free(res_AxB);
	return res;	
	
}
