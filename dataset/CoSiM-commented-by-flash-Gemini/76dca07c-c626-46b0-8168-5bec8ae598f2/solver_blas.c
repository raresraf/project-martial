/**
 * @76dca07c-c626-46b0-8168-5bec8ae598f2/solver_blas.c
 * @brief Optimized matrix expression solver utilizing Level 3 BLAS.
 *
 * Functional Utility: Solves the matrix expression Result = (A * B * B^T) + (A^T * A)
 * by decomposing it into optimized linear algebra stages. Employs GEMM for dense 
 * products and TRMM for operations involving the upper triangular matrix A.
 *
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * @brief High-performance solver kernel utilizing hardware-accelerated BLAS.
 * Logic:
 * 1. Computes M1 = B * B^T via `cblas_dgemm`.
 * 2. Computes M2 = A^T * A via `cblas_dtrmm` (in-place on A's copy).
 * 3. Final aggregation: Result = M2 + (A * M1) via `cblas_dgemm`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	// Pre-condition: Allocates workspace for intermediate and result buffers.
	double *M2 = (double *) malloc(N * N * sizeof(double));
	double *M1 = (double *) calloc(N * N, sizeof(double));
	memcpy(M2, A, N * N * sizeof(double));

	
	/**
	 * Block Logic: Compute M1 = B * B^T.
	 * Optimization: Uses general matrix multiplication with explicit transposition.
	 */
 	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 1.0, M1, N);
	
	/**
	 * Block Logic: Compute M2 = A^T * A.
	 * Optimization: Uses `cblas_dtrmm` to exploit A's upper triangular sparsity 
	 * during the Gramian matrix calculation.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, M2, N);
	
	/**
	 * Block Logic: Compute final result Result = M2 + (A * M1).
	 * Functional Utility: Dense GEMM with additive accumulation (beta=1.0) into M2.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, M1, N, 1.0, M2, N);

	/**
	 * Cleanup: Releases intermediate buffer M1.
	 */
	free(M1);
	return M2;



}
