/**
 * @713e75e8-5b1a-4582-934e-bc9c94a7c91b/solver_blas.c
 * @brief Optimized matrix expression solver utilizing Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) by decomposing it 
 * into highly optimized TRMM (Triangular Matrix Multiplication) and GEMM stages.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>


/**
 * @brief High-performance solver kernel.
 * Logic:
 * 1. Computes the Gramian matrix A^T * A using TRMM.
 * 2. Computes the product B * B^T using GEMM.
 * 3. Multiplies the result by A using TRMM (to get A * B * B^T).
 * 4. Aggregates the two terms using AXPY.
 */
double* my_solver(int N, double *A, double *B) {
	
	double *C = calloc(sizeof(double), N * N);
	double *a_copy = calloc(sizeof(double), N * N);

	// Pre-condition: Prepare operand buffer for A^T * A.
	memcpy(a_copy, A, N * N * sizeof(double));

	/**
	 * Block Logic: Compute a_copy = A^T * A.
	 * Optimization: Uses cblas_dtrmm with implicit transposition.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, a_copy, N, a_copy, N);

	/**
	 * Block Logic: Compute C = B * B^T.
	 * Functional Utility: Dense matrix multiplication with explicit transposition.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 0.0, C, N);

	/**
	 * Block Logic: Compute C = A * C (equivalent to A * B * B^T).
	 * Optimization: Exploits the upper triangularity of A for the left-hand operand.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	/**
	 * Block Logic: Final summation: C = a_copy + C.
	 * Functional Utility: Leverages vector addition (DAXPY) for final result consolidation.
	 */
	cblas_daxpy(N * N, 1.0, a_copy, 1.0, C, 1.0);

	free(a_copy);
	
    return C;
}
