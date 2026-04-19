/**
 * @raw/99113ad8-03b9-45a3-a87a-1ec710110a68/solver_blas.c
 * @brief High-performance dense matrix solver using Level 3 BLAS operations.
 * Algorithm: Hardware-accelerated computation of C = A * (B * B^T) + A^T * A using contiguous cblas functions.
 * Time Complexity: $O(N^3)$ (amortized via BLAS highly optimized kernels)
 * Space Complexity: $O(N^2)$ representing distinct output and intermediate buffers.
 */

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	/**
	 * Functional Utility: Initializes necessary dynamically allocated matrix partitions.
	 */
	double *result = (double *) calloc(N*N, sizeof(double));

	/**
	 * Functional Utility: General matrix multiplication calculating result = B * B^T natively.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, result, N);

	/**
	 * Functional Utility: In-place calculation evaluating result = A * result. Resolves to A * (B * B^T).
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, result, N);

	/**
	 * Functional Utility: Aggregates result = result + A^T * A. Solves the final system formulation directly into the resulting pointer.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, N, N, 1, A, N, A, N, 1, result, N);
	return result;
}
