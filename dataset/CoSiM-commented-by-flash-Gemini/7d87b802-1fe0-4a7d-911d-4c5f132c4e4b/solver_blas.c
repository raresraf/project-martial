/**
 * @7d87b802-1fe0-4a7d-911d-4c5f132c4e4b/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 *
 * Functional Utility: Solves the matrix expression Result = (A * B * B^T) + (A^T * A)
 * by decomposing it into distinct high-performance linear algebra stages (GEMM and TRMM).
 *
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include "cblas.h"


/**
 * @brief Computes the matrix expression via BLAS routines.
 * Algorithm: Stage-based linear algebra calculation.
 * Logic:
 * 1. Computes C = A * B using `cblas_dtrmm` (triangular multiplication).
 * 2. Computes D = C * B^T using `cblas_dgemm`.
 * 3. Computes E = A^T * A using `cblas_dtrmm`.
 * 4. Aggregates terms using `cblas_daxpy`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	// Pre-condition: Allocates stack-based VLAs for intermediate product buffers.
	double c[N][N], d[N][N];
	
	// Logic: Initializing buffer `c` with matrix `B`.
	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			c[i][j] = B[i * N + j];
		}
	}

	double *e = malloc(N * N * sizeof(double));

	double *orig_pc = &c[0][0];
	double *orig_pd = &d[0][0];

	
	/**
	 * Block Logic: Compute C = A * B.
	 * Optimization: Uses `cblas_dtrmm` to exploit A's upper triangular property.
	 * Invariant: In-place update of buffer `c`.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, orig_pc, N);

	/**
	 * Block Logic: Compute D = C * B^T.
	 * Functional Utility: Dense matrix multiplication with explicit transposition of B.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                N, N, N, 1.0, orig_pc, N, B, N, 0.0, orig_pd, N);

	
	// Pre-condition: Prepare result buffer `e` with values from `A`.
	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			e[i * N + j] = A[i * N + j];
		}
	}
	
	/**
	 * Block Logic: Compute E = A^T * A.
	 * Optimization: Computes the Gramian matrix in-place within result buffer `e`.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                N, N, 1.0, A, N, e, N);

	/**
	 * Block Logic: Final aggregation Result = E + D.
	 * Functional Utility: Leverages vector addition (DAXPY) for efficient terminal accumulation.
	 */
	cblas_daxpy(N * N, 1.0, orig_pd, 1, e, 1);	
	
	return e;
}
