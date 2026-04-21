/**
 * @66423df7-06af-4ccd-ac5c-54f5e8b29793/solver_blas.c
 * @brief Optimized matrix expression solver using Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) using TRMM and 
 * DGEMM operations to exploit matrix symmetry and triangularity.
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include "cblas.h"
#include <string.h>


/**
 * @brief High-performance solver kernel.
 * Logic: Minimizes memory traffic by performing in-place triangular multiplications 
 * and aggregating results into a single accumulation buffer.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *AB = (double *)calloc(N * N, sizeof(double));
	double *AtA = (double *) calloc(N * N, sizeof(double));

	// Pre-condition: Prepare operand buffer for A * B.
	memcpy(AB, B, N * N * sizeof(double));
	
	/**
	 * Block Logic: Compute AB = A * B.
	 * Optimization: Uses cblas_dtrmm for the left-hand upper triangular matrix A.
	 */
	cblas_dtrmm (CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit , N, N, 1.0, A, N, AB, N);
	
	// Pre-condition: Prepare operand buffer for A^T * A.
	memcpy(AtA, A, N * N * sizeof(double));
	
	/**
	 * Block Logic: Compute AtA = A^T * A.
	 * Optimization: Computes the Gramian matrix in-place using TRMM with leading transposition.
	 */
	cblas_dtrmm (CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit , N, N, 1.0, A, N, AtA, N);

	/**
	 * Block Logic: Final accumulation: AtA = AB * B^T + AtA.
	 * Functional Utility: Dense DGEMM with beta=1.0 for result aggregation.
	 */
    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, AtA, N);
	
	free(AB);
	return AtA;
}
