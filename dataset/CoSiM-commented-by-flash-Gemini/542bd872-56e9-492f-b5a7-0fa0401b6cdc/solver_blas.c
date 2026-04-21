
#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief Optimized matrix expression solver utilizing standard BLAS library routines.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Employs Level 3 BLAS routines to maximize computational 
 * throughput. It specifically uses `cblas_dtrmm` to exploit A's triangular 
 * property, `cblas_dgemm` for general multiplication with transposition, 
 * and `cblas_daxpy` for optimized vector-based addition.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Computes the complex matrix expression via sequential BLAS orchestration.
 * 
 * Algorithm: Atomic algebraic stages.
 * 1. Compute T1 = A * B using in-place `cblas_dtrmm` on a copy of B.
 * 2. Compute P1 = T1 * B^T using `cblas_dgemm` with transposition.
 * 3. Compute P2 = A^T * A using in-place `cblas_dtrmm` (Transposed multiply) on a copy of A.
 * 4. Aggregate P1 into P2 using `cblas_daxpy`.
 */
double* my_solver(int N, double *A, double *B) {
	
	// Logic: Intermediate workspace for the (A * B) product.
	double *C1 = calloc(N * N, sizeof(double));

	// Pre-condition: Prepare B for in-place triangular multiplication.
	cblas_dcopy(N * N, B, 1, C1, 1);

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM.
	 * Optimization: Exploits upper triangularity of A to transform the copy 
	 * of B ('C1') directly.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N,
				1, A, N, C1, N);

	// Logic: Intermediate workspace for the ((A * B) * B^T) component.
	double *C2 = calloc(N * N, sizeof(double));

	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM with Transposition.
	 * Logic: Multiplies the intermediate T1 by transposed B to generate P1.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
				1, C1, N, B, N, 0, C2, N);

	// Logic: Workspace for the (A^T * A) component.
	double *C3 = calloc(N * N, sizeof(double));

	// Pre-condition: Prepare A for symmetric product.
	cblas_dcopy(N * N, A, 1, C3, 1);

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: In-place DTRMM.
	 * Logic: Transforms the triangular matrix A into its Gramian matrix.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N,
				1, A, N, C3, N);

	/**
	 * Block Logic: Final aggregation Result = P1 + P2.
	 * Algorithm: DAXPY (Scalar * Vector + Vector).
	 * Optimization: Accumulates the first term into the second term, which 
	 * will be returned as the result.
	 */
	cblas_daxpy(N * N, 1, C2, 1, C3, 1);

	free(C1);
	free(C2);

	return C3;
}
