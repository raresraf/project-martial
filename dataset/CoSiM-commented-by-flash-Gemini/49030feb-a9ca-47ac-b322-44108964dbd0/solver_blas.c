
#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief High-performance matrix solver using standard BLAS library routines.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages industry-standard BLAS implementations 
 * for Level 3 matrix operations. It specifically uses `cblas_dtrmm` to exploit 
 * A's triangularity, `cblas_dgemm` for general multiplication with 
 * transposition, and `cblas_daxpy` for efficient vector-like addition.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Implementation of the matrix expression via BLAS sequence.
 * 
 * Algorithm: Atomic algebraic stages.
 * 1. Compute T1 = A * B using in-place `cblas_dtrmm` on a copy of B.
 * 2. Compute P1 = T1 * B^T using `cblas_dgemm` with transposition.
 * 3. Compute P2 = A^T * A using in-place `cblas_dtrmm` on a copy of A.
 * 4. Accumulate P2 into P1 using `cblas_daxpy`.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C = malloc(N * N * sizeof(double));
	double *mat = malloc(N * N * sizeof(double));
	
	/**
	 * Block Logic: Initial data migration.
	 * Invariant: 'mat' stores a copy of B for the first product term.
	 */
	for(int i = 0; i < N; i++){
		double *idx = &B[i * N];
		double *id = &mat[i * N];
		for(int j = 0; j < N; j++) {
			*id = *idx;
			id++;
			idx++;
		}
	}

	// Logic: Output matrix initialization to zero.
	double *point_C = &C[0];
	for(int i = 0; i < N * N; i++){
		*point_C = 0;
		point_C++;
	}

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM.
	 * Optimization: Exploits upper triangular property of matrix A to 
	 * transform 'mat' in-place.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
				CblasNonUnit, N, N, 1.0, A, N, mat, N);

	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM with transposition.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 
				1.0, mat, N, B, N, 1.0, C, N);
	
	/**
	 * Block Logic: Compute (A^T * A).
	 * Invariant: 'mat' is re-purposed to store a copy of A.
	 */
	for(int i = 0; i < N; i++){
		double *idx = &A[i * N];
		double *id = &mat[i * N];
		for(int j = 0; j < N; j++) {
			*id = *idx;
			id++;
			idx++;
		}
	}

	/**
	 * Block Logic: Triangular symmetric product.
	 * Algorithm: In-place DTRMM.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, 
				CblasNonUnit, N, N, 1.0, mat, N, mat, N);

	/**
	 * Block Logic: Final aggregation Result = P1 + P2.
	 * Algorithm: DAXPY (Scalar * Vector + Vector).
	 * Optimization: Treats the matrices as flat buffers for fast vector addition.
	 */
	cblas_daxpy(N * N, 1.0, mat, 1, C, 1);

	free(mat);
	return C;
}
