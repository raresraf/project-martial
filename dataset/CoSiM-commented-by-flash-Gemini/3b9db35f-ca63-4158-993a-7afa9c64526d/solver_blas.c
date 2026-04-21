
#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief Optimized matrix solver using Level 3 BLAS routines.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages `cblas_dtrmm` for triangular matrix 
 * products and `cblas_dgemm` for general matrix multiplication. Interestingly, 
 * it uses an identity matrix (un) to perform matrix summation through a 
 * generalized multiplication operation.
 * 
 * Domain: HPC, Linear Algebra, BLAS Orchestration.
 */

/**
 * my_solver - Implementation of the matrix expression via sequence of BLAS calls.
 * 
 * Algorithm: Algebraic orchestration.
 * 1. Initialize buffers and a local identity matrix.
 * 2. Compute T1 = A * B using `cblas_dtrmm` (Left, Upper, NoTrans).
 * 3. Compute P1 = T1 * B^T using `cblas_dgemm` (NoTrans, Trans).
 * 4. Compute P2 = A^T * A using `cblas_dtrmm` (Left, Upper, Trans) in-place on A.
 * 5. Aggregate P1 into P2 using `cblas_dgemm` with an identity matrix (effectively P2 = 1*P1 + 1*P2).
 */
double* my_solver(int N, double *A, double *B) {
	double  *sum,*sum2, *c,  *un;
	int i,j;
	
	// Logic: Allocation and initialization of workspace.
	sum = calloc(N * N, sizeof(double));
	sum2 = calloc(N * N, sizeof(double));
	c = calloc(N * N,sizeof(double));
	un = calloc(N * N, sizeof(double));

	// Block Logic: Setup phase.
	// Invariant: 'sum' is a copy of B, 'un' is initialized as an Identity Matrix.
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			sum[i * N + j] = B[i * N + j];
			if (i == j)
				un[i * N + j] = 1;
			else un[i * N + j] = 0;
		}
	}

	/**
	 * Block Logic: Compute (A * B).
	 * Algorithm: DTRMM (Triangular Matrix Multiply).
	 * Optimization: Exploys A's upper triangularity.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft , CblasUpper, CblasNoTrans,
	CblasNonUnit, N, N, 1, A, N, sum, N);
	
	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Algorithm: DGEMM.
	 * Logic: Computes the product of the intermediate 'sum' and transposed B.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, sum, N, B, N, 1, sum2, N);

	/**
	 * Block Logic: Compute (A^T * A).
	 * Algorithm: In-place DTRMM.
	 * Logic: Transforms the triangular matrix A into the symmetric product A^T * A.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft , CblasUpper, CblasTrans,
	CblasNonUnit, N, N, 1, A, N, A, N);

	/**
	 * Block Logic: Final summation.
	 * Algorithm: DGEMM-based addition.
	 * Logic: Uses sum2 * Identity + A to perform the final addition of the 
	 * two major terms. Result is stored in A.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, sum2, N, un, N, 1, A, N);
	
	// Synchronization: Copying the final result from the workspace to the output matrix c.
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			c[i * N + j] = A[i * N + j];
		}
	}

	free(sum);
	free(sum2);
	free(un);
	return c;
}
