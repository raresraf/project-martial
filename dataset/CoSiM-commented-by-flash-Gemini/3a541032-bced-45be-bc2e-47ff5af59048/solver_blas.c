
#include "utils.h"
#include "cblas.h"

/**
 * @file solver_blas.c
 * @brief Matrix expression solver using standardized BLAS library calls.
 * 
 * Functional Intent: Computes the resulting matrix for the expression:
 * Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 * 
 * Performance Optimization: Leverages BLAS Level 3 operations to achieve 
 * high throughput. It uses `cblas_dtrmm` for triangular multiplication 
 * and `cblas_dgemm` for general matrix multiplication with transposition 
 * support.
 * 
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

/**
 * my_solver - Computes the complex matrix expression via sequential BLAS orchestration.
 * 
 * Algorithm: Algebraic stage-based execution.
 * 1. Initialize workspace buffers with copies of input matrices.
 * 2. Compute T1 = A * B using `cblas_dtrmm` (Left, Upper).
 * 3. Compute P1 = T1 * B^T using `cblas_dgemm` (NoTrans, Trans).
 * 4. Compute P2 = A^T * A using `cblas_dgemm` (Trans, NoTrans).
 * 5. Aggregate results P1 and P2 via element-wise addition.
 */
double* my_solver(int N, double *A, double *B) {
	int i = 0;
	int j = 0;

	// Logic: Allocation and initialization of workspace for (A * B).
	double *auxB = (double *)calloc(N * N, sizeof(double));
	for(i = 0; i < N * N; i++)
                auxB[i] = B[i];
	
	// Logic: Allocation for the primary result container P1.
	double *C = (double *)calloc(N * N, sizeof(double));
        for(i = 0; i < N * N; i++)
                C[i] = B[i];
	
	// Logic: Allocation for the symmetric product component P2.
	double *D = (double *)calloc(N * N, sizeof(double));
        for(i = 0; i < N * N; i++)
                D[i] = A[i];
	
	/**
	 * Block Logic: Compute (A * B).
	 * Optimization: Uses DTRMM to exploit the upper triangular property of matrix A.
	 * Invariant: 'auxB' is transformed from B to the product (A * B).
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, auxB, N);
	
	/**
	 * Block Logic: Compute (A * B) * B^T.
	 * Logic: Multiplies the intermediate product by the transpose of B.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, auxB, N, B, N, 0, C, N);
	
	/**
	 * Block Logic: Compute (A^T * A).
	 * Logic: Computes the gramian matrix of A using generalized multiplication 
	 * with leading transposition.
	 */
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, D, N);

	/**
	 * Block Logic: Final accumulation pass.
	 * Logic: Naive element-wise addition of the two matrix products.
	 */
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += D[i * N + j];
		}
	}

	free(auxB);
	free(D);	
	return C;
}
