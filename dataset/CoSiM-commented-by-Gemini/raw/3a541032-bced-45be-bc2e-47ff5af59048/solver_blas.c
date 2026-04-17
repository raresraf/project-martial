/**
 * @file solver_blas.c
 * @brief BLAS-optimized implementation of matrix operations.
 *
 * Utilizes high-performance CBLAS routines to compute A * B * B^T + A^T * A.
 */

#include "utils.h"
#include "cblas.h"

/**
 * @brief Solves the matrix equation using BLAS routines.
 *
 * @param N Matrix dimension.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return Pointer to the resulting matrix.
 */
double* my_solver(int N, double *A, double *B) {
	
	int i = 0;
	int j = 0;

	double *auxB = (double *)calloc(N * N, sizeof(double));
	/**
	 * @brief Initialize auxB with B.
	 * Pre-condition: auxB is allocated.
	 * Invariant: Array initialized up to index i.
	 */
	for(i = 0; i < N * N; i++)
                auxB[i] = B[i];
	
	double *C = (double *)calloc(N * N, sizeof(double));
	/**
	 * @brief Initialize C with B (will be overwritten by BLAS).
	 * Pre-condition: C is allocated.
	 * Invariant: Array initialized up to index i.
	 */
        for(i = 0; i < N * N; i++)
                C[i] = B[i];
	
	double *D = (double *)calloc(N * N, sizeof(double));
	/**
	 * @brief Initialize D with A (will be overwritten by BLAS).
	 * Pre-condition: D is allocated.
	 * Invariant: Array initialized up to index i.
	 */
        for(i = 0; i < N * N; i++)
                D[i] = A[i];
	
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, auxB, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, auxB, N, B, N, 0, C, N);
	
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, D, N);

	/**
	 * @brief Accumulate the final result C = (A * B * B^T) + (A^T * A).
	 * Pre-condition: C and D contain partial BLAS results.
	 * Invariant: Rows up to index i are summed.
	 */
	for (i = 0; i < N; ++i) {
		/**
		 * @brief Element-wise addition.
		 * Pre-condition: Valid row index i.
		 * Invariant: Elements up to j are summed.
		 */
		for (j = 0; j < N; ++j) {
			C[i * N + j] += D[i * N + j];
		}
	}

	free(auxB);
	free(D);	
	return C;
}
