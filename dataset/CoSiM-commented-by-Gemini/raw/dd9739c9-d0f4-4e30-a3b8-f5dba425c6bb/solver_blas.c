/**
 * @file solver_blas.c
 * @brief BLAS-based optimized matrix solver computing $C = A \times B \times B^T + A^T \times A$.
 * Algorithm: Relies on optimized numerical linear algebra kernels (DGEMM, DTRMM).
 * Time Complexity: $O(N^3)$ due to dense matrix multiplications.
 * Space Complexity: $O(N^2)$ for storing the result matrix C.
 */

#include "utils.h"
#include <string.h>
#include "cblas.h"



/**
 * @brief Computes C = A * B * B^T + A^T * A.
 * @param N Matrix dimension.
 * @param A Input matrix A.
 * @param B Input matrix B.
 * @return Pointer to resulting matrix C.
 */
double* my_solver(int N, double *A, double *B) {
	double* C = calloc(N * N, sizeof(double));
	double* X = calloc(N * N, sizeof(double));

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		N, N, N,
		1.0, A, N,
		B, N, 0, 
		X, N
	);


	
	double* Y = calloc(N * N, sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		112,
		N, N, N,
		1.0, X, N,
		B, N, 0, 
		Y, N
	);


	
	double* Z = calloc(N * N, sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		112,
		CblasNoTrans,
		N, N, N,
		1.0, A, N,
		A, N, 0, 
		Z, N
	);

	
	double* I = calloc(N * N, sizeof(double));
	/**
	 * Block Logic: Iterative processing loop.
	 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
	 */
	for(int i = 0; i < N; i++) {
		/**
		 * Block Logic: Iterative processing loop.
		 * Invariant: Maintains sequence integrity while progressing through the defined bounds.
		 */
		for(int j = 0; j < N; j++) {
			/**
			 * Block Logic: Conditional state branch.
			 * Invariant: The conditional branch maintains control flow invariants.
			 */
			if(j == i) {
				I[i * N + j] = 1;
			}
		}
	}


	
	memcpy(C, Z, N * N * sizeof(double));
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasNoTrans,
		N, N, N,
		1.0, I, N,
		Y, N, 1.0, 
		C, N
	);
	free(X);
	free(Y);
	free(I);
	free(Z);
	return C;
}
