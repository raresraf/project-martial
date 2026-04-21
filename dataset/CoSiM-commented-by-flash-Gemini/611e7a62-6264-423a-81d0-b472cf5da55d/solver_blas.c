/**
 * @611e7a62-6264-423a-81d0-b472cf5da55d/solver_blas.c
 * @brief Matrix expression solver utilizing high-performance Level 3 BLAS.
 * Functional Utility: Computes Result = (A * B) * B^T + (A^T * A) where A is upper 
 * triangular, using optimized triangular matrix-matrix multiplication (TRMM).
 * Domain: HPC Numerical Optimization.
 */

#include "utils.h"
#include "cblas.h"
#include <stdlib.h>
#include <string.h>

/**
 * @brief Computes the matrix expression via BLAS stage decomposition.
 * Logic: Exploits triangular properties of A to reduce computational complexity.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *B1;
	double *C;
	double *A1;
	int i, j;
	B1 = calloc(N*N, sizeof(double));
	C = calloc(N*N, sizeof(double));
	A1 = calloc(N*N, sizeof(double));
	
	// Pre-condition: Buffers are initialized with input data to allow in-place TRMM.
	memcpy(B1, B, N*N * sizeof(double));
	memcpy(A1, A, N*N * sizeof(double));

	/**
	 * Block Logic: Compute B1 = A * B.
	 * Optimization: Uses cblas_dtrmm to take advantage of A being upper triangular.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N,
		N,
		1.0,
		A, N,
		B1, N
	);

	/**
	 * Block Logic: Compute C = B1 * B^T.
	 * Functional Utility: General matrix multiplication with transposition.
	 */
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1.0,
		B1, N, B,
		N, 0.0, C, N
	);

	/**
	 * Block Logic: Compute A1 = A^T * A.
	 * Optimization: In-place triangular multiplication with implicit transposition.
	 */
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N, 
		1.0, A, N,
		A1, N
	);

	/**
	 * Block Logic: Aggregate the two matrix product terms.
	 */
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += A1[i * N + j];
		}
	}
	free(B1);
	free(A1);
	return C;
}
