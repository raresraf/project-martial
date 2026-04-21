/**
 * @file solver_blas.c
 * @brief Matrix expression solver using standardized BLAS routines.
 *
 * Computes: Result = (A * B) * B^T + (A^T * A)
 * where A is an upper triangular matrix.
 *
 * Algorithm: Stage-based linear algebra calculation.
 * 1. Compute T1 = A * B using cblas_dtrmm.
 * 2. Compute T2 = A^T * A using cblas_dtrmm with transposition.
 * 3. Final accumulation T2 = T1 * B^T + T2 using cblas_dgemm.
 *
 * Domain: HPC, Linear Algebra, BLAS Optimization.
 */

#include "utils.h"
#include <cblas.h>
#include <string.h>

/**
 * my_solver - Implementation of the matrix expression via BLAS calls.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	/**
	 * Pre-condition: Allocation of temporary buffer X for product term B.
	 */
	double *X = (double*) calloc(N * N, sizeof(double));
	if (X == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    memcpy(X, B, N*N*sizeof(double));
	
	/**
	 * Block Logic: Compute B = A * B.
	 * Optimization: Exploits upper triangularity of A using DTRMM.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, B, N);

	/**
	 * Block Logic: Compute A = A^T * A.
	 * Optimization: Computes the symmetric product of a triangular matrix in-place.
	 */
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	/**
	 * Block Logic: Compute A = (B * X^T) + A.
	 * Functional Utility: Fuses the second multiplication and final addition.
	 */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, X, N, 1.0, A, N);
	
	double *Z = (double*) calloc(N * N, sizeof(double));
	if (Z == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
	memcpy(Z, A, N*N*sizeof(double));
	free(X);
	return Z;
}
