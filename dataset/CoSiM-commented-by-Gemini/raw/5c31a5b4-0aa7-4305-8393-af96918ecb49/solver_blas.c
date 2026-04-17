/**
 * @file solver_blas.c
 * @brief BLAS-based matrix solver implementation utilizing optimized LAPACK/BLAS routines.
 * Leverages high-performance matrix multiplications for calculating the matrix operations sequence.
 */
#include "utils.h"
#include "cblas.h"
#include <string.h>


/**
 * @brief Executes the BLAS-accelerated matrix equation solver.
 * @param N The dimension of the square matrices A and B.
 * @param A Pointer to the first input matrix.
 * @param B Pointer to the second input matrix.
 * @return A dynamically allocated array containing the result matrix Z.
 */
double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	/**
	 * @pre N > 0, defining matrix dimension N x N.
	 * @post X is allocated with N * N doubles and zero-initialized.
	 */
	double *X = (double*) calloc(N * N, sizeof(double));
	if (X == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
    memcpy(X, B, N*N*sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, B, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, X, N, 1.0, A, N);
	
	/**
	 * @pre X and A contain intermediate matrix products.
	 * @post Z is allocated and stores the final result duplicated from A.
	 */
	double *Z = (double*) calloc(N * N, sizeof(double));
	if (Z == NULL) {
		printf("Calloc failed!\n");
      	exit(1);
	}
	memcpy(Z, A, N*N*sizeof(double));
	free(X);
	return Z;
}
