/**
 * @file solver_blas.c
 * @brief LAPACK/BLAS optimized solver utilizing dgemm and dtrmm.
 * Computes matrix transformations using high-performance block operations.
 */
#include "utils.h"
#include "cblas.h"


/**
 * @brief Executes the solver using cblas primitives.
 * @param N Dimension of matrices A and B.
 * @param A Left hand input matrix.
 * @param B Right hand input matrix.
 * @return Allocated matrix C containing final computed results.
 */
double* my_solver(int N, double *A, double *B) {

	double *AB = calloc(N * N, sizeof(double));
	if(AB == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	double *C = calloc(N * N, sizeof(double));
	if(C == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	double *A_TA = calloc(N * N, sizeof(double));
	if(A_TA == NULL) {
		perror("Calloc failed!");
		exit(ENOMEM);
	}

	
	memcpy(AB, B, N * N * sizeof(double));

	
	memcpy(A_TA, A, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AB, N
	);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		A_TA, N
	);

	
	memcpy(C, A_TA, N * N * sizeof(double));

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1,
		AB, N,
		B, N,
		1, C, N);

	free(A_TA);
	free(AB);

	return C;
}
