
#include "utils.h"
#include "cblas.h"
#include "string.h"


double* my_solver(int N, double *A, double *B) {
	double* X = malloc(N * N * sizeof(double));
	if (X == NULL) {
		return NULL;
	}

	double* Y = malloc(N * N * sizeof(double));
	if (Y == NULL) {
		return NULL;
	}
	
	
	memcpy(Y, B, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		Y, N
	);

	
    memcpy(X, A, N * N * sizeof(double));
	
	
    cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		X, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1.0,
		Y, N, 
		B, N,
		1, X, N
	);

	free(Y);
	return X;
}
