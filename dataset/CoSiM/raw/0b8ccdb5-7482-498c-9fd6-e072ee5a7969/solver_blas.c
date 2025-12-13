

#include <string.h>

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	
	int i,j;
	double *C = (double *)calloc(N * N, sizeof(double));
	double *D = (double *)malloc(N * N * sizeof(double));
	memcpy(D, A, N * N * sizeof(double));
	
	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N,
		N,
		N,
		1.0,
		B,
		N,
		B,
		N,
		1.0,
		C,
		N
	);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N,
		N,
		1.0,
		A,
		N,
		C,
		N
	);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N,
		N,
		1.0, 
		A,
		N,
		D,
		N
	);

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			D[i * N + j] += C[i * N + j];

	free(C);
	return NULL;
}
