
#include <string.h>
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	int i, j;
	double *C = malloc(N * N * sizeof(double));
	double *AUX = malloc(N * N * sizeof(double));

	
	
	memcpy(AUX, B, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0,
		A, N,
		AUX, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AUX, N,
		B, N,
		0.0,
		C, N
	);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0,
		A, N,
		A, N
	);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
				C[i * N + j] += A[i * N + j];
		} 
	}

	free(AUX);

	return C;
}
