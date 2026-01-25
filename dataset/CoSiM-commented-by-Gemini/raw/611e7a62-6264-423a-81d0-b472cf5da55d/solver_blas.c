
#include "utils.h"
#include "cblas.h"
#include <stdlib.h>
#include <string.h>

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *B1;
	double *C;
	double *A1;
	int i, j;
	B1 = calloc(N*N, sizeof(double));
	C = calloc(N*N, sizeof(double));
	A1 = calloc(N*N, sizeof(double));
	memcpy(B1, B, N*N * sizeof(double));
	memcpy(A1, A, N*N * sizeof(double));

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

	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N,
		N, 1.0,
		B1, N, B,
		N, 0.0, C, N
	);

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

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += A1[i * N + j];
		}
	}
	free(B1);
	free(A1);
	return C;
}
