
#include "utils.h"
#include <string.h>
#include "cblas.h"


void alloc_matrix(int N, double **C) {
		*C = malloc(N * N * sizeof(**C));
	if (NULL == *C)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C;
	int i;

	
	alloc_matrix(N, &C);

	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N, 1,
				B, N,
				B, N,
				0.0, C, N);
	
	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		C, N);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		A, N);

	
	for(i = 0; i < N * N; i++) {
			C[i] += A[i];
	}

	return C;
}
