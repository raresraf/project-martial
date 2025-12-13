
#include "utils.h"
#include "cblas.h"
#include "string.h"


double* my_solver(int N, double *A, double *B) {
	double *AxB = malloc(N * N * sizeof(double));
	double *A_copy = malloc(N * N * sizeof(double));
	double *C = malloc(N * N * sizeof(double));
	int i, j;

	
	memcpy(AxB, B, N * N * sizeof(double));
	memcpy(A_copy, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
			 CblasNonUnit, N, N, 1, A, N, AxB, N);
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
			 CblasNonUnit, N, N, 1, A, N, A_copy, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans,
			 CblasTrans, N, N, N, 1, AxB, N, B, N, 0, C, N);
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			*(A_copy + i * N + j) += *(C + i * N + j); 
	
	free(AxB);
	free(C);
	return A_copy;

}
