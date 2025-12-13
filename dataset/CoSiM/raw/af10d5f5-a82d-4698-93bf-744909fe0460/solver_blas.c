
#include <cblas.h>
#include <string.h>
#include "utils.h"

double *allocate_matrix(int N) {
	double *res = calloc(N * N, sizeof(double));
	return res;
}


double* my_solver(int N, double *A, double *B) {
	double *C = allocate_matrix(N);
	double *AxB = allocate_matrix(N);
	double *AxBxBt = allocate_matrix(N);
	double *AtxA = allocate_matrix(N);
    int i, j;
	memcpy(AxB, B, sizeof(double) * N * N);
	
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, AxB, N);
   	
	cblas_dgemm(CblasRowMajor ,CblasNoTrans, CblasTrans, N, N, N, 1.0, AxB, N, B, N, 0.0, AxBxBt, N);
	memcpy(AtxA, A, sizeof(double) * N * N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, AtxA, N);
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = AxBxBt[i * N + j] + AtxA[i * N + j];
		}
	}	
	free(AxB);
	free(AxBxBt);
	free(AtxA);
	return C;
}
