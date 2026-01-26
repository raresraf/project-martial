
#include "utils.h"
#include <string.h>
#include <cblas.h>

 
double* my_solver(int N, double *A, double *B) {
	double *C = (double*)malloc(N * N * sizeof(double));
	double *D = (double*)malloc(N * N * sizeof(double));
	double *E = (double*)malloc(N * N * sizeof(double));
	
	memcpy(C, A, N * N * sizeof(double));
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, C, N);
	
	memcpy(D, B, N * N * sizeof(double));
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, D, N, B, N, 0.0, E, N);
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 
		1.0, A, N, E, N, 1.0, C, N);
	free(D);
	free(E);
	return C;
}
