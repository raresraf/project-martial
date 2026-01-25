
#include "utils.h"
#include<cblas.h>

double* my_solver(int N, double *A, double *B) {
	
	double *rez, *ident;
	int i;

	rez = (double*) calloc(N * N, sizeof(double));
	if (!rez)
		return NULL;
	
	
	ident = (double*) calloc(N * N, sizeof(double));
	if (!ident)
		return NULL;
	for (i = 0; i < N; i++)
		ident[i * N + i] = 1;

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, rez, N);	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, rez, N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, A, N, ident, N, 1, rez, N);
	
	free(ident); 
	
	return rez;
}
