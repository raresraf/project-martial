
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	
	

	double *C = malloc(sizeof(double) * N * N);
	if (!C) 
		return NULL;

	double *AB = malloc(sizeof(double) * N * N);
	if (!AB) 
		return NULL;

	
	memmove(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	
	memmove(C, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);
	
	free(AB);
	return C;
}
