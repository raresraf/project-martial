
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	double *AB, *C, *AtA;

	AB = malloc(N * N * sizeof(double));
	memcpy(AB, B, N * N * sizeof(double));
	C = calloc(N * N, sizeof(double));
	AtA = malloc(N * N * sizeof(double));
	memcpy(AtA, A, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit, 
		N, N,
		1.0,
		A, N, AB, N);
	
	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit, 
		N, N,
		1.0,
		A, N, AtA, N);
	
	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AB, N,
		B, N, 1.0,
		AtA, N);

	memcpy(C, AtA, N * N * sizeof(double));

	free(AB);
	free(AtA);
	return C;
}
