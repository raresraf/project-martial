
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <cblas.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *Bt;
	double *C;
	
	Bt = calloc(N * N, sizeof(double));
	C = calloc(N * N, sizeof(double));
	
	
	memcpy(C, A, N * N * sizeof(double));
	memcpy(Bt, B, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		B, N
	);
	
	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, C, N,
		C, N
	);
	
	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, B, N,
		Bt, N,
		1.0, C, N
	);
	
	free(Bt);
	
	return C;
}
