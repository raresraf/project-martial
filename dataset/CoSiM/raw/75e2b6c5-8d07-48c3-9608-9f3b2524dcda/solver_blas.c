

#include <stdio.h>
#include <string.h>

#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	register int size = N * N * sizeof(double);

	
	double *BB = malloc(size);
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1, B, N,
		B, N, 0,
		BB, N
	);

	double *AB = malloc(size);
	memcpy(AB, BB, size);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1, A, N,
		AB, N
	);

	
	double *C = malloc(size);
	memcpy(C, AB, size);

	cblas_dgemm(
		CblasRowMajor,
		CblasTrans,
		CblasNoTrans,
		N, N, N,
		1, A, N,
		A, N, 1,
		C, N
	);


	free(AB);
	free(BB);

	return C;
}
