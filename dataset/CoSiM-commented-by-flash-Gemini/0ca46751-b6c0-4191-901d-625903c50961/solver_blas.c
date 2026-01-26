
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>


double* my_solver(int N, double *A, double *B) {

	double *prod11, *prod2;

	int i, j;

	
	prod11 = calloc(N * N, sizeof(double));
	if (prod11 == NULL)
		exit(EXIT_FAILURE);

	
	prod2 = calloc(N * N, sizeof(double));
	if (prod2 == NULL)
		exit(EXIT_FAILURE);

	
	memcpy(prod2, A, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N, 1.0, A, N, prod2, N
	);


	
	memcpy(prod11, B, N * N * sizeof(double));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N, 1.0, A, N, prod11, N
	);


	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N, 1.0, prod11, N, B, N, 1.0, prod2, N
	);

	free(prod11);

	return prod2;
}
