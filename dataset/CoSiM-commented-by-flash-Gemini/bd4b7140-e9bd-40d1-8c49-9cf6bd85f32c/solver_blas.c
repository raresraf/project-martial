
#include "utils.h"
#include <string.h>
#include <stdlib.h>

#include "cblas.h"

double* my_solver(int N, double *A, double *B) {

	double *AtA_copy;
	AtA_copy = malloc(N * N * sizeof(double));
	memcpy(AtA_copy, A, N * N * sizeof(double));

	double *D_copy;
	D_copy = malloc(N * N * sizeof(double));
	memcpy(D_copy, B, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N,
				N,
				1.0,
				A, N,
				AtA_copy, N);

	
	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N,
				N,
				1.0,
				A, N,
				D_copy, N);

	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans, 
				CblasTrans,
				N,
				N, 
				N, 
				1.0,
				D_copy,
				N,
				B, 
				N,
				1.0,
				AtA_copy,
				N
				);

	free(D_copy);	
	return AtA_copy;
}
