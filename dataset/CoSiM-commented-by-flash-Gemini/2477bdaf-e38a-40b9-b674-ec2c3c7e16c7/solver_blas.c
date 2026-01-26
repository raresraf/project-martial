
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "utils.h"
#include "cblas.h"



double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double* result_1 = (double *)calloc(N * N, sizeof(double));
	double* result_2 = (double *)calloc(N * N, sizeof(double));
	double* result_3 = (double *)calloc(N * N, sizeof(double));

	
	
	
	
	
	
	memcpy(result_1, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, result_1, N);	
	
	
	
	
	
	
	
	memcpy(result_3, A, N*N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, result_3, N);

	
	
	
	
	
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans,CblasTrans, N, N, N, 1.0, result_1, N, B, N, 1.0, result_3, N);
	
	free(result_1);
	free(result_2);
	return result_3;
}
