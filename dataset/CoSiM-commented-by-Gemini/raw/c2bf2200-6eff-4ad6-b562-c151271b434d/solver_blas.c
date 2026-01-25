
#include "utils.h"

#include <cblas.h>
#include <stdlib.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *AB = malloc(N * N * sizeof(double));
	double *ABBt = malloc(N * N * sizeof(double));

	
	memcpy(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	
	memcpy(ABBt, A, N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, ABBt, N);

	free(AB);

	return ABBt;
}
