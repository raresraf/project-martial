
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C = calloc(N * N, sizeof(double));
	double *AB = calloc(N * N, sizeof(double));

	
	memcpy(AB, B, N * N * sizeof(double));
	
	memcpy(C, A, N * N * sizeof(double));	

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, C, N);

	free(AB);
	
	return C;
}
