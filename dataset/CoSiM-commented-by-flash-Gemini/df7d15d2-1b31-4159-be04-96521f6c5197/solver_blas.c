
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	
	double *C = calloc(sizeof(double), N * N);
	double *tmp2 = calloc(sizeof(double), N * N);
	memcpy(C, B, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	memcpy(tmp2, C, N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, tmp2, N, B, N, 0, C, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	cblas_daxpy(N * N, 1, A, 1, C, 1);

	free(tmp2);

	return C;
}
