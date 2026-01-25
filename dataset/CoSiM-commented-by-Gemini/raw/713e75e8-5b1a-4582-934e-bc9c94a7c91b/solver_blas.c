
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	
	double *C = calloc(sizeof(double), N * N);
	double *a_copy = calloc(sizeof(double), N * N);

	memcpy(a_copy, A, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, a_copy, N, a_copy, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 0.0, C, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	cblas_daxpy(N * N, 1.0, a_copy, 1.0, C, 1.0);

	free(a_copy);
	
    return C;
}
