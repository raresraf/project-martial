
#include "utils.h"
#include "cblas.h"
#include "string.h"


double* my_solver(int N, double *A, double *B) {
	double *C = malloc(N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, B, N, B, N, 0.0, C, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	
	cblas_daxpy(N * N, 1.0, A, 1.0, C, 1.0);

	return C;
}
