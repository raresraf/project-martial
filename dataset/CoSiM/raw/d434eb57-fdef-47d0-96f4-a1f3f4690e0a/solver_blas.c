
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double *C = (double *)calloc(N * N, sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0,
	B, N, B, N, 0,C, N);
 
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
	CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper,
	CblasTrans, CblasNonUnit, N, N, 1.0, A, N, A, N);

	cblas_daxpy(N * N, 1, A, 1, C, 1);

	return C;
}
