
/*
 * @raw/542bd872-56e9-492f-b5a7-0fa0401b6cdc/solver_blas.c
 * Module Level: Optimized matrix solver implementation using BLAS libraries.
 */
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	
	double *C1 = calloc(N * N, sizeof(double));

	
	cblas_dcopy(N * N, B, 1, C1, 1);

	
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N,
				1, A, N, C1, N);

	double *C2 = calloc(N * N, sizeof(double));

	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
				1, C1, N, B, N, 1, C2, N);

	double *C3 = calloc(N * N, sizeof(double));

	
	cblas_dcopy(N * N, A, 1, C3, 1);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N,
				1, A, N, C3, N);

	
	cblas_daxpy(N * N, 1, C2, 1, C3, 1);

	free(C1);
	free(C2);

	return C3;
}
