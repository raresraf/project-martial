
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *AB = (double*) calloc(N * N, sizeof(double));

	
	cblas_dcopy(N * N, B, 1, AB, 1);

	
	cblas_dtrmm(CblasRowMajor, 
		CblasLeft, 
		CblasUpper, 
		CblasNoTrans,
		CblasNonUnit,
		N,
		N,
		1.0,
		A,
		N,
		AB,
		N);

	double *result = (double*) calloc(N * N, sizeof(double));

	
	cblas_dcopy(N * N, A, 1, result, 1);

	
	cblas_dtrmm(CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N,
		N, 
		1.0,
		A,
		N,
		result,
		N);
		
	
	cblas_dgemm(CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N,
		N,
		N,
		1.0,
		AB,
		N,
		B,
		N,
		1.0,
		result,
		N);

	free(AB);
	return result;
}
