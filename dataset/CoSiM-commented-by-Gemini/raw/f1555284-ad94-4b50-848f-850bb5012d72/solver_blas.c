
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {

	double *D = (double *)malloc(N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor, 
	CblasNoTrans,
	CblasTrans,
	N, N, N, 1, B,
	N, B, N, 0, D, N
	);

	
	cblas_dtrmm(CblasRowMajor,
	CblasLeft,
	CblasUpper,
	CblasNoTrans,
	CblasNonUnit,
	N, N, 1, A, N, D, N);

	
	cblas_dtrmm(CblasRowMajor,
	CblasLeft,
	CblasUpper,
	CblasTrans,
	CblasNonUnit,
	N, N, 1, A, N, A, N);
	
	
	for (int i = 0; i < N * N; i++) {
		D[i] += A[i];
	}

   	return D;
}
