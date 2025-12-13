
#include <stdlib.h>
#include "utils.h"
#include "cblas.h"




double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *C = (double *) malloc(N * N * sizeof(double));
	double *BBt = (double *) malloc(N * N * sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, BBt, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, BBt, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);
	int i;
	for (i = 0; i < N * N; ++i) {
		C[i] = BBt[i] + A[i];
	}

	return C;
}
