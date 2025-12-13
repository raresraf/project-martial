
#include "utils.h"
#include "cblas.h"
#include <string.h>

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C, *left;
	C =(double *)malloc(N * N * sizeof(double));
	if (C == NULL)
		exit(-1);
	left =(double *)malloc(N * N * sizeof(double));
	if (left == NULL)
		exit(-1);
	memcpy(left, B, N * N * sizeof(double));
	memcpy(C, A, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, left, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, C, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, left, N, B, N, 1.0, C, N);

	free(left);

	return C;
}
