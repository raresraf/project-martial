
#include "utils.h"

#include <cblas.h>

double* my_solver(int N, double *A, double *B) {
	double *AB, *AtA, *C, *B2;

	
	AB = calloc(N * N, sizeof(double));
	if (AB == NULL)
		exit(-1);
	AtA = calloc(N * N, sizeof(double));
	if (AtA == NULL)
		exit(-1);
	C = calloc(N * N, sizeof(double));
	if (C == NULL)
		exit(-1);
	B2 = calloc(N * N, sizeof(double));
	if (B2 == NULL)
		exit(-1);

	
	cblas_dcopy(N * N, B, 1, B2, 1);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, B, N);

	
	cblas_dcopy(N * N, A, 1, AtA, 1);
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, AtA, N);

	
	cblas_dcopy(N * N, AtA, 1, C, 1);
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B2, N, 1, C, N);

	
    free(AB);
    free(AtA);
	free(B2);

	return C;
}
