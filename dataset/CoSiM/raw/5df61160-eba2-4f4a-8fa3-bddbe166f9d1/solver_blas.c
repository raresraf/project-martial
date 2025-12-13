
#include "utils.h"
#include "cblas.h"
#include "string.h"


double* my_solver(int N, double *A, double *B) {
	
	double *TEMPORARY = calloc(N * N, sizeof(double));
	double *RESULT = calloc(N * N, sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			N, N, N, 1, A, N, B, N, 0, TEMPORARY, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			N, N, N, 1, TEMPORARY, N, B, N, 0, RESULT, N);

	
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
			N, N, N, 1, A, N, A, N, 1, RESULT, N);

	free(TEMPORARY);
	return RESULT;
}
