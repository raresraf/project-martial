
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	double *C = (double *)calloc(N * N, sizeof(double));
	if (C == NULL) return NULL;

	double *result1 = (double *)calloc(N * N, sizeof(double));
	if (result1 == NULL) return NULL;

	double *result2 = (double *)calloc(N * N, sizeof(double));
	if (result2 == NULL) return NULL;

	for (int i = 0; i < N * N; i++) {
		result1[i] = B[i];
		result2[i] = A[i];
	}

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, 
		CblasNonUnit, N, N, 1, A, N, result1, N);

	
	cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1, A, N, result1, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1, A, N, result2, N);

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = result1[i * N + j] + result2[i * N + j];
		}
	}

	free(result1);
	free(result2);

	return C;
}
