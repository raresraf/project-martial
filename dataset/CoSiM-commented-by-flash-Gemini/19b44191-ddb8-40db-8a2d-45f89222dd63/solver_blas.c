
#include "utils.h"
#include <cblas.h>

double* my_solver(int N, double *A, double *B) {
	double *C, *rez;
	int i, j;
	C = calloc(N * N, sizeof(double));
	rez = calloc(N * N, sizeof(double));

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B, N, 0, rez, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, rez, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1, A, N, A, N);

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] = A[i * N + j] + rez[i * N + j];
		}
	}
	free(rez);
	return C;
}
