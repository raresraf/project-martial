
#include "utils.h"
#include "cblas.h"
#include <string.h>

double* my_solver(int N, double *A, double *B) {
	double *ABBT, *ATA, *C;

	ABBT = calloc(N * N , sizeof(double));
	ATA = calloc(N * N , sizeof(double));
	C = calloc(N * N, sizeof(double));

	cblas_dgemm	(
		CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1,
		B, N,
		B, N,
		1, ABBT, N);

	cblas_dtrmm (
		CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		ABBT, N
	);

	memcpy(ATA, A, N * N * sizeof(double));
	cblas_dtrmm (
		CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
		N, N,
		1.0, A, N,
		ATA, N
	);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			C[i * N + j] = ABBT[i * N + j] + ATA[i * N + j];
		}
	}

	free(ABBT);
	free(ATA);

	return C;
}
