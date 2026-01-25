
#include "utils.h"
#include <cblas.h>
#include <string.h>

#define ALPHA 1.0


double* my_solver(int N, double *A, double *B) {
	double *C = (double*) calloc(N * N, sizeof(double));
	if (!C)
		return NULL;

	double *aux = (double*) calloc(N * N, sizeof(double));
	if (!aux)
		return NULL;

	memcpy(aux, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, ALPHA, A, N, aux, N);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, ALPHA, aux, N, B, N, ALPHA, C, N);

	memcpy(aux, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, ALPHA, A, N, aux, N);

	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += aux[i * N + j];
		}
	}

	free(aux);


	return C;
}
