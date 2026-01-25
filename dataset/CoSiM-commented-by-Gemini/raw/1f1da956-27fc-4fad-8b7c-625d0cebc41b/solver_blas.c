
#include <string.h>
#include "utils.h"
#include "cblas.h"

void addition(double *C, double *A, double *B, int N) {
	for (register int i = 0; i < N; ++i) {
		for (register int j = 0; j < N; ++j) {
			C[i * N + j] = *A + *B;
			++A;
			++B;
		}
	}
}

double* my_solver(int N, double *A, double* B) {
	double *C = calloc(N * N, sizeof(*C));
	if (!C) {
		exit(-1);
	}

	double *AB = malloc(N * N * sizeof(*AB));
	if (!AB) {
		exit(-1);
	}
	memcpy(AB, B, N * N * sizeof(*B));

	double *ABBt = calloc(N * N, sizeof(*ABBt));
	if (!ABBt) {
		exit(-1);
	}

	double *AtA = malloc(N * N * sizeof(*AtA));
	if (!AtA) {
		exit(-1);
	}
	memcpy(AtA, A, N * N * sizeof(*A));

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AB, N
	);

	
	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0,
		AB, N,
		B, N,
		0.0,
		ABBt, N);

	
	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		AtA, N
	);

	addition(C, ABBt, AtA, N);

	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
