
#include <string.h>
#include <stdlib.h>
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	int i, j;
	double *C = calloc(N * N, sizeof(*C));
	if (C == NULL) {
		exit(EXIT_FAILURE);
	}

	double *AB = calloc(N * N, sizeof(*AB ));
	if (AB == NULL) {
		exit(EXIT_FAILURE);
	}


	double *A_tA = calloc(N * N, sizeof(*A_tA));
	if (A_tA == NULL) {
		exit(EXIT_FAILURE);
	}

	
	memcpy(C, B, N * N * sizeof(*B));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasNoTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N
	);

	
	memcpy(AB, C, N * N * sizeof(*C));

	cblas_dgemm(
		CblasRowMajor,
		CblasNoTrans,
		CblasTrans,
		N, N, N,
		1.0, AB, N,
		B, N, 0.0, C, N
	);

	
	memcpy(A_tA, A, N * N * sizeof(*A));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		A_tA, N
	);

	
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i * N + j] += A_tA[i * N + j];
		}
	}

	free(A_tA);
	free(AB);

	return C;
}
