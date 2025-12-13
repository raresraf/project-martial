
#include "utils.h"
#include <cblas.h>
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	size_t i;
	size_t j;

	
	double *C = calloc(N * N, sizeof(double));

	if (C == NULL) {
		perror("Calloc C");
		exit(EXIT_FAILURE);
	}

	
	double *AB = calloc(N * N, sizeof(double));

	if (AB == NULL) {
		perror("Calloc AB");
		exit(EXIT_FAILURE);
	}

	
	double *P1 = calloc(N * N, sizeof(double));

	if (P1 == NULL) {
		perror("Calloc P1");
		exit(EXIT_FAILURE);
	}

	
	double *P2 = calloc(N * N, sizeof(double));

	if (P2 == NULL) {
		perror("Calloc P2");
		exit(EXIT_FAILURE);
	}

	
	memcpy(AB, B, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, AB, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, AB, N, B, N, 1.0, P1, N);

	
	memcpy(P2, A, N * N * sizeof(double));

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N, N, 1.0, A, N, P2, N);

	
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] = P1[i * N + j] + P2[i * N + j];
		}
	}

	free(AB);
	free(P1);
	free(P2);

	return C;
}
