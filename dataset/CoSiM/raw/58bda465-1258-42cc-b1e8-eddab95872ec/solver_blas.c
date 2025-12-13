
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	double *AB = calloc(N * N, sizeof(double));
	double *ABBt = calloc(N * N, sizeof(double));
	double *AtA = calloc(N * N, sizeof(double));
	double *C = malloc(N * N * sizeof(double));

	if (AB == NULL || ABBt == NULL || AtA == NULL || C == NULL){
		fprintf(stderr, "malloc error\n");
		exit(EXIT_FAILURE);
	}

	
	memcpy(AB, B, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
		CblasNonUnit, N, N, 1.0, A, N, AB, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N,
		1.0, AB, N, B, N, 1.0, ABBt, N);

	
	memcpy(AtA, A, N * N * sizeof(double));
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
		CblasNonUnit, N, N, 1.0, A, N, AtA, N);

	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i * N + j] = ABBt[i * N + j] + AtA[i * N + j];
		}
	}

	
	free(AB);
	free(ABBt);
	free(AtA);

	return C;
}
