
#include "utils.h"
#include "cblas.h"
#include <string.h>



double* my_solver(int N, double *A, double *B) {
	int i, j;
	double *ABBt, *result, *AtA;
	ABBt = malloc(N * N * sizeof(double));
	AtA = malloc(N * N * sizeof(double));
	result = malloc(N * N * sizeof(double));

	memcpy(AtA, A, N * N * sizeof(double));

	cblas_dgemm(
		CblasRowMajor, CblasNoTrans,
		CblasNoTrans, N, N, N, 1.0, A,
		 N, B, N, 0.0, result, N);
	cblas_dgemm(
		CblasRowMajor, CblasNoTrans,
		CblasTrans, N, N, N, 1.0, result, 
		N, B, N, 0.0, ABBt, N);
	cblas_dtrmm( CblasRowMajor,
		CblasLeft, CblasUpper,
		CblasTrans, CblasNonUnit,
		N, N, 1.0, A, N, AtA, N
	);

	for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            result[i * N + j]  = ABBt[i * N + j] + AtA[i * N + j];
        }
    }

    free(AtA);
	free(ABBt);

	return result;

}