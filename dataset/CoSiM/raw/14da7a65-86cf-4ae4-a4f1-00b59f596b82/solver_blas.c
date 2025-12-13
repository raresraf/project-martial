
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	double *C, *AB, *ABB_t;
	int i, j;

	C = (double *)calloc(N * N, sizeof(double));
	if (NULL == C)
		exit(EXIT_FAILURE);
	
	AB = (double *)calloc(N * N, sizeof(double));
	if (NULL == AB)
		exit(EXIT_FAILURE);

	ABB_t = (double *)calloc(N * N, sizeof(double));
	if (NULL == ABB_t)
		exit(EXIT_FAILURE);

	
	memcpy(C, A, N * N * sizeof(double));

	cblas_dtrmm(
		CblasRowMajor,
		CblasLeft,
		CblasUpper,
		CblasTrans,
		CblasNonUnit,
		N, N,
		1.0, A, N,
		C, N
	);


	
	memcpy(AB, B, N * N * sizeof(double));

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
		1.0, AB, N,
		B, N, 
		0.0, ABB_t, N
	);

	
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			C[i * N + j] += ABB_t[i * N + j];
	
	free(AB);
	free(ABB_t);
	return C;
}

