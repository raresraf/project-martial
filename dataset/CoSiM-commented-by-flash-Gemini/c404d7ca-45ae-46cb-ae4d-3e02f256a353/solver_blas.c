
#include "utils.h"
#include "cblas.h"
#include <string.h>


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	
	int i, j;
	int size = N * N * sizeof(double);
	double *C, *AB, *AtA;

	C = (double *) calloc(1, size);

	if (C == NULL)
		exit(-1);

	AB = (double *) calloc(1, size);

	if (AB == NULL)
		exit(-1);

	AtA = (double *) calloc(1, size);

	if (AtA == NULL)
		exit(-1);

	
	memcpy(AB, B, size);

	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasNoTrans,
				CblasNonUnit,
				N, N,
				1.0, A, N,
				AB, N);

	
	cblas_dgemm(CblasRowMajor,
				CblasNoTrans,
				CblasTrans,
				N, N, N,
				1.0, AB, N,
				B, N,
				1.0, C, N);

	
	memcpy(AtA, A, size);

	cblas_dtrmm(CblasRowMajor,
				CblasLeft,
				CblasUpper,
				CblasTrans,
				CblasNonUnit,
				N, N,
				1.0, A, N,
				AtA, N);

	for (i = 0; i < N; ++i)
		for (j = 0; j < N; ++j)
			C[i * N + j] += AtA[i * N + j];

	
	free(AB);
	free(AtA);

	return C;
}
