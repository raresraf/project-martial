
#include "utils.h"
#include "cblas.h"
#include <string.h>


void alloc_matr(int N, double **C, double **B_c)
{
	*C = malloc(N * N * sizeof(**C));
	if (!*C)
		exit(EXIT_FAILURE);

	*B_c = malloc(N * N * sizeof(**B_c));
	if (!*B_c)
		exit(EXIT_FAILURE);
}

double* my_solver(int N, double *A, double *B)
{
	printf("BLAS SOLVER\n");

	int i, j;
	double *B_c = NULL, *C = NULL;

	alloc_matr(N, &C, &B_c);

	memcpy(B_c, B, N * N * sizeof(*B_c));

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N,
	            N, 1.0, A, N, B, N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, B, N, B_c, N,
	            0, C, N);

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, N,
	            N, 1.0, A, N, A, N);

	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * N + j] += A[i * N + j];
		}
	}

	return C;
}
