
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	int i = 0;
	int j = 0;

	
	double *result1 = (double *)malloc(N * N * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                N, N, N, 1, A, N, B, N, 0, result1, N);

	double *result2 = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, result1, N, B, N, 0, result2, N);
	
	double *res = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, res, N);

	for (i = 0; i < N; ++i) {
		register int in = i * N;
		for (j = 0; j < N; ++j) {
			res[in + j] += result2[in + j];
		}
	}

	free(result1);
	free(result2);	
	return res;
}
