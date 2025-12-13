
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double* res_AxB = (double*)malloc(N * N * sizeof(double));
	double* res_ABBt = (double*)malloc(N * N * sizeof(double));
	double* res_AtA = (double*)malloc(N * N * sizeof(double));
	double* res = (double*)malloc(N * N * sizeof(double));

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		N, N, N, 1, A, N, B, N, 0, res_AxB, N);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		N, N, N, 1, res_AxB, N, B, N, 0, res_ABBt, N);

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		N, N, N, 1, A, N, A, N, 0, res_AtA, N);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			res[i * N + j] = 0;
			res[i * N + j] += res_ABBt[i * N + j] + res_AtA[i * N + j];
		}
	}
	free(res_ABBt);
	free(res_AtA);
	free(res_AxB);
	return res;	
	
}
