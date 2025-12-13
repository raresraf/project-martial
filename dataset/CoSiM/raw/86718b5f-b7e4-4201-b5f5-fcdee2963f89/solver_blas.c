
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {
	
	register int i = 0;
	register int j = 0;

	double *temp = (double *)malloc(N * N * sizeof(double));
	for(i = 0; i < N * N; i++)
                temp[i] = B[i];
	 
	cblas_dtrmm(CblasRowMajor, CblasLeft,
                CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1, A, N, temp, N);
	
	double *res1 = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, N, N, 1, temp, N, B, N, 0, res1, N);
	
	double *res2 = (double *)malloc(N * N * sizeof(double));
	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
				N, N, N, 1, A, N, A, N, 0, res2, N);

	for (i = 0; i < N; ++i) {
		register int in = i * N;
		for (j = 0; j < N; ++j) {
			res1[in + j] += res2[in + j];
		}
	}

	free(temp);
	free(res2);	
	return res1;
}
