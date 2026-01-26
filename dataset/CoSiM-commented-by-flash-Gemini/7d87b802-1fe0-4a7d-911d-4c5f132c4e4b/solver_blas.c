
#include "utils.h"
#include "cblas.h"


double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");

	double c[N][N], d[N][N];
	
	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			c[i][j] = B[i * N + j];
		}
	}

	double *e = malloc(N * N * sizeof(double));

	double *orig_pc = &c[0][0];
	double *orig_pd = &d[0][0];

	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, N, N, 1.0, A, N, orig_pc, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
                N, N, N, 1.0, orig_pc, N, B, N, 0.0, orig_pd, N);

	
	for(int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			e[i * N + j] = A[i * N + j];
		}
	}
	
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                N, N, 1.0, A, N, e, N);

	
	cblas_daxpy(N * N, 1.0, orig_pd, 1, e, 1);	
	
	return e;
}
