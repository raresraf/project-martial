
#include "utils.h"
#include "cblas.h"

double* my_solver(int N, double *A, double *B) {

	double  *sum,*sum2, *c,  *un;
	int i,j;
	
	sum = calloc(N * N, sizeof(double));
	sum2 = calloc(N * N, sizeof(double));
	c = calloc(N * N,sizeof(double));
	un = calloc(N * N, sizeof(double));
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			sum[i * N + j] = B[i * N + j];
			if (i == j)
				un[i * N + j] = 1;
			else un[i * N + j] = 0;
		}
	}

	 
	cblas_dtrmm(CblasRowMajor, CblasLeft , CblasUpper, CblasNoTrans,
	CblasNonUnit, N, N, 1, A, N, sum, N);
	
	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1, sum, N, B, N, 1, sum2, N);

	
	cblas_dtrmm(CblasRowMajor, CblasLeft , CblasUpper, CblasTrans,
	CblasNonUnit, N, N, 1, A, N, A, N);

	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, sum2, N, un, N, 1, A, N);
	
	for (i = 0; i < N; i++){
		for (j = 0; j < N; j++){
			c[i * N + j] = A[i * N + j];
		}
	}
	free(sum);
	free(sum2);
	free(un);
	
	return c;
}
