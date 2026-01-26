
#include "utils.h"
#include <cblas.h>

double* my_solver(int N, double *A, double *B) {
	printf("BLAS SOLVER\n");
	double *C = malloc(sizeof(double) * N * N);
	double *aux = malloc(sizeof(double) * N * N);
	for (int i = 0; i < N*N; ++i) {
		C[i] = B[i];
		aux[i] = A[i];
	}
	
	
	cblas_dtrmm(	CblasRowMajor,
			CblasLeft,
			CblasUpper,
			CblasTrans,
			CblasNonUnit,
			N,
			N,
			1,
			A,
			N,
			aux,
			N
		);
	
	cblas_dtrmm(	CblasRowMajor,
			CblasLeft,
			CblasUpper,
			CblasNoTrans,
			CblasNonUnit,
			N,
			N,
			1,
			A,
			N,
			C,
			N
		);
	
	
	cblas_dgemm(	CblasRowMajor,
			CblasNoTrans,
			CblasTrans,
			N,
			N,
			N,
			1,
			C,
			N,
			B,
			N,
			1,
			aux,
			N
		);

	free(C);
	return aux;
}
